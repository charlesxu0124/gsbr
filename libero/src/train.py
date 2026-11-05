"""Train a frozen pi0.5 policy with an outer RL policy on LIBERO.

Run: python -m src.train --config-name=gsbr_t14 seed=0
"""

from copy import deepcopy
import json
import os
import re
import signal
import time
from typing import Tuple
import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from gymnasium import Env
from src.outer_policies.base import RLOuterPolicy, SamplingMode
from src.utils.rollout import Rollout
from src.utils.rl_performance_tracker import PerformanceTracker
from src.utils.task_throughput import (
    episode_env_steps,
    control_frequency,
    task_throughput,
)

_TERMINATE_REQUESTED = False


def _register_sigterm_handler():
    """Request a safe-boundary checkpoint on SIGTERM.

    A second SIGTERM uses the default action, which keeps an unhealthy process
    from ignoring the process supervisor indefinitely if graceful shutdown itself gets stuck.
    Returns the previous handler so tests/embedding callers can restore it.
    """
    global _TERMINATE_REQUESTED
    _TERMINATE_REQUESTED = False
    previous_handler = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum, frame):
        del signum, frame
        global _TERMINATE_REQUESTED
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        _TERMINATE_REQUESTED = True
        print(
            "[train] received SIGTERM; checkpointing at the next safe boundary.",
            flush=True,
        )

    signal.signal(signal.SIGTERM, _handle_sigterm)
    return previous_handler


def _make_dirs(cfg: DictConfig, outer_policy: RLOuterPolicy) -> dict:
    run_dir = os.path.join(cfg.results_dir, type(outer_policy).__name__, cfg.exp_name)
    dirs = {
        "run": run_dir,
        "checkpoints": os.path.join(run_dir, "checkpoints"),
        "videos": os.path.join(run_dir, "videos"),
        "eval": os.path.join(run_dir, "eval"),
        "rollouts": os.path.join(run_dir, "rollouts"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def _save_stats(
    path: str, total_trajs: int, total_env_steps: int, total_grad_steps: int
) -> None:
    from nicerl.utils.atomic_io import atomic_pickle

    atomic_pickle(
        {
            "total_trajs": total_trajs,
            "total_env_steps": total_env_steps,
            "total_grad_steps": total_grad_steps,
        },
        path,
    )


def _load_stats(path: str) -> Tuple[int, int, int]:
    if not os.path.exists(path):
        print(f"[train] no stats at {path}; starting from zeroed stats.")
        return (0, 0, 0)
    from nicerl.utils.atomic_io import load_pickle

    d = load_pickle(path)
    return (d["total_trajs"], d["total_env_steps"], d["total_grad_steps"])


def _save_training_checkpoint(
    outer_policy: RLOuterPolicy,
    perf: PerformanceTracker,
    checkpoint_dir: str,
    perf_path: str,
    stats_path: str,
    total_trajs: int,
    total_env_steps: int,
    total_grad_steps: int,
    reason: str,
) -> None:
    """Persist all state needed for an exact episode-boundary resume."""
    outer_policy.save_state(checkpoint_dir)
    perf.save(perf_path)
    _save_stats(stats_path, total_trajs, total_env_steps, total_grad_steps)
    print(
        f"[train] checkpoint saved ({reason}): trajectories={total_trajs} env_steps={total_env_steps} grad_steps={total_grad_steps}",
        flush=True,
    )


def _wandb_resume_id(run_dir: str) -> str:
    """Return a stable W&B id, including for runs made before this helper."""
    id_path = os.path.join(run_dir, "wandb_run_id.txt")
    if os.path.exists(id_path):
        with open(id_path) as file:
            run_id = file.read().strip()
        if run_id:
            return run_id
    latest = os.path.join(run_dir, "wandb", "latest-run")
    match = re.fullmatch(
        "run-\\d{8}_\\d{6}-(.+)", os.path.basename(os.path.realpath(latest))
    )
    if match:
        run_id = match.group(1)
    elif hasattr(wandb.util, "generate_id"):
        run_id = wandb.util.generate_id()
    else:
        from wandb.sdk.lib.runid import generate_id

        run_id = generate_id()
    temp_path = f"{id_path}.tmp.{os.getpid()}"
    with open(temp_path, "w") as file:
        file.write(f"{run_id}\n")
    os.replace(temp_path, id_path)
    return run_id


def _log_metrics(metrics: dict, step: int, path: str) -> None:
    """Log to wandb and mirror to a JSONL file in the run dir.

    The file makes runs inspectable without the wandb UI (tail/grep/jq); one
    line per log call: {"step": ..., "time": <unix seconds>, **metrics}.
    """
    wandb.log(metrics, step=step)
    with open(path, "a") as f:
        f.write(
            json.dumps(
                {"step": step, "time": round(time.time(), 1), **metrics}, default=float
            )
            + "\n"
        )


def _maybe_save_video(rollout: Rollout, path: str, fps: int) -> None:
    """Best-effort episode video from agentview frames; never fatal."""
    try:
        import numpy as np
        import imageio

        if "all_frames" in rollout.infos[0]:
            frames = []
            for info in rollout.infos:
                frames.extend(info["all_frames"])
        else:
            possible_keys = ["agentview_image"]
            found_key = next(
                (k for k in possible_keys if k in rollout.next_observations[0]), None
            )
            if found_key is None:
                print(f"[train] no frames found in rollout; skipping video ({path})")
                return
            frames = [np.asarray(o[found_key]) for o in rollout.next_observations]
        imageio.mimsave(path, frames, fps=fps)
    except Exception as exc:
        print(f"[train] skipped video ({path}): {exc}")


def _select_mode(traj_idx: int, cfg: DictConfig) -> SamplingMode:
    return (
        SamplingMode.RANDOM
        if traj_idx < cfg.train.learning_starts
        else SamplingMode.TRAIN
    )


def _should_eval(traj_idx: int, cfg: DictConfig) -> bool:
    return traj_idx >= cfg.train.learning_starts and traj_idx % cfg.eval.eval_every == 0


def collect_episode(
    env: Env, outer_policy: RLOuterPolicy, mode: SamplingMode
) -> Rollout:
    """Roll out one episode and return it."""
    outer_policy.reset()
    obs, _ = env.reset()
    rollout = Rollout()
    done = False
    while not done:
        action_chunk = outer_policy.sample(obs, mode=mode)
        next_obs, reward, terminated, truncated, info = env.step(action_chunk)
        rollout.add_step(
            obs, action_chunk, reward, next_obs, terminated, truncated, info
        )
        obs = deepcopy(next_obs)
        done = terminated or truncated
    return rollout


def run_eval(
    cfg: DictConfig, env: Env, outer_policy: RLOuterPolicy, traj_idx: int, eval_dir: str
) -> dict:
    save_dir = os.path.join(eval_dir, f"train_step={traj_idx}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    successes = []
    rewards = []
    chunks = []
    env_steps = []
    for eval_idx in tqdm(range(cfg.eval.eval_num_trajs), desc="Eval"):
        rollout = collect_episode(env, outer_policy, SamplingMode.EVAL)
        successes.append(float(bool(rollout.is_success)))
        rewards.append(sum(rollout.rewards))
        chunks.append(len(rollout.rewards))
        env_steps.append(episode_env_steps(env, len(rollout.rewards)))
        if eval_idx < cfg.eval.eval_num_videos:
            _maybe_save_video(
                rollout,
                os.path.join(
                    save_dir, f"traj={eval_idx}_succ={rollout.is_success}.mp4"
                ),
                fps=cfg.env.fps,
            )
    return {
        "success_rate": sum(successes) / max(len(successes), 1),
        "avg_reward": sum(rewards) / max(len(rewards), 1),
        "avg_chunks": sum(chunks) / max(len(chunks), 1),
        "avg_env_steps": sum(env_steps) / max(len(env_steps), 1),
        "episode_length": sum(env_steps) / max(len(env_steps), 1),
        "throughput": task_throughput(
            successes, env_steps, control_frequency(env, cfg.env.fps)
        )
        if successes
        else 0.0,
        "control_hz": control_frequency(env, cfg.env.fps),
    }


def run_training(cfg: DictConfig, env: Env, outer_policy: RLOuterPolicy) -> None:
    dirs = _make_dirs(cfg, outer_policy)
    stats_path = os.path.join(dirs["checkpoints"], "stats.pkl")
    perf_path = os.path.join(dirs["checkpoints"], "performance_tracker.pkl")
    metrics_path = os.path.join(dirs["run"], "metrics.jsonl")
    example_obs, _ = env.reset()
    outer_policy.setup(example_obs)
    total_trajs, total_env_steps, total_grad_steps = (0, 0, 0)
    perf = PerformanceTracker()
    if cfg.do_restore:
        outer_policy.load_state(dirs["checkpoints"])
        total_trajs, total_env_steps, total_grad_steps = _load_stats(stats_path)
        perf.load(perf_path)
    wandb_resume_id = _wandb_resume_id(dirs["run"]) if cfg.do_restore else None
    run = wandb.init(
        name=cfg.exp_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=dirs["run"],
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.wandb.group,
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        id=wandb_resume_id,
        resume="allow" if wandb_resume_id is not None else None,
    )
    max_num_env_steps = cfg.train.get("max_num_env_steps", None)
    completed_trajs = total_trajs
    last_checkpoint_trajs = total_trajs
    terminated_for_signal = False
    try:
        for traj_idx in tqdm(
            range(total_trajs, cfg.train.max_num_trajs), desc="train step"
        ):
            if _TERMINATE_REQUESTED:
                terminated_for_signal = True
                break
            if max_num_env_steps is not None and total_env_steps >= max_num_env_steps:
                print(
                    f"[train] reached max_num_env_steps={max_num_env_steps}; stopping."
                )
                break
            mode = _select_mode(traj_idx, cfg)
            if _should_eval(traj_idx, cfg):
                eval_results = run_eval(cfg, env, outer_policy, traj_idx, dirs["eval"])
                _log_metrics(
                    {f"eval/{k}": v for k, v in eval_results.items()},
                    step=traj_idx,
                    path=metrics_path,
                )
                if _TERMINATE_REQUESTED:
                    terminated_for_signal = True
                    break
            episode_start = time.perf_counter()
            rollouts, _ = outer_policy.collect_experience(env, mode)
            missing = {"rl_rollout", "rollout"} - set(rollouts)
            if missing:
                raise KeyError(
                    f"collect_experience must return rollouts under 'rl_rollout' (metrics) and 'rollout' (video frames); missing {sorted(missing)} from keys {sorted(rollouts)}"
                )
            rl_rollout = rollouts["rl_rollout"]
            if cfg.do_restore:
                for key, rollout in rollouts.items():
                    if key != "rl_rollout":
                        continue
                    key_dir = os.path.join(dirs["rollouts"], key)
                    os.makedirs(key_dir, exist_ok=True)
                    rollout.save_to_disk(os.path.join(key_dir, f"{traj_idx}.pkl"))
            episode_len = len(rl_rollout.rewards)
            total_env_steps += episode_len
            perf.update(
                success=int(bool(rl_rollout.is_success)),
                reward=float(sum(rl_rollout.rewards)),
                episode_length=episode_env_steps(env, episode_len),
                control_hz=control_frequency(env, cfg.env.fps),
            )
            _maybe_save_video(
                rollouts["rollout"],
                os.path.join(
                    dirs["videos"], f"traj={traj_idx}_succ={rl_rollout.is_success}.mp4"
                ),
                fps=cfg.env.fps,
            )
            if mode == SamplingMode.TRAIN:
                num_grad_steps = episode_len * cfg.train.grad_steps_per_env_step
                info = outer_policy.update(num_grad_steps)
                if info is not None:
                    total_grad_steps += num_grad_steps
                if info:
                    _log_metrics(
                        {f"training/{k}": v for k, v in info.items()},
                        step=traj_idx,
                        path=metrics_path,
                    )
            completed_trajs = traj_idx + 1
            episode_time_s = time.perf_counter() - episode_start
            perf.training_time_s += episode_time_s
            perf.timed_training_episodes += 1
            _log_metrics(
                {
                    "status/num_trajs": traj_idx,
                    "status/env_steps": total_env_steps,
                    "status/grad_steps": total_grad_steps,
                    "timing/train_episode_s": episode_time_s,
                    "timing/training_time_s": perf.training_time_s,
                    "timing/timed_training_episodes": perf.timed_training_episodes,
                    "performance/episode_chunks": episode_len,
                    **{f"performance/{k}": v for k, v in perf.get_metrics().items()},
                },
                step=traj_idx,
                path=metrics_path,
            )
            if traj_idx % cfg.train.save_every == 0 and cfg.do_restore:
                _save_training_checkpoint(
                    outer_policy,
                    perf,
                    dirs["checkpoints"],
                    perf_path,
                    stats_path,
                    completed_trajs,
                    total_env_steps,
                    total_grad_steps,
                    reason="periodic",
                )
                last_checkpoint_trajs = completed_trajs
            if _TERMINATE_REQUESTED:
                terminated_for_signal = True
                break
        if (
            not terminated_for_signal
            and completed_trajs >= cfg.train.max_num_trajs
            and cfg.eval.get("final_eval", True)
        ):
            eval_results = run_eval(
                cfg, env, outer_policy, cfg.train.max_num_trajs, dirs["eval"]
            )
            _log_metrics(
                {f"eval/{k}": v for k, v in eval_results.items()},
                step=cfg.train.max_num_trajs,
                path=metrics_path,
            )
        if cfg.do_restore and last_checkpoint_trajs != completed_trajs:
            _save_training_checkpoint(
                outer_policy,
                perf,
                dirs["checkpoints"],
                perf_path,
                stats_path,
                completed_trajs,
                total_env_steps,
                total_grad_steps,
                reason="SIGTERM" if terminated_for_signal else "final",
            )
    finally:
        env.close()
        run.finish()
    if terminated_for_signal:
        print(
            "[train] graceful SIGTERM shutdown complete; the process supervisor may now requeue.",
            flush=True,
        )


@hydra.main(config_path="cfgs", config_name="gsbr_t14", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    env: Env = instantiate(cfg.env, _convert_="all")
    outer_policy: RLOuterPolicy = instantiate(cfg.outer_policy, _convert_="all")
    dirs = _make_dirs(cfg, outer_policy)
    OmegaConf.save(cfg, os.path.join(dirs["run"], "config.yaml"))
    previous_sigterm_handler = _register_sigterm_handler()
    try:
        run_training(cfg, env, outer_policy)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


if __name__ == "__main__":
    main()
