"""Evaluate the frozen pi0.5 base policy on one LIBERO task."""

import json
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from gymnasium import Env
from src.outer_policies.base import OuterPolicy, SamplingMode
from src.train import collect_episode, _maybe_save_video
from src.utils.stats import calculate_beta_posterior_percentile


def _run_dir(cfg: DictConfig, outer_policy_cls: type, env_cls: type) -> str:
    return os.path.join(cfg.results_dir, "rollout", env_cls.__name__, cfg.exp_name)


def _make_dirs(run_dir: str) -> dict:
    dirs = {
        "run": run_dir,
        "videos": os.path.join(run_dir, "videos"),
        "rollouts": os.path.join(run_dir, "rollouts"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def run_rollouts(
    cfg: DictConfig, env: Env, outer_policy: OuterPolicy, run_dir: str
) -> None:
    dirs = _make_dirs(run_dir)
    example_obs, _ = env.reset()
    outer_policy.setup(example_obs)
    successes, rewards, env_steps = ([], [], [])
    for idx in tqdm(range(cfg.num_rollouts), desc="rollout"):
        rollout = collect_episode(env, outer_policy, SamplingMode.EVAL)
        successes.append(float(bool(rollout.is_success)))
        rewards.append(float(sum(rollout.rewards)))
        env_steps.append(
            float(
                getattr(getattr(env, "env", None), "current_step", len(rollout.rewards))
            )
        )
        print(
            f"[rollout] episode {idx}: steps={len(rollout.rewards)} reward={rewards[-1]:.2f} success={rollout.is_success}"
        )
        if idx < cfg.num_videos:
            _maybe_save_video(
                rollout,
                os.path.join(
                    dirs["videos"], f"traj={idx}_succ={rollout.is_success}.mp4"
                ),
                fps=cfg.env.fps,
            )
        if cfg.cache_rollouts:
            rollout.save_to_disk(os.path.join(dirs["rollouts"], f"{idx}.pkl"))
        lower, sample_mean, upper = calculate_beta_posterior_percentile(
            successes, percentile=0.95
        )
        print(
            f"[rollout] success rate: sample mean {sample_mean:.3f}, 95% interval [{lower:.3f}, {upper:.3f}]"
        )
    summary = {
        "num_rollouts": len(rewards),
        "num_successes": int(sum(successes)),
        "success_rate": sample_mean,
        "success_rate_beta_posterior_95": {"lower": lower, "upper": upper},
        "total_reward": sum(rewards),
        "avg_reward": sum(rewards) / max(len(rewards), 1),
        "avg_env_steps": sum(env_steps) / max(len(env_steps), 1),
    }
    with open(os.path.join(dirs["run"], "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[rollout] done: {summary}")
    env.close()


@hydra.main(config_path="cfgs", config_name="rollout_libero", version_base=None)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    env: Env = instantiate(cfg.env, _convert_="all")
    outer_policy: OuterPolicy = instantiate(cfg.outer_policy, _convert_="all")
    run_dir = _run_dir(cfg, type(outer_policy), type(env))
    _make_dirs(run_dir)
    OmegaConf.save(cfg, os.path.join(run_dir, "config.yaml"))
    run_rollouts(cfg, env, outer_policy, run_dir)


if __name__ == "__main__":
    main()
