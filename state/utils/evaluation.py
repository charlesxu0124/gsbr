import copy
from collections import defaultdict
from functools import partial
from typing import List, Optional
import jax
import numpy as np


def supply_rng(f, rng=None):
    """Helper function to split the random number generator key before each call to the function."""
    if rng is None:
        rng = jax.random.PRNGKey(0)

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key="", sep="."):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def discounted_return(rewards, discount):
    """Compute a per-timestep discounted return for one episode."""
    rewards = np.asarray(rewards, dtype=np.float64)
    discounts = np.power(discount, np.arange(len(rewards), dtype=np.float64))
    return np.sum(discounts * rewards)


def discounted_return_to_go(rewards, discount, chunk_length=1):
    """Compute returns with undiscounted rewards within each decision chunk.

    The default discounts every environment step. For a chunk of length H,
    rewards at offsets 0..H-1 have weight 1, H..2H-1 have weight discount**H,
    and so on, matching Dataset.sample_sequence(chunk_discount="chunk").
    """
    if chunk_length < 1:
        raise ValueError("chunk_length must be positive")
    rewards = np.asarray(rewards, dtype=np.float64)
    returns_to_go = np.empty_like(rewards)
    chunk_discount = discount**chunk_length
    for timestep in range(len(rewards) - 1, -1, -1):
        returns_to_go[timestep] = rewards[timestep : timestep + chunk_length].sum()
        next_chunk = timestep + chunk_length
        if next_chunk < len(rewards):
            returns_to_go[timestep] += chunk_discount * returns_to_go[next_chunk]
    return returns_to_go


def _slice_diagnostics(diagnostics, idx):
    out = {}
    for key, value in diagnostics.items():
        arr = np.asarray(value)
        out[key] = arr[idx].copy() if arr.ndim > 0 else arr.copy()
    return out


def _aggregate_episode_diagnostics(traj, diagnostic_return_to_go):
    diagnostics = traj.get("diagnostics", [])
    if not diagnostics:
        return {}
    out = {}
    keys = set().union(*(diagnostic.keys() for diagnostic in diagnostics))
    for key in keys:
        values = [np.asarray(d[key]) for d in diagnostics if key in d]
        if not values:
            continue
        if values[0].ndim == 0:
            arr = np.asarray(values, dtype=np.float32)
            out[f"diagnostics.{key}"] = np.mean(arr)
    q_timesteps = [
        timestep for timestep, diagnostic in enumerate(diagnostics) if "q" in diagnostic
    ]
    if q_timesteps:
        q = np.asarray(
            [diagnostics[timestep]["q"] for timestep in q_timesteps], dtype=np.float64
        ).reshape(-1)
        return_to_go = diagnostic_return_to_go[q_timesteps]
        q_error = q - return_to_go
        out["q_mae"] = np.mean(np.abs(q_error))
        out["q_rmse"] = np.sqrt(np.mean(np.square(q_error)))
        out["q_bias"] = np.mean(q_error)
    return out


def _episode_diagnostic_trace(traj, discount, diagnostic_options):
    diagnostics = traj.get("diagnostics", [])
    diagnostic_rewards = np.asarray(traj["reward"], dtype=np.float64)
    if diagnostic_options["sparse"]:
        diagnostic_rewards = np.where(diagnostic_rewards != 0.0, -1.0, 0.0)
    diagnostic_rewards = (
        diagnostic_rewards * diagnostic_options["reward_scale"]
        + diagnostic_options["reward_bias"]
    )
    chunk_length = (
        diagnostic_options["horizon_length"]
        if diagnostic_options["chunk_discount"] == "chunk"
        else 1
    )
    out = {
        "reward": np.asarray(traj["reward"], dtype=np.float32),
        "return_to_go": discounted_return_to_go(traj["reward"], discount),
        "diagnostic_reward": diagnostic_rewards,
        "diagnostic_return_to_go": discounted_return_to_go(
            diagnostic_rewards, discount, chunk_length=chunk_length
        ),
        "diagnostic_options": dict(diagnostic_options, discount=discount),
    }
    keys = set().union(*(diagnostic.keys() for diagnostic in diagnostics))
    for key in keys:
        values = [np.asarray(d[key]) for d in diagnostics if key in d]
        if values:
            out[key] = np.stack(values, axis=0)
    out["q_timesteps"] = np.asarray(
        [
            timestep
            for timestep, diagnostic in enumerate(diagnostics)
            if "q" in diagnostic
        ],
        dtype=np.int32,
    )
    if "q" in out:
        out["q_prediction_mask"] = np.zeros(len(traj["reward"]), dtype=bool)
        out["q_prediction_mask"][out["q_timesteps"]] = True
        out["q_by_timestep"] = np.full(len(traj["reward"]), np.nan)
        out["q_by_timestep"][out["q_timesteps"]] = out["q"].reshape(-1)
    rendered = traj.get("rendered", [])
    render_timesteps = [i for i, value in enumerate(rendered) if value]
    if render_timesteps:
        out["render_timesteps"] = np.asarray(render_timesteps, dtype=np.int32)
    return out


class SingleEnvBatchAdapter:
    """Adapter to present a single Gymnasium env as a batched (num_envs=1) env.

    Accepts batched actions with leading batch dimension 1 and always returns
    batched outputs, emulating vector env API sufficiently for evaluation.
    """

    def __init__(self, env):
        self._env = env
        self.num_envs = 1

    def reset(self, *, seed=None, options=None):
        obs, info = self._env.reset(seed=seed, options=options)
        return (np.expand_dims(obs, axis=0), info)

    def step(self, actions):
        action = (
            actions[0]
            if isinstance(actions, np.ndarray)
            and actions.ndim >= 1
            and (actions.shape[0] == 1)
            else actions
        )
        obs, reward, terminated, truncated, info = self._env.step(action)
        return (
            np.expand_dims(obs, axis=0),
            np.array([reward], dtype=np.float32),
            np.array([terminated], dtype=np.bool_),
            np.array([truncated], dtype=np.bool_),
            info,
        )

    def render(self):
        return self._env.render()

    @property
    def single_action_space(self):
        return self._env.action_space


def _get_num_envs(env):
    if "num_envs" in vars(env):
        return env.num_envs
    if hasattr(type(env), "get_wrapper_attr"):
        try:
            return env.get_wrapper_attr("num_envs")
        except AttributeError:
            return None
    return None


def _ensure_batch(x):
    if isinstance(x, tuple):
        raise TypeError("Expected observation to be array-like, got tuple.")
    if isinstance(x, np.ndarray) and x.ndim > 0:
        return x
    return np.expand_dims(x, axis=0)


def _maybe_concat_goal_to_obs(
    observations: np.ndarray, goals: List, num_envs: int, policy_obs_dim: Optional[int]
) -> np.ndarray:
    """If env returns state-only but training used state‖goal, match dataset observation width."""
    if policy_obs_dim is None:
        return observations
    obs = np.asarray(observations)
    if obs.ndim == 1:
        obs = obs[None, :]
    rows = []
    for i in range(num_envs):
        o = np.asarray(obs[i]).reshape(-1)
        g = goals[i] if i < len(goals) else None
        if g is not None:
            g = np.asarray(g).reshape(-1)
            if o.shape[0] == policy_obs_dim:
                rows.append(o)
            elif o.shape[0] + g.shape[0] == policy_obs_dim:
                rows.append(np.concatenate([o, g], axis=-1))
            else:
                rows.append(o)
        else:
            rows.append(o)
    return np.stack(rows, axis=0)


def _vector_infos_to_list(infos, num_envs):
    if isinstance(infos, dict) and any((k.startswith("_") for k in infos)):
        per_env = [dict() for _ in range(num_envs)]
        for key, value in infos.items():
            if key.startswith("_"):
                continue
            mask = infos.get(f"_{key}")
            if mask is None:
                mask = np.ones(num_envs, dtype=bool)
            for idx in range(num_envs):
                if mask[idx]:
                    per_env[idx][key] = value[idx]
        return per_env
    if isinstance(infos, dict):
        return [infos for _ in range(num_envs)]
    if isinstance(infos, (list, tuple)):
        return list(infos)
    if infos is None:
        return [dict() for _ in range(num_envs)]
    return [{"info": infos} for _ in range(num_envs)]


def _prepare_actor(agent, temperature):
    action_fn = getattr(agent, "sample_actions_with_diagnostics", agent.sample_actions)
    return partial(
        supply_rng(action_fn, rng=jax.random.PRNGKey(np.random.randint(0, 2**32))),
        temperature=temperature,
    )


def run_episodes(
    agent,
    env,
    task_id=None,
    eval_temperature=0,
    eval_gaussian=None,
    goal_conditioned=False,
    should_render=False,
    video_frame_skip=3,
):
    """Shared rollout for sequential and vectorized environments (batch-first).

    Always treat inputs/outputs as batched. If `env` is single, use SingleEnvBatchAdapter.
    """
    num_envs = _get_num_envs(env)
    if num_envs is None:
        env = SingleEnvBatchAdapter(env)
        num_envs = 1
    if num_envs > 1 and should_render:
        raise ValueError("Rendering is only supported for single environments.")
    actor_fn = _prepare_actor(agent, temperature=eval_temperature)
    horizon_length = 1
    action_dim = None
    policy_obs_dim = None
    if hasattr(agent, "config"):
        horizon_length = int(agent.config.get("horizon_length", 1))
        action_dim = agent.config.get("action_dim", None)
        if action_dim is not None:
            action_dim = int(action_dim)
        _pod = agent.config.get("policy_observation_dim")
        if _pod is not None:
            policy_obs_dim = int(_pod)
    reset_options = dict(render_goal=should_render)
    if task_id is not None:
        reset_options["task_id"] = task_id
    observations, reset_infos = env.reset(options=reset_options)
    active = np.ones(num_envs, dtype=bool)
    returns = np.zeros(num_envs, dtype=np.float32)
    lengths = np.zeros(num_envs, dtype=np.int32)
    trajectories = [defaultdict(list) for _ in range(num_envs)]
    renders = [[] for _ in range(num_envs)]
    infos_per_env = (
        _vector_infos_to_list(reset_infos, num_envs)
        if isinstance(reset_infos, dict)
        else [reset_infos] * num_envs
    )
    goals = [info.get("goal") for info in infos_per_env]
    goal_frames = [info.get("goal_rendered") for info in infos_per_env]
    observations = _maybe_concat_goal_to_obs(
        observations, goals, num_envs, policy_obs_dim
    )
    rng = np.random.default_rng()
    action_queues = [[] for _ in range(num_envs)]
    diagnostic_queues = [[] for _ in range(num_envs)]
    while not np.all(~active):
        if horizon_length > 1:
            need_chunk = [i for i in range(num_envs) if len(action_queues[i]) == 0]
        else:
            need_chunk = list(range(num_envs))
        if need_chunk:
            subset_obs = observations[need_chunk]
            if goal_conditioned:
                subset_goals = [goals[i] for i in need_chunk]
                actor_out = actor_fn(observations=subset_obs, goals=subset_goals)
            else:
                actor_out = actor_fn(observations=subset_obs)
            chunk_diagnostics = None
            if isinstance(actor_out, tuple) and len(actor_out) == 2:
                raw, chunk_diagnostics = actor_out
                chunk_diagnostics = jax.tree_util.tree_map(
                    lambda x: np.asarray(x), chunk_diagnostics
                )
            else:
                raw = actor_out
            raw = np.array(raw)
            if raw.ndim == 0:
                raw = np.array([raw])
            if raw.ndim == 1:
                raw = raw[None, ...]
            if horizon_length > 1:
                if action_dim is None:
                    if raw.shape[-1] % horizon_length != 0:
                        raise ValueError(
                            f"Cannot infer per-step action_dim: raw.shape[-1]={raw.shape[-1]}, horizon_length={horizon_length}"
                        )
                    action_dim = raw.shape[-1] // horizon_length
                chunks = raw.reshape(len(need_chunk), horizon_length, action_dim)
                for j, idx in enumerate(need_chunk):
                    action_queues[idx].extend(chunks[j])
                    if chunk_diagnostics is not None:
                        step_diag = _slice_diagnostics(chunk_diagnostics, j)
                        diagnostic_queues[idx].extend(
                            [step_diag] + [{} for _ in range(horizon_length - 1)]
                        )
            else:
                for j, idx in enumerate(need_chunk):
                    action_queues[idx].append(raw[j])
                    if chunk_diagnostics is not None:
                        diagnostic_queues[idx].append(
                            _slice_diagnostics(chunk_diagnostics, j)
                        )
        actions = np.array([action_queues[i].pop(0) for i in range(num_envs)])
        step_diagnostics = [
            diagnostic_queues[i].pop(0) if diagnostic_queues[i] else {}
            for i in range(num_envs)
        ]
        if eval_gaussian is not None:
            actions = rng.normal(loc=actions, scale=eval_gaussian)
        actions = np.clip(
            actions, env.single_action_space.low, env.single_action_space.high
        )
        next_observations, rewards, terminations, truncations, step_infos = env.step(
            actions
        )
        infos_per_step = _vector_infos_to_list(step_infos, num_envs)
        done_now = np.logical_or(terminations, truncations)
        for idx in range(num_envs):
            reward = rewards[idx]
            info = infos_per_step[idx]
            next_observation = next_observations[idx]
            if active[idx]:
                lengths[idx] += 1
                returns[idx] += reward
                render_now = False
                if done_now[idx]:
                    if "final_observation" in info:
                        next_observation = info["final_observation"]
                    if "final_info" in info:
                        info = copy.deepcopy(info["final_info"])
                if "goal" in info:
                    goals[idx] = info["goal"]
                if should_render and "goal_rendered" in info:
                    goal_frames[idx] = info["goal_rendered"]
                if should_render:
                    step_count = lengths[idx]
                    render_now = step_count % video_frame_skip == 0 or done_now[idx]
                transition = dict(
                    observation=observations[idx],
                    next_observation=next_observation,
                    action=actions[idx],
                    reward=reward,
                    done=done_now[idx],
                    info=info,
                    diagnostics=step_diagnostics[idx],
                    rendered=render_now,
                )
                add_to(trajectories[idx], copy.deepcopy(transition))
                if render_now:
                    frame = env.render().copy()
                    goal_frame = goal_frames[idx]
                    if goal_frame is not None:
                        renders[idx].append(np.concatenate([goal_frame, frame], axis=0))
                    else:
                        renders[idx].append(frame)
                if done_now[idx]:
                    action_queues[idx] = []
                    diagnostic_queues[idx] = []
                    active[idx] = False
            o_next = np.asarray(next_observation).reshape(-1)
            if (
                policy_obs_dim is not None
                and goals[idx] is not None
                and (o_next.shape[0] < policy_obs_dim)
            ):
                g = np.asarray(goals[idx]).reshape(-1)
                if o_next.shape[0] + g.shape[0] == policy_obs_dim:
                    o_next = np.concatenate([o_next, g], axis=-1)
            observations[idx] = o_next
    if should_render:
        renders = [np.array(r) for r in renders]
    return (trajectories, renders, returns, lengths)


def rollout(
    agent,
    env,
    venv=None,
    task_id=None,
    num_eval_episodes=50,
    num_video_episodes=0,
    video_frame_skip=3,
    eval_temperature=0,
    eval_gaussian=None,
    goal_conditioned=False,
    return_video_diagnostics=False,
    diagnostic_reward_scale=1.0,
    diagnostic_reward_bias=0.0,
    diagnostic_sparse=False,
    diagnostic_chunk_discount="step",
    diagnostic_horizon_length=None,
):
    """Evaluate the agent in the environment with optimized execution.

    Returns a tuple: (stats, trajectories, renders).

    Diagnostic options must match the training replay reward transform and
    discount convention. They affect Q error metrics and diagnostic traces;
    episode return, throughput, and raw discounted_return keep their usual
    environment-reward definitions. Video traces retain raw return_to_go and
    include diagnostic_return_to_go in the critic's reward/discount units.
    """
    trajs = []
    stats = defaultdict(list)
    renders = []
    video_diagnostics = []
    batch_size = _get_num_envs(venv) if venv is not None else 1
    assert num_eval_episodes % batch_size == 0
    total_batches = num_eval_episodes // batch_size
    discount = agent.config.get("discount", 1.0)
    if diagnostic_chunk_discount not in ("step", "chunk"):
        raise ValueError("diagnostic_chunk_discount must be 'step' or 'chunk'")
    if diagnostic_horizon_length is None:
        diagnostic_horizon_length = agent.config.get("horizon_length", 1)
    if diagnostic_horizon_length < 1:
        raise ValueError("diagnostic_horizon_length must be positive")
    diagnostic_options = dict(
        reward_scale=diagnostic_reward_scale,
        reward_bias=diagnostic_reward_bias,
        sparse=diagnostic_sparse,
        chunk_discount=diagnostic_chunk_discount,
        horizon_length=diagnostic_horizon_length,
    )
    for _ in range(total_batches):
        traj_batch, _, returns, lengths = run_episodes(
            agent,
            venv if venv is not None else env,
            task_id,
            eval_temperature,
            eval_gaussian,
            goal_conditioned,
            should_render=False,
            video_frame_skip=video_frame_skip,
        )
        for idx in range(batch_size):
            diagnostic_trace = _episode_diagnostic_trace(
                traj_batch[idx], discount, diagnostic_options
            )
            info_flat = {
                "episode.return": returns[idx],
                "discounted_return": discounted_return(
                    traj_batch[idx]["reward"], discount
                ),
                "episode.length": lengths[idx],
                **flatten(traj_batch[idx]["info"][-1]),
                "diagnostics.discounted_return": diagnostic_trace[
                    "diagnostic_return_to_go"
                ][0],
                **_aggregate_episode_diagnostics(
                    traj_batch[idx], diagnostic_trace["diagnostic_return_to_go"]
                ),
            }
            hz = 15
            info_flat["throughput"] = returns[idx] * 3600 * hz / (lengths[idx] + 1)
            add_to(stats, info_flat)
        trajs.extend(traj_batch)
    for _ in range(num_video_episodes):
        traj_batch, render_batch, _, _ = run_episodes(
            agent,
            env,
            task_id,
            eval_temperature,
            eval_gaussian,
            goal_conditioned,
            should_render=True,
            video_frame_skip=video_frame_skip,
        )
        renders.append(render_batch[0])
        video_diagnostics.append(
            _episode_diagnostic_trace(traj_batch[0], discount, diagnostic_options)
        )
    for k, v in stats.items():
        stats[k] = np.mean(v)
    if return_video_diagnostics:
        return (stats, trajs, renders, video_diagnostics)
    return (stats, trajs, renders)


EVAL_THROUGHPUT_FIXED_TIME_STEPS = 10 * 3600


def add_eval_success_metrics(eval_info, trajs):
    episode_lengths = []
    episode_successes = []
    for traj in trajs:
        rewards = traj.get("reward", [])
        length = len(rewards)
        infos = traj.get("info", [])
        success = any(
            (np.any(np.asarray(info.get("success", 0)) > 0) for info in infos)
        )
        episode_lengths.append(length)
        episode_successes.append(success)
    success_rate = np.mean(episode_successes)
    mean_episode_length = np.mean(episode_lengths)
    success_lengths = [
        length for length, success in zip(episode_lengths, episode_successes) if success
    ]
    if len(success_lengths):
        eval_info["success_episode.length"] = np.mean(success_lengths)
    eval_info["throughput"] = (
        EVAL_THROUGHPUT_FIXED_TIME_STEPS / mean_episode_length * success_rate
    )
    return eval_info
