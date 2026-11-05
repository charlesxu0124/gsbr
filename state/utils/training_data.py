"""Offline data preparation, replay restoration, sampling, and BC prefill."""

import gc
import json
import os
import jax
import numpy as np
from utils.datasets import (
    Dataset,
    ReplayBuffer,
    load_replay_buffer,
    process_offline_train_dataset,
)
from utils.evaluation import discounted_return


def _remap_sparse_env_reward(reward):
    """Dense -> sparse step reward for online replay."""
    r = np.asarray(reward)
    out = np.where(r != 0.0, -1.0, 0.0).astype(r.dtype)
    return out.item() if out.ndim == 0 else out


def transform_replay_reward(reward, *, sparse, reward_scale, reward_bias):
    """Apply the same reward transform used for offline replay data."""
    if sparse:
        reward = _remap_sparse_env_reward(reward)
    reward = np.asarray(reward) * reward_scale + reward_bias
    return reward.item() if reward.ndim == 0 else reward


def prepare_dataset(dataset, options):
    if dataset is None:
        return None
    dataset = process_offline_train_dataset(dataset, sparse=options.sparse)
    if options.reward_scale != 1.0 or options.reward_bias != 0.0:
        dataset = dataset.copy(
            add_or_replace=dict(
                rewards=dataset["rewards"] * options.reward_scale + options.reward_bias
            )
        )
    dataset.p_aug = options.p_aug
    dataset.frame_stack = options.frame_stack
    return dataset


def prepare_training_data(benchmark, options, config):
    env = benchmark.env
    train_dataset, val_dataset = benchmark.train_dataset, benchmark.val_dataset
    # Release the setup object's references so shard rotation can free old data.
    benchmark.train_dataset = benchmark.val_dataset = None
    skip_offline_dataset = (
        options.bc_steps == 0
        and options.offline_steps == 0
        and (not options.balanced_sampling)
    )
    if skip_offline_dataset:
        train_dataset = None
        val_dataset = None
        gc.collect()
        ex_obs = np.asarray(env.observation_space.sample(), dtype=np.float32)
        ex_act = np.asarray(env.action_space.sample(), dtype=np.float32)
        example_transition = dict(
            observations=ex_obs,
            actions=ex_act,
            rewards=np.float32(0.0),
            masks=np.float32(1.0),
            terminals=np.float32(0.0),
            next_observations=ex_obs,
        )
        replay_buffer = ReplayBuffer.create(
            example_transition, size=options.buffer_size
        )
    else:
        if options.override_dataset_path:
            print(
                f"Loading train_dataset from saved replay buffer: {options.override_dataset_path}"
            )
            train_dataset = Dataset.load(options.override_dataset_path)
            print(f"Loaded train_dataset with size: {train_dataset.size}")
        else:
            train_dataset = Dataset.create(**train_dataset)
        train_dataset = prepare_dataset(train_dataset, options)
        val_dataset = prepare_dataset(val_dataset, options)
        example_transition = {key: value[0] for key, value in train_dataset.items()}
        replay_buffer = ReplayBuffer.create(
            example_transition, size=options.buffer_size
        )
    if options.restore_epoch > options.bc_steps + options.offline_steps and (
        not options.eval_only
    ):
        legacy_is_offline = None
        restore_flags_path = os.path.join(options.restore_path, "flags.json")
        if os.path.exists(restore_flags_path):
            with open(restore_flags_path) as f:
                restore_flags = json.load(f)
            if restore_flags.get("balanced_sampling", False) or (
                restore_flags.get("offline_steps") == 0
                and restore_flags.get("preload_offline_data") is False
            ):
                legacy_is_offline = 0.0
        replay_buffer = load_replay_buffer(
            options.restore_path,
            options.restore_epoch,
            legacy_is_offline=legacy_is_offline,
        )
        print(
            f"Restored replay buffer from step {options.restore_epoch} (size: {replay_buffer.size})"
        )
    for dataset in [train_dataset, val_dataset, replay_buffer]:
        if dataset is not None:
            dataset.p_aug = options.p_aug
            dataset.frame_stack = options.frame_stack
    validation_discounted_return = None
    if val_dataset is not None and len(val_dataset.terminal_locs) > 0:
        episode_starts = np.concatenate([[0], val_dataset.terminal_locs[:-1] + 1])
        validation_discounted_return = np.mean(
            [
                discounted_return(
                    val_dataset["rewards"][start : end + 1], config.get("discount", 1.0)
                )
                for start, end in zip(episode_starts, val_dataset.terminal_locs)
            ]
        )
    if train_dataset is not None:
        example_batch = train_dataset.sample(1)
    else:
        ex_obs = np.asarray(env.observation_space.sample(), dtype=np.float32)[None]
        ex_act = np.asarray(env.action_space.sample(), dtype=np.float32)[None]
        example_batch = dict(observations=ex_obs, actions=ex_act)
    return (
        train_dataset,
        val_dataset,
        replay_buffer,
        example_batch,
        validation_discounted_return,
    )


def prefill_replay_buffer(
    *, options, agent, env, replay_buffer, num_episodes, cache_path, seed
):
    """Populate replay_buffer with `num_episodes` rollouts of the BC base flow.

    If `cache_path` exists, load and replay the cached transitions. Otherwise
    roll out fresh and write to `cache_path` (when set).

    """
    import pickle

    transitions = None
    if cache_path is not None and os.path.exists(cache_path):
        print(f"Loading prefill cache from {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, dict):
            transitions = cached["transitions"]
            for t, terminated in zip(transitions, cached["terminated"]):
                t["masks"] = np.float32(
                    1.0
                    - (
                        terminated
                        if options.bootstrap_on_truncation
                        else t["terminals"]
                    )
                )
        else:
            transitions = cached
        print(f"Loaded {len(transitions)} cached prefill transitions.", flush=True)
    if transitions is None:
        print(
            f"Rolling out {num_episodes} BC episodes for replay-buffer prefill...",
            flush=True,
        )
        transitions = []
        terminated_flags = []
        rng = jax.random.PRNGKey(seed + 12345)
        action_dim = env.action_space.shape[0]
        for ep in range(num_episodes):
            ob, _ = env.reset()
            done = False
            action_queue = []
            ep_success = False
            while not done:
                if len(action_queue) == 0:
                    rng, key = jax.random.split(rng)
                    sampled = agent.sample_base_actions_only(observations=ob, seed=key)
                    chunk = np.array(sampled).reshape(-1, action_dim)
                    action_queue.extend(chunk)
                action = action_queue.pop(0)
                next_ob, reward, terminated, truncated, info = env.step(action.copy())
                done = terminated or truncated
                ep_success = ep_success or bool(info.get("success", 0))
                reward_t = transform_replay_reward(
                    reward,
                    sparse=options.sparse,
                    reward_scale=options.reward_scale,
                    reward_bias=options.reward_bias,
                )
                t = dict(
                    observations=np.asarray(ob, dtype=np.float32),
                    actions=np.asarray(action, dtype=np.float32),
                    rewards=np.float32(reward_t),
                    terminals=np.float32(done),
                    masks=np.float32(
                        1.0 - (terminated if options.bootstrap_on_truncation else done)
                    ),
                    next_observations=np.asarray(next_ob, dtype=np.float32),
                )
                transitions.append(t)
                terminated_flags.append(np.float32(terminated))
                ob = next_ob
            print(
                f"  prefill episode {ep + 1}/{num_episodes} done  (success={ep_success})",
                flush=True,
            )
        if cache_path is not None:
            os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(
                    dict(transitions=transitions, terminated=terminated_flags), f
                )
            print(
                f"Wrote {len(transitions)} prefill transitions to {cache_path}",
                flush=True,
            )
    if transitions:
        replay_buffer.truncate_episode()
    for t in transitions:
        replay_buffer.add_transition(t)
    print(f"Replay buffer size after prefill: {replay_buffer.size}", flush=True)
