"""Environment and local-dataset utilities for OCBench."""

import numpy as np
import gymnasium
from gymnasium.spaces import Box
from envs.env_utils import EpisodeMonitor, FrameStackWrapper
from utils.datasets import Dataset

DATASET_KEYS = (
    "observations",
    "actions",
    "next_observations",
    "rewards",
    "masks",
    "terminals",
)


def validate_ocbench_env_dataset(env_name, dataset_path):
    if "-cpu-" not in env_name:
        raise ValueError("The release supports the paper CPU OCBench tasks only")


class ActionNormalizer:
    """Percentile action normalizer used by the upstream OCBench trainer."""

    def __init__(self, q_low, q_high, clip):
        self.q_low = np.asarray(q_low, dtype=np.float32)
        self.q_high = np.asarray(q_high, dtype=np.float32)
        self.clip = float(clip)
        self.gripper_dims = np.zeros_like(self.q_low, dtype=bool)
        self.gripper_dims[-1] = True
        self.q_low[self.gripper_dims] = -1.0
        self.q_high[self.gripper_dims] = 1.0
        self.center = (self.q_low + self.q_high) / 2
        self.half_range = (self.q_high - self.q_low) / 2
        self.valid = self.half_range > 0
        self.action_low = np.full_like(self.q_low, -self.clip)
        self.action_high = np.full_like(self.q_high, self.clip)
        self.action_low[self.gripper_dims] = -1.0
        self.action_high[self.gripper_dims] = 1.0

    @classmethod
    def fit(cls, actions, percentile, clip):
        q_low = np.percentile(actions, percentile, axis=0).astype(np.float32)
        q_high = np.percentile(actions, 100 - percentile, axis=0).astype(np.float32)
        return cls(q_low, q_high, clip)

    def normalize(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        normalized = np.zeros_like(actions, dtype=np.float32)
        normalized[..., self.valid] = (
            actions[..., self.valid] - self.center[self.valid]
        ) / self.half_range[self.valid]
        return np.clip(normalized, self.action_low, self.action_high)

    def unnormalize(self, actions):
        actions = np.clip(actions, self.action_low, self.action_high)
        return actions * self.half_range + self.center


class ActionNormWrapper(gymnasium.ActionWrapper):
    """Map normalized policy actions back to the OCBench action space."""

    def __init__(self, env, action_normalizer):
        super().__init__(env)
        self.action_normalizer = action_normalizer
        self.action_space = Box(
            low=action_normalizer.action_low,
            high=action_normalizer.action_high,
            dtype=np.float32,
        )

    def action(self, action):
        action = self.action_normalizer.unnormalize(action)
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)


def make_ocbench_eval_env(env_name, frame_stack=None, action_normalizer=None):
    """Create a monitored OCBench evaluation environment."""
    import ocbench

    raw_env = ocbench.make(env_name)
    env = EpisodeMonitor(raw_env, filter_regexes=[".*privileged.*", ".*proprio.*"])
    if action_normalizer is not None:
        env = ActionNormWrapper(env, action_normalizer)
    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
    env.reset()
    return env


def load_ocbench_datasets(
    dataset_path,
    action_normalizer=None,
    action_norm_percentile=None,
    action_norm_clip=5.0,
    load_validation=True,
    return_action_normalizer=False,
    dataset_format="auto",
    extra_train_paths=(),
):
    """Load transition arrays or dense episode observations with final endpoints.

    ``extra_train_paths`` are further train shards concatenated after ``dataset_path``
    (``--ocbench_load_all_shards``): the action normalizer is always fitted on ``dataset_path``
    alone, every shard is normalized and optionally success-filtered on its own, and a shard
    boundary ends an episode (terminal on each shard's last row).
    """
    if dataset_format not in {"auto", "transitions", "episode_observations"}:
        raise ValueError(f"Unknown OCBench dataset format: {dataset_format}")
    val_path = dataset_path.replace(".npz", "-val.npz")
    train_paths = [dataset_path, *extra_train_paths]
    paths = train_paths + ([val_path] if load_validation else [])
    datasets = []
    for path in paths:
        with np.load(path) as data:
            packed_observations = dataset_format == "episode_observations" or (
                dataset_format == "auto" and "next_observations" not in data
            )
            keys = (
                [key for key in DATASET_KEYS if key != "next_observations"]
                if packed_observations
                else DATASET_KEYS
            )
            dataset = {key: data[key].astype(np.float32, copy=False) for key in keys}
            if packed_observations:
                if (
                    "observation_interval" not in data
                    or data["observation_interval"].shape != ()
                    or data["observation_interval"] != 1
                ):
                    raise ValueError(
                        f"Episode observations require observation_interval=1 in {path}; missing intermediate observations cannot reconstruct one-step transitions."
                    )
                size = len(dataset["actions"])
                terminals = dataset["terminals"]
                if (
                    terminals.shape != (size,)
                    or any((len(dataset[key]) != size for key in ("rewards", "masks")))
                    or (not size)
                    or (terminals[-1] != 1)
                    or (not np.all((terminals == 0) | (terminals == 1)))
                ):
                    raise ValueError(
                        f"Invalid episode transition boundaries in {path}."
                    )
                observations = dataset["observations"]
                if len(observations) != size + np.count_nonzero(terminals):
                    raise ValueError(
                        f"Expected one final observation per episode in {path}."
                    )
                observation_idxs = np.arange(size) + np.concatenate(
                    [[0], np.cumsum(terminals[:-1], dtype=np.int64)]
                )
                dataset["observations"] = observations[observation_idxs]
                dataset["next_observations"] = observations[observation_idxs + 1]
        datasets.append(dataset)
    if action_normalizer is None and action_norm_percentile is not None:
        action_normalizer = ActionNormalizer.fit(
            datasets[0]["actions"], action_norm_percentile, action_norm_clip
        )
    if action_normalizer is not None:
        for dataset in datasets:
            dataset["actions"] = action_normalizer.normalize(dataset["actions"])
    train_parts = datasets[: len(train_paths)]
    if len(train_parts) > 1:
        for part in train_parts:
            part["terminals"] = part["terminals"].copy()
            part["terminals"][-1] = 1.0
        train_dict = {
            key: np.concatenate([part[key] for part in train_parts])
            for key in DATASET_KEYS
        }
    else:
        train_dict = train_parts[0]
    train_dataset = Dataset.create(**train_dict)
    val_dataset = Dataset.create(**datasets[-1]) if load_validation else None
    if return_action_normalizer:
        return (train_dataset, val_dataset, action_normalizer)
    return (train_dataset, val_dataset)


def make_ocbench_env_and_datasets(
    env_name,
    dataset_path,
    frame_stack=None,
    action_norm_percentile=1.0,
    action_norm_clip=5.0,
    extra_dataset_paths=(),
):
    """Create OCBench train/eval environments and load a local dataset shard (plus ``extra_dataset_paths``)."""
    validate_ocbench_env_dataset(env_name, dataset_path)
    train_dataset, val_dataset, action_normalizer = load_ocbench_datasets(
        dataset_path,
        action_norm_percentile=action_norm_percentile,
        action_norm_clip=action_norm_clip,
        return_action_normalizer=True,
        extra_train_paths=extra_dataset_paths,
    )
    env_kwargs = dict(frame_stack=frame_stack, action_normalizer=action_normalizer)
    env = make_ocbench_eval_env(env_name, **env_kwargs)
    eval_env = make_ocbench_eval_env(env_name, **env_kwargs)
    return (env, eval_env, train_dataset, val_dataset, action_normalizer)


def make_training_benchmark(options, action_clip_eps):
    """Keep shard selection and the shared action normalizer together."""
    from functools import partial
    from envs.benchmark import Benchmark, DatasetShards, find_dataset_shards

    paths = find_dataset_shards(
        options.ocbench_dataset_dir,
        options.dataset_proportion,
        options.dataset_replace_interval,
    )
    extra_paths = paths[1:] if options.ocbench_load_all_shards else ()
    if options.ocbench_load_all_shards:
        paths = paths[:1]
    env, eval_env, train, validation, normalizer = make_ocbench_env_and_datasets(
        options.env_name,
        paths[0],
        frame_stack=options.frame_stack,
        action_norm_percentile=options.ocbench_action_norm_percentile,
        action_norm_clip=options.ocbench_action_norm_clip,
        extra_dataset_paths=extra_paths,
    )

    def load_shard(path):
        train, _ = load_ocbench_datasets(
            path, action_normalizer=normalizer, load_validation=False
        )
        return train

    return Benchmark(
        env,
        eval_env,
        train,
        validation,
        partial(
            make_ocbench_eval_env,
            options.env_name,
            frame_stack=options.frame_stack,
            action_normalizer=normalizer,
        ),
        DatasetShards(paths, options.dataset_replace_interval, load_shard),
    )
