"""Common benchmark setup and dataset-shard lifecycle for the state trainer."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import gymnasium as gym


@dataclass
class DatasetShards:
    paths: tuple = ()
    interval: int = 0
    loader: Callable | None = None
    index: int = 0

    def should_rotate(self, step):
        return (
            len(self.paths) > 1
            and self.interval > 0
            and step // self.interval % len(self.paths) != self.index
        )

    def load(self, step):
        self.index = step // self.interval % len(self.paths)
        path = self.paths[self.index]
        print(
            f"Loading dataset shard {self.index + 1}/{len(self.paths)}: {path}",
            flush=True,
        )
        return self.loader(path)


def find_dataset_shards(directory, proportion, interval):
    if interval <= 0:
        raise ValueError(
            "dataset_replace_interval must be positive for sharded datasets"
        )
    if not 0 < proportion <= 1:
        raise ValueError("dataset_proportion must be in (0, 1]")
    paths = sorted(
        str(p) for p in Path(directory).glob("*.npz") if not p.name.endswith("-val.npz")
    )
    if not paths:
        raise FileNotFoundError(f"No training shards found in {directory}")
    return tuple(paths[: max(1, int(len(paths) * proportion))])


@dataclass
class Benchmark:
    env: object
    eval_env: object
    train_dataset: object
    val_dataset: object
    make_eval_env: Callable
    shards: DatasetShards = field(default_factory=DatasetShards)
    vector_eval_env: object = field(default=None, init=False)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.vector_eval_env is not None:
            self.vector_eval_env.close(timeout=5)
        self.eval_env.close()
        if self.env is not self.eval_env:
            self.env.close()

    def make_vector_eval_env(self, options):
        if options.eval_batch_size <= 1:
            return None
        if options.eval_episodes % options.eval_batch_size:
            raise ValueError("eval_episodes must be divisible by eval_batch_size")
        if (
            options.online_steps >= 1_000_000
            and options.million_step_eval_episodes % options.eval_batch_size
        ):
            raise ValueError(
                "million_step_eval_episodes must be divisible by eval_batch_size"
            )
        self.vector_eval_env = gym.vector.AsyncVectorEnv(
            [self.make_eval_env] * options.eval_batch_size
        )
        return self.vector_eval_env


def configure_agent_observations(env_name, config):
    """Record the Transport observation layout in checkpoint configuration."""
    if env_name.startswith("transport-") and env_name.endswith("-low_dim"):
        from envs.robomimic_observations import get_low_dim_obs_keys

        config["observation_keys"] = get_low_dim_obs_keys(env_name)


def make_benchmark(options, config):
    if options.ogbench_dataset_dir and options.ocbench_dataset_dir:
        raise ValueError("Specify only one benchmark dataset directory")
    if "-cpu-" in options.env_name:
        if not options.ocbench_dataset_dir:
            raise ValueError("OCBench requires --ocbench_dataset_dir")
        from envs.ocbench_utils import make_training_benchmark
    elif "low_dim" in options.env_name:
        if options.ocbench_dataset_dir or options.ogbench_dataset_dir:
            raise ValueError("RoboMimic datasets are loaded from ~/.robomimic")
        from envs.robomimic_utils import make_training_benchmark
    elif "singletask" in options.env_name:
        if options.ocbench_dataset_dir:
            raise ValueError("OGBench tasks cannot use an OCBench dataset directory")
        from envs.ogbench_utils import make_training_benchmark
    else:
        raise ValueError(f"Unsupported benchmark: {options.env_name}")
    return make_training_benchmark(options, config.get("dataset_action_clip_eps", 1e-5))
