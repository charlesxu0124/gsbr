"""Small synthetic environment for exercising the real trainer and checkpoints."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import numpy as np
import main
from envs.env_utils import EpisodeMonitor
from envs.benchmark import Benchmark
from utils.datasets import Dataset


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (4,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)

    def reset(self, **kwargs):
        self.steps = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        return (
            np.zeros(4, dtype=np.float32),
            0.0,
            self.steps == 6,
            False,
            {"success": int(self.steps == 6)},
        )


def make_env(*args, **kwargs):
    terminals = np.zeros(24, dtype=np.float32)
    terminals[5::6] = 1
    data = Dataset.create(
        observations=np.zeros((24, 4), dtype=np.float32),
        actions=np.zeros((24, 2), dtype=np.float32),
        rewards=np.zeros(24, dtype=np.float32),
        masks=1 - terminals,
        terminals=terminals,
        next_observations=np.zeros((24, 4), dtype=np.float32),
    )
    return EpisodeMonitor(TinyEnv()), EpisodeMonitor(TinyEnv()), data, None


if __name__ == "__main__":
    benchmarks = []

    def make_benchmark(options, config):
        benchmark = Benchmark(*make_env(), lambda: EpisodeMonitor(TinyEnv()))
        benchmarks.append(benchmark)
        return benchmark

    main.make_benchmark = make_benchmark
    main.FLAGS(sys.argv)
    main.main([])
    print(f"Training environment steps: {benchmarks[0].env.total_timesteps}")
