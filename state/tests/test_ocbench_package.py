"""Exercise the release's actual PyPI OCBench CPU environment interface."""

from importlib.metadata import version

import numpy as np
import pytest

from envs.ocbench_utils import make_ocbench_eval_env


@pytest.mark.parametrize(
    "task",
    [f"block-cpu-lite-double-task{i}-v0" for i in (1, 2)]
    + [f"chamber-cpu-lite-easy-task{i}-v0" for i in (1, 2, 3)],
)
def test_official_cpu_task_reset_and_step(task):
    assert version("ocbench") == "1.0.2"
    assert version("mujoco") == "3.3.6"
    env = make_ocbench_eval_env(task)
    try:
        observation, _ = env.reset(seed=0)
        assert observation.shape == env.observation_space.shape
        assert np.isfinite(observation).all()
        observation, reward, _, _, _ = env.step(np.zeros(env.action_space.shape))
        assert np.isfinite(observation).all() and np.isfinite(reward)
    finally:
        env.close()
