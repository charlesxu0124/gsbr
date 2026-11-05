from copy import deepcopy
from typing import Dict, Optional, Tuple
import numpy as np
import gymnasium
import pathlib
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import benchmark


def get_libero_task(benchmark_id: str, task_id: int):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[benchmark_id]()
    task = task_suite.get_task(task_id)
    return task


def get_libero_env(
    task, resolution: Tuple[int, int], seed: Optional[int] = None
) -> OffScreenRenderEnv:
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution[0],
        "camera_widths": resolution[1],
    }
    env = OffScreenRenderEnv(**env_args)
    if seed is not None:
        env.seed(seed)
    return env


class LiberoEnv(gymnasium.Env):
    OBS_KEYS = [
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    ]
    CAMERA_NAMES = ["agentview", "robot0_eye_in_hand"]
    LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
    NUM_WARMUP_STEPS = 10

    def __init__(
        self,
        task_id: int,
        benchmark_id: str,
        max_steps: int,
        image_resolution: Tuple[int, int],
        use_negative_rewards: bool = False,
        reward_scale: float = 1.0,
        done_at_timeout: bool = False,
        seed: int = 42,
    ):
        """
        image_resolution: (H, W) tuple for the camera images
        """
        super().__init__()
        self.task_id = task_id
        self.benchmark_id = benchmark_id
        self.max_steps = max_steps
        self.image_resolution = image_resolution
        self.use_negative_rewards = use_negative_rewards
        self.reward_scale = reward_scale
        self.done_at_timeout = done_at_timeout
        assert image_resolution[0] == image_resolution[1], (
            "Only square image resolutions are supported for now"
        )
        task = get_libero_task(benchmark_id, task_id)
        self.env = get_libero_env(task, resolution=image_resolution, seed=seed)
        self.task_language = task.language
        self.action_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )
        assert self.CAMERA_NAMES == self.env.env.camera_names, (
            "Camera names do not match"
        )
        for camera_name in self.env.env.camera_names:
            assert f"{camera_name}_image" in self.OBS_KEYS, (
                f"Unexpected camera name {camera_name}"
            )
        self.current_step = 0
        self._needs_to_be_reset = False

    def _process_obs(self, obs):
        out = {key: obs[key] for key in self.OBS_KEYS if key in obs}
        for camera_name in self.env.env.camera_names:
            out[f"{camera_name}_image"] = obs[f"{camera_name}_image"][::-1, ::-1]
        out["task_language"] = self.task_language
        return out

    def _make_info(self, obs, reward: Optional[float] = None) -> Dict:
        return {"is_success": reward == 1.0 if reward is not None else False}

    def _process_reward(self, reward: float) -> float:
        if self.use_negative_rewards:
            if reward > 0:
                reward = 0.0
            else:
                reward = -1.0
        reward *= self.reward_scale
        return reward

    def reset(self, **kwargs):
        obs = self.env.reset()
        for _ in range(self.NUM_WARMUP_STEPS):
            obs, _, _, _ = self.env.step(self.LIBERO_DUMMY_ACTION)
        obs = self._process_obs(obs)
        self._needs_to_be_reset = False
        self.current_step = 0
        return (obs, self._make_info(obs))

    def step(self, action):
        assert not self._needs_to_be_reset, (
            "Attempted to step an environment that requires resetting."
        )
        self.current_step += 1
        obs, reward, terminated, info = self.env.step(action)
        truncated = self.current_step >= self.max_steps
        if self.done_at_timeout:
            terminated = False
        obs = self._process_obs(obs)
        info = self._make_info(obs, reward)
        reward = self._process_reward(reward)
        if terminated or truncated:
            self._needs_to_be_reset = True
        return (obs, reward, terminated, truncated, info)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def set_initial_state(self, state):
        self.env.set_init_state(state)


class LiberoMultiStepEnv(gymnasium.Env):
    def __init__(
        self,
        env: LiberoEnv,
        chunk_size: int = 1,
        auto_reset: bool = False,
        info_captures_all_frames: bool = False,
        reward_mode: str = "negative",
        defer_camera_rendering: bool = False,
        **kwargs,
    ):
        """reward_mode: "negative" gives -1 per executed chunk and 0 on the success chunk
        (returns in [-40, 0] at discount 0.975); "sparse" gives 1 on the success chunk and 0
        otherwise (returns in [0, 1])."""
        if reward_mode not in ("negative", "sparse"):
            raise ValueError(
                f"reward_mode must be 'negative' or 'sparse', got {reward_mode!r}"
            )
        self.env = env
        self.reward_mode = reward_mode
        self.chunk_size = chunk_size
        self.auto_reset = auto_reset
        self.info_captures_all_frames = info_captures_all_frames
        self.defer_camera_rendering = defer_camera_rendering
        self._deferred_renderer = None

    def reset(self, **kwargs):
        obs, info = self.env.reset()
        return (deepcopy(obs), info)

    def step(self, action_chunk: np.ndarray):
        if self.defer_camera_rendering:
            from src.envs.deferred_rgb import DeferredRGBRenderer

            if (
                self._deferred_renderer is None
                or self._deferred_renderer.sim is not self.env.env.env.sim
            ):
                self._deferred_renderer = DeferredRGBRenderer(self)
            return self._deferred_renderer.step(self._step, action_chunk)
        return self._step(action_chunk)

    def _step(self, action_chunk: np.ndarray):
        all_frames = []
        horizon, action_dim = action_chunk.shape
        assert horizon in [1, self.chunk_size], (
            f"Action horizon {horizon} does not match chunk size {self.chunk_size}"
        )
        assert action_dim == 7, f"Action dim {action_dim} does not match expected dim 7"
        for action in action_chunk:
            obs, _, terminated, truncated, info = self.env.step(action)
            if self.info_captures_all_frames:
                all_frames.append(obs["agentview_image"].copy())
            if terminated or truncated:
                break
        obs = deepcopy(obs)
        success = info["is_success"]
        if self.reward_mode == "sparse":
            reward = 1.0 if success else 0.0
        else:
            reward = 0 if success else -1.0
        if self.info_captures_all_frames:
            info["all_frames"] = np.asarray(all_frames)
        if self.auto_reset and (terminated or truncated):
            info["terminal_observation"] = deepcopy(obs)
            obs, _ = self.reset()
        return (obs, reward, terminated, truncated, info)

    def close(self):
        self.env.close()
