import os
from os.path import expanduser

os.environ.setdefault("USE_FLAX", "0")
import gymnasium as gym
import h5py
import imageio
import mujoco
import numpy as np
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from gymnasium.spaces import Box
from envs.robomimic_observations import TRANSPORT_OBSERVATION_KEYS, get_low_dim_obs_keys
from utils.datasets import Dataset

if not hasattr(mujoco.MjData, "qM"):
    _mj_fullM = mujoco.mj_fullM

    def _mj_fullM_legacy_args(model, destination, data):
        return _mj_fullM(model, data, destination)

    mujoco.MjData.qM = property(lambda data: data)
    mujoco.mj_fullM = _mj_fullM_legacy_args


def is_robomimic_env(env_name):
    """Determine if an env name corresponds to a robomimic dataset."""
    if "low_dim" not in env_name:
        return False
    task, dataset_type, _ = env_name.split("-")
    return task in (
        "lift",
        "can",
        "square",
        "transport",
        "tool_hang",
    ) and dataset_type in ("mh", "ph")


def make_eval_env(
    env_name, frame_stack=None, render_videos=False, reward_scale=1.0, reward_bias=0.0
):
    from envs.env_utils import EpisodeMonitor, FrameStackWrapper, RewardWrapper

    if not is_robomimic_env(env_name):
        raise ValueError(f"Unsupported RoboMimic environment: {env_name}")
    env = EpisodeMonitor(make_env(env_name, render_offscreen=render_videos))
    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)
    if reward_scale != 1.0 or reward_bias != 0.0:
        env = RewardWrapper(env, reward_scale, reward_bias)
    env.reset()
    return env


def make_env_and_datasets(
    env_name,
    frame_stack=None,
    action_clip_eps=1e-5,
    render_videos=False,
    truncate_dataset_at_first_success=False,
    reward_scale=1.0,
    reward_bias=0.0,
):
    from envs.env_utils import clip_dataset_actions

    # Load observations before applying frame stacking, as in the original trainer.
    env = make_eval_env(env_name, reward_scale=reward_scale, reward_bias=reward_bias)
    data = get_dataset(
        env, env_name, truncate_at_first_success=truncate_dataset_at_first_success
    )
    if frame_stack is not None:
        from envs.env_utils import FrameStackWrapper

        env = FrameStackWrapper(env, frame_stack)
        env.reset()
    eval_env = make_eval_env(
        env_name, frame_stack, render_videos, reward_scale, reward_bias
    )
    return env, eval_env, clip_dataset_actions(data, action_clip_eps), None


def make_training_benchmark(options, action_clip_eps):
    from functools import partial
    from envs.benchmark import Benchmark

    envs_and_data = make_env_and_datasets(
        options.env_name,
        options.frame_stack,
        action_clip_eps,
        render_videos=options.render_videos and options.video_episodes > 0,
        truncate_dataset_at_first_success=options.truncate_dataset_at_first_success,
    )
    return Benchmark(
        *envs_and_data,
        partial(make_eval_env, options.env_name, frame_stack=options.frame_stack),
    )


ObsUtils.initialize_obs_modality_mapping_from_dict(
    {"low_dim": TRANSPORT_OBSERVATION_KEYS}
)


def _get_max_episode_length(env_name):
    if env_name.startswith("lift"):
        return 300
    if env_name.startswith("can"):
        return 300
    if env_name.startswith("square"):
        return 400
    if env_name.startswith("transport"):
        return 800
    if env_name.startswith("tool_hang"):
        return 1000
    raise ValueError(f"Unsupported environment: {env_name}")


def make_env(env_name, seed=0, render_offscreen=False):
    """
    NOTE: call get_dataset() first so metadata exists before creating the env.
    """
    dataset_path = _check_dataset_exists(env_name)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    max_episode_length = _get_max_episode_length(env_name)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=render_offscreen
    )
    env = RobomimicLowdimWrapper(
        env,
        low_dim_keys=get_low_dim_obs_keys(env_name),
        max_episode_length=max_episode_length,
    )
    if env_name.startswith("transport") and env.observation_space.shape != (59,):
        raise ValueError("Transport requires the full 59D state")
    env.seed(seed)
    return env


def _check_dataset_exists(env_name):
    task, dataset_type, _ = env_name.split("-")
    if dataset_type == "mg":
        file_name = "low_dim_sparse_v15.hdf5"
    else:
        file_name = "low_dim_v15.hdf5"
    download_folder = os.path.join(
        expanduser("~/.robomimic"), task, dataset_type, file_name
    )
    assert os.path.exists(download_folder)
    return download_folder


def get_dataset(env, env_name, truncate_at_first_success=False):
    """Concatenate the demos of ``env_name``'s hdf5 into one ``Dataset``.

    ``truncate_at_first_success`` cuts each demo at the first step whose
    cumulative reward reaches 1 and marks it terminal (DICE-RL's
    ``ph_finetune`` data), matching the online wrapper, which ends the episode
    at the first success; the raw demos instead keep several reward-1 steps.
    """
    dataset_path = _check_dataset_exists(env_name)
    obs_keys = get_low_dim_obs_keys(env_name)
    rm_dataset = h5py.File(dataset_path, "r")
    demos = list(rm_dataset["data"].keys())
    num_demos = len(demos)
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]
    num_timesteps = 0
    for ep in demos:
        num_timesteps += int(rm_dataset[f"data/{ep}/actions"].shape[0])
    print(f"the size of the dataset is {num_timesteps}")
    _ = env.action_space.sample()
    observations = []
    actions = []
    next_observations = []
    terminals = []
    rewards = []
    masks = []
    for ep in demos:
        actions_np = np.array(rm_dataset[f"data/{ep}/actions"])
        obs, next_obs = ([], [])
        for key in obs_keys:
            obs.append(np.array(rm_dataset[f"data/{ep}/obs/{key}"]))
        for key in obs_keys:
            next_obs.append(np.array(rm_dataset[f"data/{ep}/next_obs/{key}"]))
        obs = np.concatenate(obs, axis=-1)
        next_obs = np.concatenate(next_obs, axis=-1)
        if env_name.startswith("transport") and (
            obs.shape[-1] != 59 or next_obs.shape[-1] != 59
        ):
            raise ValueError("Transport dataset must include both arms (59D)")
        dones = np.array(rm_dataset[f"data/{ep}/dones"])
        reward = np.array(rm_dataset[f"data/{ep}/rewards"])
        if truncate_at_first_success:
            success_steps = np.nonzero(np.cumsum(reward) >= 1.0)[0]
            if len(success_steps) > 0:
                end = success_steps[0] + 1
                obs, next_obs, actions_np = (
                    obs[:end],
                    next_obs[:end],
                    actions_np[:end],
                )
                dones, reward = (dones[:end].copy(), reward[:end])
                dones[-1] = 1.0
        observations.append(obs.astype(np.float32))
        actions.append(actions_np.astype(np.float32))
        rewards.append(reward.astype(np.float32))
        terminals.append(dones.astype(np.float32))
        masks.append(1.0 - dones.astype(np.float32))
        next_observations.append(next_obs.astype(np.float32))
    return Dataset.create(
        observations=np.concatenate(observations, axis=0),
        actions=np.concatenate(actions, axis=0),
        rewards=np.concatenate(rewards, axis=0),
        terminals=np.concatenate(terminals, axis=0),
        masks=np.concatenate(masks, axis=0),
        next_observations=np.concatenate(next_observations, axis=0),
    )


class RobomimicLowdimWrapper(gym.Env):
    """
    Environment wrapper for Robomimic environments with state observations.
    Modified from https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/env/robomimic/robomimic_lowdim_wrapper.py
    """

    def __init__(
        self,
        env,
        low_dim_keys,
        normalization_path=None,
        clamp_obs=False,
        init_state=None,
        render_hw=(256, 256),
        render_camera_name="agentview",
        max_episode_length=None,
    ):
        self.env = env
        self.obs_keys = low_dim_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.video_writer = None
        self.clamp_obs = clamp_obs
        self.max_episode_length = max_episode_length
        self.env_step = 0
        self.n_episodes = 0
        self.normalize = normalization_path is not None
        if self.normalize:
            normalization = np.load(normalization_path)
            self.obs_min = normalization["obs_min"]
            self.obs_max = normalization["obs_max"]
            self.action_min = normalization["action_min"]
            self.action_max = normalization["action_max"]
        low = np.full(env.action_dimension, fill_value=-1.0)
        high = np.full(env.action_dimension, fill_value=1.0)
        self.action_space = Box(low=low, high=high, shape=low.shape, dtype=low.dtype)
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(
            low=low, high=high, shape=low.shape, dtype=low.dtype
        )

    def normalize_obs(self, obs):
        obs = 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-06) - 0.5)
        if self.clamp_obs:
            obs = np.clip(obs, -1, 1)
        return obs

    def unnormalize_action(self, action):
        action = (action + 1) / 2
        return action * (self.action_max - self.action_min) + self.action_min

    def get_observation(self):
        raw_obs = self.env.get_observation()
        raw_obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
        if self.normalize:
            return self.normalize_obs(raw_obs)
        return raw_obs

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options={}, **kwargs):
        """Ignore passed-in arguments like seed."""
        self.t = 0
        self.episode_return, self.episode_length = (0, 0)
        self.n_episodes += 1
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        if options is None:
            options = {}
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)
        new_seed = options.get("seed", None)
        if self.init_state is not None:
            self.env.reset_to({"states": self.init_state})
        elif new_seed is not None:
            self.seed(seed=new_seed)
            self.env.reset()
        else:
            self.env.reset()
        return (self.get_observation(), {})

    def step(self, action):
        if self.normalize:
            action = self.unnormalize_action(action)
        raw_obs, reward, done, info = self.env.step(action)
        raw_obs = np.concatenate([raw_obs[key] for key in self.obs_keys], axis=0)
        if self.normalize:
            obs = self.normalize_obs(raw_obs)
        else:
            obs = raw_obs
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)
        self.t += 1
        self.env_step += 1
        self.episode_return += reward
        self.episode_length += 1
        if reward > 0.0:
            done = True
            info["success"] = 1
        else:
            info["success"] = 0
        if done:
            return (obs, reward, True, False, info)
        if self.t >= self.max_episode_length:
            return (obs, reward, False, True, info)
        return (obs, reward, False, False, info)

    def render(self, mode="rgb_array"):
        h, w = self.render_hw
        return self.env.render(
            mode=mode, height=h, width=w, camera_name=self.render_camera_name
        )

    def get_episode_info(self):
        return {"return": self.episode_return, "length": self.episode_length}

    def get_info(self):
        return {"env_step": self.env_step, "n_episodes": self.n_episodes}
