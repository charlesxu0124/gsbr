import collections
import re
import time
import gymnasium
import numpy as np
from gymnasium.spaces import Box


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env, filter_regexes=None):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0
        self.filter_regexes = filter_regexes if filter_regexes is not None else []

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        for filter_regex in self.filter_regexes:
            for key in list(info.keys()):
                if re.match(filter_regex, key) is not None:
                    del info[key]
        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info["total"] = {"timesteps": self.total_timesteps}
        if terminated or truncated:
            info["episode"] = {}
            info["episode"]["final_reward"] = reward
            info["episode"]["return"] = self.reward_sum
            info["episode"]["length"] = self.episode_length
            info["episode"]["duration"] = time.time() - self.start_time
            if hasattr(self.unwrapped, "get_normalized_score"):
                info["episode"]["normalized_return"] = (
                    self.unwrapped.get_normalized_score(info["episode"]["return"])
                    * 100.0
                )
        return (observation, reward, terminated, truncated, info)

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class RewardWrapper(gymnasium.RewardWrapper):
    def __init__(self, env, reward_scale=1.0, reward_bias=0.0):
        super().__init__(env)
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias

    def reward(self, reward):
        return self.reward_scale * reward + self.reward_bias


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)
        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if "goal" in info:
            info["goal"] = np.concatenate([info["goal"]] * self.num_stack, axis=-1)
        return (self.get_observation(), info)

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(ob)
        return (self.get_observation(), reward, terminated, truncated, info)


def clip_dataset_actions(dataset, action_clip_eps):
    if dataset is None or action_clip_eps is None:
        return dataset
    return dataset.copy(
        add_or_replace=dict(
            actions=np.clip(
                dataset["actions"], -1 + action_clip_eps, 1 - action_clip_eps
            )
        )
    )


def make_env_and_datasets(
    env_name,
    frame_stack=None,
    action_clip_eps=1e-5,
    reward_scale=1.0,
    reward_bias=0.0,
    eval_env_only=False,
    render_videos=False,
    truncate_dataset_at_first_success=False,
):
    """Dispatch the default dataset/evaluation loader to its benchmark module."""
    if "singletask" in env_name:
        from envs import ogbench_utils as benchmark

        if eval_env_only:
            return benchmark.make_eval_env(
                env_name, frame_stack, reward_scale, reward_bias
            )
        return benchmark.make_default_env_and_datasets(
            env_name, frame_stack, action_clip_eps, reward_scale, reward_bias
        )
    if "low_dim" in env_name:
        from envs import robomimic_utils as benchmark

        if eval_env_only:
            return benchmark.make_eval_env(
                env_name, frame_stack, render_videos, reward_scale, reward_bias
            )
        return benchmark.make_env_and_datasets(
            env_name,
            frame_stack,
            action_clip_eps,
            render_videos,
            truncate_dataset_at_first_success,
            reward_scale,
            reward_bias,
        )
    raise ValueError(f"Unsupported environment: {env_name}")
