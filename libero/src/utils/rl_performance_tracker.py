from collections import deque
import os
from typing import List
import numpy as np


class PerformanceTracker:
    def __init__(
        self,
        succ_buffer_sizes=[16, 32, 64, 128, 256, 512],
        ema_rates: List[float] = [0.99, 0.95, 0.9, 0.8],
    ):
        self.succ_buffer_sizes = succ_buffer_sizes
        self.succ_buffers = {size: deque(maxlen=size) for size in succ_buffer_sizes}
        self.reward_buffers = {size: deque(maxlen=size) for size in succ_buffer_sizes}

        self.ema_rates = ema_rates
        self.ema_min_counts = {rate: int(1 / (1 - rate)) for rate in ema_rates}
        self.ema_succ_buffer = deque(maxlen=max(self.ema_min_counts.values()) * 2)
        self.ema_reward_buffer = deque(maxlen=max(self.ema_min_counts.values()) * 2)
        self.ema_succ_last = {rate: None for rate in ema_rates}
        self.ema_reward_last = {rate: None for rate in ema_rates}
        self.ema_count = 0
        # New fields remain initialized when loading a legacy checkpoint.
        self.task_episodes = deque(maxlen=10)
        self.training_time_s = 0.0
        self.timed_training_episodes = 0

    def update(self, success: int, reward: float, episode_length=None, control_hz=None):
        if episode_length is not None:
            self.task_episodes.append(
                (int(success), int(episode_length), float(control_hz))
            )
        for size in self.succ_buffer_sizes:
            self.succ_buffers[size].append(success)
            self.reward_buffers[size].append(reward)

        self.ema_succ_buffer.append(success)
        self.ema_reward_buffer.append(reward)
        self.ema_count += 1
        for rate in self.ema_rates:
            if self.ema_count >= self.ema_min_counts[rate]:
                # Initialize EMA with mean of buffer if first time
                if self.ema_succ_last[rate] is None:
                    self.ema_succ_last[rate] = np.mean(self.ema_succ_buffer)
                    self.ema_reward_last[rate] = np.mean(self.ema_reward_buffer)
                # Update EMA using the new success and reward
                else:
                    self.ema_succ_last[rate] = (
                        rate * self.ema_succ_last[rate] + (1 - rate) * success
                    )
                    self.ema_reward_last[rate] = (
                        rate * self.ema_reward_last[rate] + (1 - rate) * reward
                    )

    def get_metrics(self):
        metrics = {}
        if self.task_episodes:
            from src.utils.task_throughput import task_throughput

            successes, lengths, frequencies = zip(*self.task_episodes)
            if len(set(frequencies)) != 1:
                raise ValueError("Control frequency changed within training history")
            metrics.update(
                episode_length=lengths[-1],
                episode_length_last_10=float(np.mean(lengths)),
                success_rate_last_10=float(np.mean(successes)),
                throughput=task_throughput(successes, lengths, frequencies[-1]),
                throughput_window_episodes=len(lengths),
                control_hz=frequencies[-1],
            )
        for size in self.succ_buffer_sizes:
            if len(self.succ_buffers[size]) >= size:
                metrics[f"success_rate_last_{size}"] = np.mean(self.succ_buffers[size])
                metrics[f"reward_avg_last_{size}"] = np.mean(self.reward_buffers[size])

        for rate in self.ema_rates:
            if self.ema_succ_last[rate] is not None:
                metrics[f"success_rate_ema_{rate}"] = self.ema_succ_last[rate]
                metrics[f"reward_avg_ema_{rate}"] = self.ema_reward_last[rate]

        return metrics

    def save(self, path):
        from nicerl.utils.atomic_io import atomic_pickle

        atomic_pickle(self.__dict__, path)

    def load(self, path):

        if not os.path.exists(path):
            print(
                f"Performance stats file {path} not found. Starting with empty stats."
            )
            return
        from nicerl.utils.atomic_io import load_pickle

        data = load_pickle(path)
        self.__dict__.update(data)
