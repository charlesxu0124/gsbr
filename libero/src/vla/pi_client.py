from copy import deepcopy
import signal
import logging
from typing import Dict, Optional, Tuple
import time
import requests
import json_numpy
import numpy as np
from src.vla.base import InnerPolicy


class DelayedKeyboardInterrupt:
    def __enter__(self):
        self.signal_received = False
        self.old_handler = signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.signal_received = (sig, frame)
        logging.debug("SIGINT received. Delaying KeyboardInterrupt.")

    def __exit__(self, type, value, traceback):
        signal.signal(signal.SIGINT, self.old_handler)
        if self.signal_received:
            self.old_handler(*self.signal_received)


class PIClient(InnerPolicy):
    NOISE_SHAPES = {"pi05_libero": (10, 32)}

    def __init__(
        self, policy_name: str, host_ip: str = "0.0.0.0", host_port: int = 8000
    ):
        if policy_name not in self.NOISE_SHAPES:
            raise ValueError(
                f"Unknown policy name {policy_name}. Known: {sorted(self.NOISE_SHAPES)}"
            )
        self.policy_name = policy_name
        self.host_ip = host_ip
        self.host_port = host_port

    @property
    def noise_shape(self) -> Optional[Tuple[int, int]]:
        return self.NOISE_SHAPES[self.policy_name]

    @property
    def action_horizon(self) -> int:
        return self.NOISE_SHAPES[self.policy_name][0]

    @property
    def action_dim(self) -> int:
        return 7

    def sample_action(
        self,
        obs: Dict[str, np.ndarray],
        instruction: str,
        noise: Optional[np.ndarray] = None,
        meta: Optional[Dict] = None,
    ) -> np.ndarray:
        return self._request_actions(obs, instruction, noise=noise, meta=meta)

    def sample_actions(
        self,
        obs: Dict[str, np.ndarray],
        instruction: str,
        num_samples: int,
        noise: Optional[np.ndarray] = None,
        meta: Optional[Dict] = None,
    ) -> np.ndarray:
        """Sample ``num_samples`` chunks in one HTTP and model batch."""
        if num_samples < 1:
            raise ValueError("num_samples must be positive")
        if noise is not None:
            noise = np.asarray(noise, dtype=np.float32)
            expected_noise_shape = (num_samples, *self.noise_shape)
            if noise.shape != expected_noise_shape:
                raise ValueError(
                    f"noise has shape {noise.shape}, expected {expected_noise_shape}"
                )
        actions = self._request_actions(
            obs, instruction, noise=noise, meta=meta, batch_size=num_samples
        )
        expected_action_shape = (num_samples, self.action_horizon, self.action_dim)
        if actions.shape != expected_action_shape:
            raise ValueError(
                f"pi server returned shape {actions.shape}, expected {expected_action_shape}"
            )
        return actions

    def _request_actions(
        self,
        obs: Dict[str, np.ndarray],
        instruction: str,
        noise: Optional[np.ndarray] = None,
        meta: Optional[Dict] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        obs_copy = deepcopy(obs)
        if instruction.endswith("."):
            instruction = instruction[:-1]
        obs_copy["task_language"] = instruction.lower()
        url = f"http://{self.host_ip}:{self.host_port}/act"
        payload = {
            "obs": json_numpy.dumps(obs_copy),
            "policy_name": self.policy_name,
            "noise": json_numpy.dumps(noise) if noise is not None else None,
            "meta": json_numpy.dumps(meta) if meta is not None else None,
            "batch_size": batch_size,
        }
        last_error = None
        for attempt in range(6):
            try:
                with DelayedKeyboardInterrupt():
                    response = requests.post(url, json=payload, timeout=(10, 900))
                response.raise_for_status()
                output = response.json()
                if not isinstance(output, str):
                    raise ValueError(f"unexpected server payload: {output!r}")
                break
            except (requests.exceptions.RequestException, ValueError) as e:
                last_error = e
                logging.warning(
                    f"[PIClient] request {attempt + 1}/6 to {url} failed: {e}"
                )
                time.sleep(min(5 * 2**attempt, 60))
        else:
            raise RuntimeError(
                f"pi server at {url} unreachable after 6 attempts: {last_error}"
            )
        json_output = json_numpy.loads(output)
        action = json_output["action"]
        return np.array(action, dtype=np.float32)
