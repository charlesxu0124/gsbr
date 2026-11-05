import numpy as np

from src.vla.pi_client import PIClient
from src.outer_policies.base import OuterPolicy


class PiPolicy(OuterPolicy):
    """An outer policy that wraps a pretrained VLA (inner) policy served by
    `src/vla/pi_server.py`. The inner policy is queried over HTTP.
    """

    def __init__(
        self,
        pi_client: PIClient,
        chunk_size: int,
    ):

        self.inner_policy = pi_client
        self.chunk_size = chunk_size
        self.task_instruction = None

    def setup(self, example_obs: dict) -> None:
        assert "task_language" in example_obs, (
            "PiPolicy requires 'task_language' in the observation dict."
        )
        self.task_instruction = example_obs["task_language"]

    def sample(
        self,
        obs: dict,
        mode: str,
    ) -> np.ndarray:
        """Map a raw env observation to an action chunk (horizon, action_dim)."""
        assert self.task_instruction is not None, (
            "PiPolicy requires setup() to be called before sample()."
        )
        action_chunk = self.inner_policy.sample_action(
            obs=obs,
            instruction=self.task_instruction,
        )
        action_chunk = action_chunk[: self.chunk_size]  # (horizon, action_dim)
        return action_chunk
