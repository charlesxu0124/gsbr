"""Generic outer-policy interface (frozen pretrained policy + learned module).

The training loop in `src/train.py` is written against *these* classes only —
it never imports a concrete policy. Hydra decides which subclass gets built
(`_target_` in the config), so a new algorithm means a new subclass plus a new
config, and zero changes to the loop.

The policy interface splits into three
groups:

  acting       `setup` / `reset` / `sample`
  learning     `add_rollout` / `update`
  persistence  `save_state` / `load_state`

and the policy carries two kinds of state:
  * *persistent* — replay buffer, learned weights: written by `save_state` and
    restored by `load_state`;
  * *per-episode* — history, step counters: cleared by `reset`, never saved.
"""

from enum import Enum
from typing import Dict, Optional, Tuple
import numpy as np
from gymnasium import Env

from src.utils.rollout import Rollout


class SamplingMode(Enum):
    """Enum for different sampling modes."""

    TRAIN = "train"
    EVAL = "eval"
    RANDOM = "random"


class OuterPolicy:
    def setup(self, example_obs: Dict) -> None:
        """Resolve anything that needs a live env before acting.

        Called once by the train loop with a raw observation, so env-dependent
        shapes (obs dims, action dim, task instruction) can be read off the env
        instead of being hard-coded in the config file.
        """
        pass

    def sample(self, obs: Dict, mode: SamplingMode) -> np.ndarray:
        """Map a raw env observation to an action chunk (horizon, action_dim)."""
        raise NotImplementedError

    def reset(self) -> None:
        """Called at the beginning of each episode to reset per-episode state."""
        pass


class RLOuterPolicy(OuterPolicy):
    """An outer policy whose learned module is trained with RL.

    All algorithm-specific logic (replay-buffer schema, gradient stepping,
    checkpoint format) lives behind these methods; the train loop only decides
    *when* to learn and *how many* gradient steps to spend.
    """

    def collect_experience(
        self, env: Env, mode: SamplingMode
    ) -> Tuple[Dict[str, Rollout], Dict]:
        """Run one episode in `env` and return the collected rollout(s).
        The first returned dict maps from rollout name to rollout data (for logging).
        The second returned dict contains additional metadata.
        """
        raise NotImplementedError

    def update(self, num_grad_steps: int) -> Optional[Dict]:
        """Take `num_grad_steps` gradient steps; return metrics to log (or None)."""
        raise NotImplementedError

    def save_state(self, root_dir: str) -> None:
        """Persist state that must survive a restart (buffer + weights)."""
        raise NotImplementedError

    def load_state(self, root_dir: str) -> None:
        """Restore what `save_state` wrote."""
        raise NotImplementedError
