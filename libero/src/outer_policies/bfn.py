"""BFN policy using TASBC's existing cached-VLA collection and replay plumbing."""

import jax
import numpy as np

from nicerl.agents.bfn import BFNAgent, get_config
from nicerl.utils.datasets import ReplayBuffer
from src.outer_policies.base import SamplingMode
from src.outer_policies.cached import CachedPolicy


class BFNPolicy(CachedPolicy):
    def __init__(self, bfn, **kwargs):
        # The inherited collector calls these internal fields _rlt_config/rlt;
        # they hold the critic-only BFN config here. No RLT agent is created.
        kwargs.setdefault("cache_num_base_samples", bfn.get("n_base_samples", 8))
        super().__init__(rlt=bfn, agent="bfn", **kwargs)

    def setup(self, example_obs):
        self.task_instruction = example_obs["task_language"]
        self.action_shape = (
            self.inner_policy.action_horizon,
            self.inner_policy.action_dim,
        )
        config = get_config().to_dict()
        unknown = set(self.rlt) - set(config)
        if unknown:
            raise KeyError(f"Unknown BFN config keys: {sorted(unknown)}")
        config.update(self.rlt)
        config.update(
            action_horizon=self.action_shape[0], raw_action_dim=self.action_shape[1]
        )
        self._rlt_config = config
        self._reference_count = config["n_base_samples"]
        if self.cache_num_base_samples < config["n_base_samples"]:
            raise ValueError("cache_num_base_samples must be at least n_base_samples")
        obs = self._preprocess_rl_obs(example_obs)
        action = np.zeros(self._flat_action_dim, dtype=np.float32)
        self._agent = BFNAgent.create(
            self.seed, self._batch(obs), self._batch(action), config
        )
        self._agent_rng = jax.random.PRNGKey(self.seed)
        self._replay_buffer = ReplayBuffer.create(
            self._example_transition(obs, action), size=self.replay_buffer_size
        )
        print(
            f"[bfn] task={self.task_instruction!r} action_shape={self.action_shape} "
            f"N={config['n_base_samples']} cache={self.cache_num_base_samples} "
            f"selection_rho={config['selection_rho']} backup_rho={config['rho']}"
        )

    def _action_from_cache(self, rl_obs, cache, mode):
        if mode not in (SamplingMode.RANDOM, SamplingMode.TRAIN, SamplingMode.EVAL):
            raise ValueError(f"BFN does not support mode {mode}")
        # Even the trainer's pre-learning RANDOM phase uses BFN. It only gates
        # gradient updates; collection never silently switches to random choice.
        selected = self._select_base_actions(cache)
        action = self._agent.sample_actions(
            self._batch(rl_obs), selected.reshape(1, len(selected), -1)
        )
        return np.asarray(action)[0].reshape(self.action_shape), selected
