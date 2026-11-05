"""RLT outer policy for cached-sample refinement of a frozen VLA.

The policy samples expensive VLA action chunks only at the Python rollout
boundary.  Cached samples for the current and next observation are persisted
with every replay transition; the JAX learner selects the method's reference set from each cache for
every gradient update, so no VLA request occurs in the update loop.
"""

import os
from typing import Dict, Optional, Sequence, Tuple, Union

import jax
import numpy as np
from gymnasium import Env
from tqdm import tqdm

from nicerl.agents import dice_rl as dice_rl_agent
from nicerl.agents import expo as expo_agent
from nicerl.agents import rlt as rlt_agent
from nicerl.agents import bfn as bfn_agent
from nicerl.agents import gsbr as gsbr_agent
from nicerl.agents import td3bc as td3bc_agent
from nicerl.utils.datasets import ReplayBuffer
from src.envs.obswranglers.base import ObsWrangler
from src.outer_policies.base import RLOuterPolicy, SamplingMode
from src.utils.generalized_action import GeneralizedAction
from src.utils.rollout import Rollout
from src.vla.base import InnerPolicy


# Learners that share this outer policy's cached-sample replay layout.  ``agent``
# selects one; ``rlt`` (the yaml key every launch script uses) is that agent's
# config, validated against its ``get_config()``.
AGENTS = {
    "bfn": (bfn_agent.BFNAgent, bfn_agent.get_config),
    "rlt": (rlt_agent.RLTAgent, rlt_agent.get_config),
    "gsbr": (gsbr_agent.GSBRAgent, gsbr_agent.get_config),
    "td3bc": (td3bc_agent.TD3BCAgent, td3bc_agent.get_config),
    "expo": (expo_agent.EXPOAgent, expo_agent.get_config),
    "dice_rl": (dice_rl_agent.DICERLAgent, dice_rl_agent.get_config),
}


class CachedPolicy(RLOuterPolicy):
    """Frozen base policy plus a learned actor (RLT, EXPO or DICE-RL) over cached
    low-level action chunks."""

    def __init__(
        self,
        inner_policy: InnerPolicy,
        obs_wrangler: ObsWrangler,
        rlt: Dict,
        cameras: Sequence[str],
        image_resolution: Sequence[int],
        agent: str = "rlt",
        use_state_info: bool = True,
        cache_num_base_samples: Optional[int] = None,
        integration_num_steps: int = 10,
        integration_start_step: int = 0,
        replay_buffer_size: int = 45_000,
        seed: int = 0,
    ):
        self.inner_policy = inner_policy
        self.obs_wrangler = obs_wrangler
        self.rlt = dict(rlt)
        if agent not in AGENTS:
            raise ValueError(f"unknown agent {agent!r}; choose from {sorted(AGENTS)}")
        self.agent = agent
        self.cameras = list(cameras)
        self.image_resolution = tuple(image_resolution)
        self.use_state_info = bool(use_state_info)
        if cache_num_base_samples is None:
            # Cache size is independent of the learner's reference count.
            cache_num_base_samples = 16
        self.cache_num_base_samples = int(cache_num_base_samples)
        self.integration_num_steps = int(integration_num_steps)
        self.integration_start_step = int(integration_start_step)
        self.replay_buffer_size = int(replay_buffer_size)
        self.seed = int(seed)

        if self.cache_num_base_samples < 1:
            raise ValueError("cache_num_base_samples must be positive")

        self.task_instruction: Optional[str] = None
        self.action_shape: Optional[Tuple[int, int]] = None
        self.rng = np.random.default_rng(seed)
        self._agent = None
        self._replay_buffer = None
        self._rlt_config = None
        self._agent_rng = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def setup(self, example_obs: Dict) -> None:
        if "task_language" not in example_obs:
            raise KeyError(
                "observation has no 'task_language' key, so RLT cannot condition "
                "the base policy on the task instruction"
            )
        self.task_instruction = example_obs["task_language"]
        self.action_shape = (
            self.inner_policy.action_horizon,
            self.inner_policy.action_dim,
        )

        ex_obs = self._preprocess_rl_obs(example_obs)
        ex_action = np.zeros(self._flat_action_dim, dtype=np.float32)
        self._rlt_config = self._make_rlt_config()
        for key in (
            "n_base_samples",
            "num_next_base_samples",
            "num_actor_base_samples",
        ):
            if self.cache_num_base_samples < self._rlt_config.get(key, 1):
                raise ValueError(
                    f"cache_num_base_samples={self.cache_num_base_samples} must be at "
                    f"least {key}={self._rlt_config[key]}"
                )

        agent_cls, _ = AGENTS[self.agent]
        self._reference_count = self._rlt_config.get(
            "n_base_samples", agent_cls.reference_count
        )
        if self.cache_num_base_samples < self._reference_count:
            raise ValueError("cache is smaller than the reference set")
        self._agent = agent_cls.create(
            self.seed,
            self._batch(ex_obs),
            self._batch(ex_action),
            self._rlt_config,
        )
        self._agent_rng = jax.random.PRNGKey(self.seed)
        self._replay_buffer = ReplayBuffer.create(
            self._example_transition(ex_obs, ex_action),
            size=self.replay_buffer_size,
        )
        print(
            f"[rlt] setup: agent={self.agent} instruction={self.task_instruction!r} "
            f"action_shape={self.action_shape} "
            f"cache={self.cache_num_base_samples} "
            f"n_base_samples={self._reference_count} "
            f"buffer_size={self.replay_buffer_size} "
            f"config={self.rlt}"
        )

    def reset(self) -> None:
        pass

    # ------------------------------------------------------------------ #
    # Acting and base-policy materialization
    # ------------------------------------------------------------------ #
    def _sample_base_action_chunks(self, obs: Dict, num_samples: int) -> np.ndarray:
        """Materialize independent base-policy chunks as (N, H, A)."""
        if self.task_instruction is None or self.action_shape is None:
            raise RuntimeError("CachedPolicy.setup() must be called before sampling")

        noise = None
        if self.inner_policy.noise_shape is not None:
            noise = self.rng.normal(
                size=(num_samples, *self.inner_policy.noise_shape)
            ).astype(np.float32)
        chunks = self.inner_policy.sample_actions(
            obs=obs,
            instruction=self.task_instruction,
            num_samples=num_samples,
            noise=noise,
            meta={
                "num_steps": self.integration_num_steps,
                "start_step": self.integration_start_step,
            },
        )
        chunks = np.asarray(chunks, dtype=np.float32)
        expected_shape = (num_samples, *self.action_shape)
        if chunks.shape != expected_shape:
            raise ValueError(
                f"{type(self.inner_policy).__name__}.sample_actions returned "
                f"shape {chunks.shape}, expected {expected_shape}"
            )
        return np.clip(chunks, -1.0, 1.0)

    def _select_base_actions(self, cached_base_actions: np.ndarray) -> np.ndarray:
        n = self._reference_count
        indices = self.rng.choice(len(cached_base_actions), size=n, replace=False)
        return cached_base_actions[indices]

    def _sample_refined_action(
        self,
        rl_obs: Union[Dict[str, np.ndarray], np.ndarray],
        base_actions: np.ndarray,
        deterministic: bool,
    ) -> np.ndarray:
        if self._agent is None:
            raise RuntimeError(
                "CachedPolicy.setup() must be called before learned sampling"
            )
        self._agent_rng, action_rng = jax.random.split(self._agent_rng)
        flat_references = base_actions.reshape(base_actions.shape[0], -1)
        actions = self._agent.sample_actions(
            self._batch(rl_obs),
            flat_references[None],
            rng=action_rng,
            deterministic=deterministic,
        )
        return np.asarray(actions, dtype=np.float32)[0].reshape(self.action_shape)

    def _action_from_cache(
        self,
        rl_obs: Union[Dict[str, np.ndarray], np.ndarray],
        cache: np.ndarray,
        mode: SamplingMode,
    ) -> tuple[np.ndarray, np.ndarray]:
        selected = self._select_base_actions(cache)
        if mode == SamplingMode.RANDOM:
            action_chunk = selected[self.rng.integers(len(selected))]
        elif mode in (SamplingMode.TRAIN, SamplingMode.EVAL):
            action_chunk = self._sample_refined_action(
                rl_obs,
                selected,
                deterministic=mode == SamplingMode.EVAL,
            )
        else:
            raise ValueError(f"RLTPolicy does not support sampling mode: {mode}")
        return np.clip(action_chunk, -1.0, 1.0), selected

    def sample(self, obs: Dict, mode: SamplingMode) -> np.ndarray:
        """Sample for evaluation; training collection uses the base-sample cache."""
        rl_obs = self._preprocess_rl_obs(obs)
        base_actions = self._sample_base_action_chunks(obs, self._reference_count)
        action_chunk, _ = self._action_from_cache(rl_obs, base_actions, mode)
        return action_chunk

    # ------------------------------------------------------------------ #
    # Collection and learning
    # ------------------------------------------------------------------ #
    def collect_experience(
        self, env: Env, mode: SamplingMode
    ) -> Tuple[Dict[str, Rollout], Dict]:
        obs, _ = env.reset()
        rollout = Rollout()
        rl_rollout = Rollout()
        base_cache = self._sample_base_action_chunks(obs, self.cache_num_base_samples)
        done = False

        while not done:
            rl_obs = self._preprocess_rl_obs(obs)
            action_chunk, selected = self._action_from_cache(rl_obs, base_cache, mode)
            next_obs, reward, terminated, truncated, info = env.step(
                action_chunk.copy()
            )
            done = terminated or truncated

            # Cache next-state samples even at a timeout because the replay mask
            # intentionally bootstraps through truncation, like the DSRL baseline.
            next_base_cache = self._sample_base_action_chunks(
                next_obs, self.cache_num_base_samples
            )
            rollout.add_step(
                observation=obs,
                action=action_chunk,
                reward=reward,
                next_observation=next_obs,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            rl_rollout.add_step(
                observation=rl_obs,
                action=GeneralizedAction(
                    instruction=self.task_instruction,
                    low_level_action_chunk=action_chunk,
                    info={
                        "base_actions": base_cache,
                        "next_base_actions": next_base_cache,
                        "acting_base_actions": selected,
                    },
                ),
                reward=reward,
                next_observation=self._preprocess_rl_obs(next_obs),
                terminated=terminated,
                truncated=truncated,
                info={"is_success": info.get("is_success", False)},
            )
            obs = next_obs
            base_cache = next_base_cache

        self._add_rollout(rl_rollout)
        return {"rollout": rollout, "rl_rollout": rl_rollout}, {}

    def _add_rollout(self, rollout: Rollout) -> None:
        if self._replay_buffer is None:
            raise RuntimeError(
                "CachedPolicy.setup() must be called before _add_rollout()"
            )
        if len(rollout) == 0:
            raise ValueError("rollout must contain at least one transition")
        if not (rollout.terminateds[-1] or rollout.truncateds[-1]):
            raise ValueError("rollout must end at a terminal or timeout")

        for obs, action, reward, next_obs, terminated, truncated in zip(
            rollout.observations,
            rollout.actions,
            rollout.rewards,
            rollout.next_observations,
            rollout.terminateds,
            rollout.truncateds,
        ):
            if action.info is None:
                raise ValueError("RLT rollout action is missing cached base actions")
            base_actions = np.asarray(action.info["base_actions"], dtype=np.float32)
            next_base_actions = np.asarray(
                action.info["next_base_actions"], dtype=np.float32
            )
            expected_cache_shape = (self.cache_num_base_samples, *self.action_shape)
            if base_actions.shape != expected_cache_shape:
                raise ValueError(
                    f"base action cache has shape {base_actions.shape}, expected "
                    f"{expected_cache_shape}"
                )
            if next_base_actions.shape != expected_cache_shape:
                raise ValueError(
                    f"next base action cache has shape {next_base_actions.shape}, "
                    f"expected {expected_cache_shape}"
                )
            self._replay_buffer.add_transition(
                dict(
                    observations=obs,
                    actions=np.asarray(
                        action.low_level_action_chunk, dtype=np.float32
                    ).reshape(-1),
                    base_actions=base_actions.reshape(self.cache_num_base_samples, -1),
                    rewards=np.float32(reward),
                    masks=np.float32(1.0 - terminated),
                    terminals=np.float32(terminated or truncated),
                    next_observations=next_obs,
                    next_base_actions=next_base_actions.reshape(
                        self.cache_num_base_samples, -1
                    ),
                )
            )

    def update(self, num_grad_steps: int) -> Optional[Dict]:
        if self._agent is None:
            raise RuntimeError("CachedPolicy.setup() must be called before update()")
        batch_size = self._rlt_config["batch_size"]
        if self._replay_buffer.size < batch_size:
            return None

        info = None
        for _ in tqdm(range(num_grad_steps), desc="RLTPolicy.update"):
            batch = self._replay_buffer.sample(batch_size)
            self._agent, info = self._agent.update(batch)
        return {key: float(value) for key, value in info.items()}

    # ------------------------------------------------------------------ #
    # Helpers and persistence
    # ------------------------------------------------------------------ #
    def _preprocess_rl_obs(self, obs: Dict):
        return self.obs_wrangler.preprocess_rl_obs(
            obs,
            cameras=self.cameras,
            use_state=self.use_state_info,
            image_resolution=self.image_resolution,
        )

    @staticmethod
    def _batch(x):
        if isinstance(x, dict):
            return {key: np.asarray(value)[None] for key, value in x.items()}
        return np.asarray(x)[None]

    def _example_transition(self, ex_obs: Dict, ex_action: np.ndarray) -> Dict:
        scalar = np.zeros((), dtype=np.float32)
        references = np.zeros(
            (self.cache_num_base_samples, self._flat_action_dim),
            dtype=np.float32,
        )
        return dict(
            observations=ex_obs,
            actions=ex_action,
            base_actions=references,
            rewards=scalar,
            masks=scalar,
            terminals=scalar,
            next_observations=ex_obs,
            next_base_actions=references,
        )

    def _make_rlt_config(self) -> Dict:
        _, get_config = AGENTS[self.agent]
        config = get_config().to_dict()
        unknown = sorted(set(self.rlt) - set(config))
        if unknown:
            raise KeyError(
                f"unknown {self.agent} config key(s) {unknown}; its get_config() "
                f"defines {sorted(config)}"
            )
        config.update(self.rlt)
        config["action_horizon"] = self.action_shape[0]
        config["raw_action_dim"] = self.action_shape[1]
        return config

    @property
    def _flat_action_dim(self) -> int:
        if self.action_shape is None:
            raise RuntimeError("CachedPolicy.setup() must resolve action_shape first")
        return int(np.prod(self.action_shape))

    AGENT_FILE = "agent.pkl"
    RNG_FILE = "rng.pkl"
    BUFFER_FILE = "replay_buffer.pkl"

    def save_state(self, root_dir: str) -> None:
        if self._agent is None:
            raise RuntimeError(
                "CachedPolicy.setup() must be called before save_state()"
            )
        os.makedirs(root_dir, exist_ok=True)
        self._agent.save(os.path.join(root_dir, self.AGENT_FILE))
        from nicerl.utils.atomic_io import atomic_pickle

        atomic_pickle(
            {
                "agent_rng": np.asarray(self._agent_rng),
                "np_rng_state": self.rng.bit_generator.state,
            },
            os.path.join(root_dir, self.RNG_FILE),
        )
        self._replay_buffer.save(os.path.join(root_dir, self.BUFFER_FILE))

    def load_state(self, root_dir: str) -> None:
        if self._agent is None:
            raise RuntimeError(
                "CachedPolicy.setup() must be called before load_state()"
            )

        agent_path = os.path.join(root_dir, self.AGENT_FILE)
        if os.path.exists(agent_path):
            self._agent = self._agent.load(agent_path)
        else:
            print(f"[rlt] load_state: no agent file at {agent_path}")

        rng_path = os.path.join(root_dir, self.RNG_FILE)
        if os.path.exists(rng_path):
            from nicerl.utils.atomic_io import load_pickle

            rng_state = load_pickle(rng_path)
            self._agent_rng = jax.numpy.asarray(rng_state["agent_rng"])
            self.rng.bit_generator.state = rng_state["np_rng_state"]
        else:
            print(f"[rlt] load_state: no rng file at {rng_path}")

        buffer_path = os.path.join(root_dir, self.BUFFER_FILE)
        if os.path.exists(buffer_path):
            replay_buffer = ReplayBuffer.load(buffer_path)
            if replay_buffer.max_size != self.replay_buffer_size:
                raise ValueError(
                    f"checkpointed buffer capacity {replay_buffer.max_size} "
                    f"!= configured replay_buffer_size {self.replay_buffer_size}"
                )
            for key in ("base_actions", "next_base_actions"):
                cached_samples = replay_buffer[key].shape[1]
                if cached_samples != self.cache_num_base_samples:
                    raise ValueError(
                        f"checkpointed {key} cache has {cached_samples} samples "
                        f"!= configured cache_num_base_samples {self.cache_num_base_samples}. "
                        "Use a matching checkpoint or collect a fresh replay buffer."
                    )
            self._replay_buffer = replay_buffer
        else:
            print(f"[rlt] load_state: no replay buffer file at {buffer_path}")
