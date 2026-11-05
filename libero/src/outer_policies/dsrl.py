"""DSRL latent-action SAC over a frozen pi0.5 policy."""

import os
import pickle
from typing import Dict, Optional, Sequence, Tuple, Union
import numpy as np
import jax
from gymnasium import Env
from tqdm import tqdm
from nicerl.agents.sac import SACAgent, get_config
from nicerl.utils.datasets import ReplayBuffer
from src.outer_policies.base import RLOuterPolicy, SamplingMode
from src.utils.generalized_action import GeneralizedAction
from src.utils.rollout import Rollout
from src.envs.obswranglers.base import ObsWrangler
from src.vla.base import InnerPolicy


class DSRLPolicy(RLOuterPolicy):
    """Frozen VLA + a SAC actor over the VLA's diffusion latent."""

    def __init__(
        self,
        inner_policy: InnerPolicy,
        obs_wrangler: ObsWrangler,
        sac: Dict,
        cameras: Sequence[str],
        image_resolution: Sequence[int],
        use_state_info: bool = True,
        integration_num_steps: int = 10,
        integration_start_step: int = 0,
        max_action: float = 1.0,
        replay_buffer_size: int = 45000,
        seed: int = 0,
    ):
        self.inner_policy = inner_policy
        self.obs_wrangler = obs_wrangler
        self.sac = dict(sac)
        self.cameras = list(cameras)
        self.image_resolution = tuple(image_resolution)
        self.use_state_info = bool(use_state_info)
        self.integration_num_steps = int(integration_num_steps)
        self.integration_start_step = int(integration_start_step)
        self.max_action = float(max_action)
        if not np.isfinite(self.max_action) or self.max_action <= 0:
            raise ValueError("max_action must be finite and positive")
        self.replay_buffer_size = int(replay_buffer_size)
        self.task_instruction: Optional[str] = None
        self.noise_shape: Optional[Tuple[int, int]] = None
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._agent = None
        self._replay_buffer = None
        self._sac_config = None
        self._agent_rng = None

    def setup(self, example_obs: Dict) -> None:
        """Resolve env-dependent config, then build the SAC agent + replay buffer."""
        if "task_language" not in example_obs:
            raise KeyError(
                f"observation has no 'task_language' key, so DSRL cannot determine the instruction to condition {type(self.inner_policy).__name__} on."
            )
        self.task_instruction = example_obs["task_language"]
        noise_shape = self.inner_policy.noise_shape
        if noise_shape is None:
            raise ValueError(
                f"{type(self.inner_policy).__name__} is not steerable by noise (noise_shape is None), so DSRL has no action space."
            )
        self.noise_shape = tuple(noise_shape)
        ex_obs = self._preprocess_rl_obs(example_obs)
        ex_act = np.zeros(self._noise_vector_dim, dtype=np.float32)
        self._sac_config = self._make_sac_config()
        self._agent = SACAgent.create(
            self.seed, self._batch(ex_obs), self._batch(ex_act), self._sac_config
        )
        self._agent_rng = jax.random.PRNGKey(self.seed)
        self._replay_buffer = ReplayBuffer.create(
            self._example_transition(ex_obs, ex_act), size=self.replay_buffer_size
        )
        print(
            f"[dsrl] setup: instruction={self.task_instruction!r} noise_shape={self.noise_shape} buffer_size={self.replay_buffer_size}"
        )

    def reset(self) -> None:
        pass

    def sample(
        self, obs: Dict, mode: SamplingMode, return_extra: bool = False
    ) -> Tuple[Dict, GeneralizedAction, np.ndarray]:
        """Return (rl_obs, generalized_action, low_level_action_chunk).

        The RL action is a low-dim noise vector; it is repeated across the VLA's
        action horizon and fed into the frozen policy to produce the low-level
        chunk that the env executes open-loop.
        """
        rl_obs = self._preprocess_rl_obs(obs)
        if mode == SamplingMode.RANDOM:
            noise_vec = self._sample_random_noise()
        elif mode in (SamplingMode.TRAIN, SamplingMode.EVAL):
            noise_vec = self._sample_learned_noise(
                rl_obs, deterministic=mode == SamplingMode.EVAL
            )
        else:
            raise ValueError(f"DSRLPolicy does not support sampling mode: {mode}")
        full_noise = self._noise_vector_to_full_noise(noise_vec)
        action_chunk = self.inner_policy.sample_action(
            obs=obs,
            instruction=self.task_instruction,
            noise=full_noise,
            meta={
                "num_steps": self.integration_num_steps,
                "start_step": self.integration_start_step,
            },
        )
        gen_action = GeneralizedAction(
            instruction=self.task_instruction,
            low_level_action_chunk=action_chunk,
            noise=noise_vec,
        )
        if not return_extra:
            return action_chunk
        return (rl_obs, gen_action, action_chunk)

    def _sample_random_noise(self) -> np.ndarray:
        noise = self.rng.uniform(-1.0, 1.0, size=self._noise_vector_dim).astype(
            np.float32
        )
        return noise

    def _sample_learned_noise(
        self, rl_obs: np.ndarray, deterministic: bool
    ) -> np.ndarray:
        if self._agent is None:
            raise RuntimeError(
                "DSRLPolicy.setup() must be called before learned sampling."
            )
        self._agent_rng, sample_rng = jax.random.split(self._agent_rng)
        noise = self._agent.sample_actions(
            self._batch(rl_obs), rng=sample_rng, deterministic=deterministic
        )
        return np.asarray(noise, dtype=np.float32)[0]

    def collect_experience(
        self, env: Env, mode: SamplingMode
    ) -> Tuple[Dict[str, Rollout], Dict]:
        obs, _ = env.reset()
        rollout = Rollout()
        rl_rollout = Rollout()
        done = False
        while not done:
            rl_obs, gen_action, action_chunk = self.sample(obs, mode, return_extra=True)
            next_obs, reward, terminated, truncated, info = env.step(
                action_chunk.copy()
            )
            done = terminated or truncated
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
                action=gen_action,
                reward=reward,
                next_observation=self._preprocess_rl_obs(next_obs),
                terminated=terminated,
                truncated=truncated,
                info={"is_success": info.get("is_success", False)},
            )
            obs = next_obs
        self._add_rollout(rl_rollout)
        return ({"rollout": rollout, "rl_rollout": rl_rollout}, {})

    def _add_rollout(self, rollout: Rollout) -> None:
        """Ingest a collected episode (rl_obs / noise transitions) into the replay buffer."""
        if self._replay_buffer is None:
            raise RuntimeError(
                "DSRLPolicy.setup() must be called before _add_rollout()."
            )
        assert len(rollout) > 0, (
            "rollout must have at least one step to add to the replay buffer."
        )
        assert rollout.terminateds[-1] or rollout.truncateds[-1], (
            "rollout must end with a terminal or truncated step to add to the replay buffer."
        )
        for obs, action, reward, next_obs, terminated, truncated in zip(
            rollout.observations,
            rollout.actions,
            rollout.rewards,
            rollout.next_observations,
            rollout.terminateds,
            rollout.truncateds,
        ):
            self._replay_buffer.add_transition(
                dict(
                    observations=obs,
                    actions=np.asarray(action.noise, dtype=np.float32),
                    rewards=np.float32(reward),
                    masks=np.float32(1.0 - terminated),
                    terminals=np.float32(terminated or truncated),
                    next_observations=next_obs,
                )
            )

    def update(self, num_grad_steps: int) -> Optional[Dict]:
        """Run `num_grad_steps` gradient steps on the learned module; return metrics."""
        if self._agent is None:
            raise RuntimeError("DSRLPolicy.setup() must be called before update().")
        batch_size = self._sac_config["batch_size"]
        if self._replay_buffer.size < batch_size:
            return None
        info = None
        for _ in tqdm(range(num_grad_steps), desc="DSRLPolicy.update"):
            batch = self._replay_buffer.sample_sequence_dict_compatible_optimized_2(
                batch_size, 1, self._sac_config["discount"]
            )
            self._agent, info = self._agent.update(batch)
        return {k: float(v) for k, v in info.items()}

    def _preprocess_rl_obs(self, obs: Dict) -> Dict[str, np.ndarray]:
        """Raw env observation ->
        {"pixels": (H, W, C, num_cameras) uint8, "state": (state_dim,) float}
        ("state" is omitted when `use_state_info` is False).
        """
        rl_obs = self.obs_wrangler.preprocess_rl_obs(
            obs,
            cameras=self.cameras,
            use_state=self.use_state_info,
            image_resolution=self.image_resolution,
        )
        return rl_obs

    @staticmethod
    def _batch(
        x: Union[Dict[str, np.ndarray], np.ndarray],
    ) -> Union[Dict[str, np.ndarray], np.ndarray]:
        """Add a leading batch dim."""
        if isinstance(x, dict):
            return {k: np.asarray(v)[None, ...] for k, v in x.items()}
        return np.asarray(x)[None, ...]

    @staticmethod
    def _example_transition(
        ex_obs: Dict[str, np.ndarray], ex_action: np.ndarray
    ) -> Dict:
        """One example transition; only its shapes/dtypes are used, to size the buffer.

        The schema is dictated by `nicerl`: `Dataset.__init__` requires
        `terminals`, and SAC's losses read exactly these keys. Dict observations
        are supported (the buffer allocates per-leaf via `tree_map`).
        """
        scalar = np.zeros((), dtype=np.float32)
        return dict(
            observations=ex_obs,
            actions=ex_action,
            rewards=scalar,
            masks=scalar,
            terminals=scalar,
            next_observations=ex_obs,
        )

    def _make_sac_config(self) -> Dict:
        """nicerl's SAC defaults with this config's `sac:` block applied on top.

        Returned as a plain dict rather than the `ml_collections.ConfigDict` from
        `get_config()`. Placeholders (`action_dim`, `target_entropy`, ...) come
        out of `to_dict()` as None and are filled in by `SACAgent.create`.
        """
        sac_cfg = get_config().to_dict()
        unknown = sorted(set(self.sac) - set(sac_cfg))
        if unknown:
            raise KeyError(
                f"unknown sac config key(s) {unknown}; nicerl's get_config() defines {sorted(sac_cfg)}"
            )
        sac_cfg.update(self.sac)
        return sac_cfg

    @property
    def _noise_dim(self) -> int:
        return self.noise_shape[1]

    @property
    def _noise_vector_dim(self) -> int:
        """Dimensionality of the RL action space (the noise vector)."""
        horizon = 1
        return self._noise_dim * horizon

    def _noise_vector_to_full_noise(self, noise_vec: np.ndarray) -> np.ndarray:
        """(noise_dim,) or (horizon*noise_dim,) -> (horizon, noise_dim)"""
        noise = np.asarray(noise_vec, dtype=np.float32)
        assert noise.shape == (self._noise_vector_dim,), (
            f"noise_vec.shape={noise.shape} != {(self._noise_vector_dim,)}"
        )
        noise = np.repeat(noise[None, :], self.noise_shape[0], axis=0).copy()
        noise *= self.max_action
        final_noise = self.rng.normal(size=self.noise_shape).astype(np.float32)
        final_noise[:, : self._noise_dim] = noise
        return final_noise

    AGENT_FILE = "agent.pkl"
    RNG_FILE = "rng.pkl"
    BUFFER_FILE = "replay_buffer.pkl"

    def save_state(self, root_dir: str) -> None:
        if self._agent is None:
            raise RuntimeError("DSRLPolicy.setup() must be called before save_state().")
        os.makedirs(root_dir, exist_ok=True)
        self._agent.save(os.path.join(root_dir, self.AGENT_FILE))
        with open(os.path.join(root_dir, self.RNG_FILE), "wb") as f:
            pickle.dump(
                dict(
                    agent_rng=np.asarray(self._agent_rng),
                    np_rng_state=self.rng.bit_generator.state,
                ),
                f,
            )
        self._replay_buffer.save(os.path.join(root_dir, self.BUFFER_FILE))

    def load_state(self, root_dir: str) -> None:
        if self._agent is None:
            raise RuntimeError(
                "DSRLPolicy.setup() must be called before load_state(): restore needs the freshly-built agent as the target structure."
            )
        if not os.path.exists(os.path.join(root_dir, self.AGENT_FILE)):
            print(
                f"[dsrl] load_state: no agent file at {os.path.join(root_dir, self.AGENT_FILE)}"
            )
        else:
            self._agent = self._agent.load(os.path.join(root_dir, self.AGENT_FILE))
        if not os.path.exists(os.path.join(root_dir, self.RNG_FILE)):
            print(
                f"[dsrl] load_state: no rng file at {os.path.join(root_dir, self.RNG_FILE)}"
            )
        else:
            with open(os.path.join(root_dir, self.RNG_FILE), "rb") as f:
                rng_state = pickle.load(f)
            self._agent_rng = jax.numpy.asarray(rng_state["agent_rng"])
            self.rng.bit_generator.state = rng_state["np_rng_state"]
        if not os.path.exists(os.path.join(root_dir, self.BUFFER_FILE)):
            print(
                f"[dsrl] load_state: no replay buffer file at {os.path.join(root_dir, self.BUFFER_FILE)}"
            )
        else:
            self._replay_buffer = ReplayBuffer.load(
                os.path.join(root_dir, self.BUFFER_FILE)
            )
            if self._replay_buffer.max_size != self.replay_buffer_size:
                raise ValueError(
                    f"checkpointed buffer capacity {self._replay_buffer.max_size} != configured replay_buffer_size {self.replay_buffer_size}"
                )
