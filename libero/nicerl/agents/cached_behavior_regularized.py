"""Shared actor/critic training regularized by cached base-policy actions.

The frozen pi0.5 policy is served separately. Its action chunks are collected
by the outer policy and stored in replay as ``base_actions`` and
``next_base_actions`` for the method-specific behavior constraints.
"""

import copy
from functools import partial
from typing import Any, Dict, Optional
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from nicerl.agents.sac import SACAgent, get_config as get_sac_config
from nicerl.utils.encoders.networks import (
    PixelStateEncoder,
    ObservationActionMultiplexer,
)
from nicerl.utils.flax_utils import ModuleDict, TrainState
from nicerl.utils.networks import LogParam, MLP, TanhNormal, Value


class CachedBehaviorRegularizedAgent(SACAgent):
    """Actor/critic learning with frozen pi0.5 samples stored in replay."""

    @property
    def reference_count(self):
        return self.config["n_base_samples"]

    @staticmethod
    def _flatten_reference_actions(base_actions: jnp.ndarray) -> jnp.ndarray:
        """(B, N, chunk_action_dim) -> (B, N * chunk_action_dim)."""
        return base_actions.reshape(base_actions.shape[0], -1)

    def _subsample_base_actions(
        self, cached_base_actions: jnp.ndarray, rng: jax.Array, n: Optional[int] = None
    ) -> jnp.ndarray:
        """Draw an independent, without-replacement n-of-cache set per state
        (``n_base_samples`` by default)."""
        n = self.reference_count if n is None else int(n)
        if cached_base_actions.ndim != 3:
            raise ValueError(
                f"cached base actions must have shape (batch, cache, action_dim), got {cached_base_actions.shape}"
            )
        if cached_base_actions.shape[-1] != self.config["action_dim"]:
            raise ValueError(
                f"cached base action dim {cached_base_actions.shape[-1]} does not match agent action dim {self.config['action_dim']}"
            )
        if cached_base_actions.shape[1] < n:
            raise ValueError(
                f"replay only has {cached_base_actions.shape[1]} base samples, but {n} were requested"
            )
        scores = jax.random.uniform(rng, cached_base_actions.shape[:2])
        indices = jnp.argsort(scores, axis=1)[:, :n]
        return jnp.take_along_axis(cached_base_actions, indices[..., None], axis=1)

    def _td_target(self, batch: Dict, next_q: jnp.ndarray) -> jnp.ndarray:
        """``r + discount * mask * next_q`` per flattened chunk, optionally clipped to
        the known return bounds (LIBERO: -1 per chunk until success, 0 at success,
        discount 0.975 -> [-40, 0]) via ``critic_target_min/max``, as
        behavior-constraint's keys of the same name.  None leaves it unclipped."""
        target_q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_q
        if (
            self.config["critic_target_min"] is not None
            or self.config["critic_target_max"] is not None
        ):
            target_q = jnp.clip(
                target_q,
                self.config["critic_target_min"],
                self.config["critic_target_max"],
            )
        return target_q

    def _reference_bc_loss(self, actions, base_actions):
        raise NotImplementedError

    def _actor_distribution(
        self, observations: Any, base_actions: jnp.ndarray, params: Optional[Any] = None
    ):
        return self.network.select("actor")(
            observations,
            self._flatten_reference_actions(base_actions)
            if self.reference_passthrough
            else None,
            params=params,
        )

    def _sample_next_actions(
        self, batch: Dict, rng: jax.Array
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        subset_rng, action_rng = jax.random.split(rng)
        next_base_actions = self._subsample_base_actions(
            batch["next_base_actions"], subset_rng
        )
        dist = self._actor_distribution(batch["next_observations"], next_base_actions)
        next_actions, log_probs = dist.sample_and_log_prob(seed=action_rng)
        return (next_actions, log_probs)

    def critic_loss(self, batch, grad_params, rng):
        rng, action_rng = jax.random.split(rng)
        next_actions, next_log_probs = self._sample_next_actions(batch, action_rng)
        next_qs = self.network.select("target_critic")(
            batch["next_observations"], next_actions
        )
        next_q = self._aggregate_target_q(next_qs)
        target_q = self._td_target(batch, next_q)
        qs = self.network.select("critic")(
            batch["observations"], batch["actions"], params=grad_params
        )
        critic_loss = jnp.square(qs - target_q).mean()
        return (
            critic_loss,
            {
                "critic/total_loss": critic_loss,
                "critic/critic_loss": critic_loss,
                "critic/q_mean": qs.mean(),
                "critic/q_max": qs.max(),
                "critic/q_min": qs.min(),
                "critic/target_q": target_q.mean(),
            },
        )

    def actor_loss(self, batch, grad_params, rng):
        subset_rng, dropout_rng, action_rng = jax.random.split(rng, 3)
        base_actions = self._subsample_base_actions(batch["base_actions"], subset_rng)
        keep = jax.random.bernoulli(
            dropout_rng,
            p=1.0 - self.config["base_action_dropout"],
            shape=(base_actions.shape[0], 1, 1),
        )
        actor_base_actions = base_actions * keep.astype(base_actions.dtype)
        dist = self._actor_distribution(
            batch["observations"], actor_base_actions, params=grad_params
        )
        actions, log_probs = dist.sample_and_log_prob(seed=action_rng)
        qs = self.network.select("critic")(batch["observations"], actions)
        q = qs.mean(axis=0)
        alpha = self.network.select("alpha")()
        sac_actor_loss = -q.mean()
        entropy = -log_probs.mean()
        alpha_loss = jnp.asarray(0.0)
        bc_loss, constraint_info = self._reference_bc_loss(actions, base_actions)
        total_loss = (
            sac_actor_loss + alpha_loss + self.config["bc_loss_weight"] * bc_loss
        )
        diversity = jnp.sqrt(jnp.mean(jnp.var(base_actions, axis=1), axis=-1) + 1e-08)
        action_std = dist._distribution.stddev()
        info = {
            "actor/total_loss": total_loss,
            "actor/actor_loss": sac_actor_loss,
            "actor/alpha_loss": alpha_loss,
            "actor/alpha": alpha,
            "actor/entropy": entropy,
            "actor/action_std": action_std.mean(),
            "actor/q": q.mean(),
            "actor/bc_loss": bc_loss,
            "actor/bc_loss_weight": jnp.asarray(self.config["bc_loss_weight"]),
            "actor/base_action_dropout": jnp.asarray(
                self.config["base_action_dropout"]
            ),
            "actor/base_action_magnitude": jnp.abs(base_actions).mean(),
            "actor/ref_action_diversity": diversity.mean(),
            "actor/n_base_samples": jnp.asarray(self.reference_count),
        }
        info.update({f"actor/{key}": value for key, value in constraint_info.items()})
        return (total_loss, info)

    def total_loss(self, batch, grad_params, rng=None):
        rng = self.rng if rng is None else rng
        actor_rng, critic_rng = jax.random.split(rng)
        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        return (critic_loss + actor_loss, {**critic_info, **actor_info})

    @partial(jax.jit, static_argnames=("deterministic",))
    def sample_actions(self, observations, base_actions, rng=None, deterministic=False):
        """Sample refined flattened chunks from an already selected N-set."""
        if base_actions.shape[1] != self.reference_count:
            raise ValueError(
                f"sample_actions expects {self.reference_count} references, got {base_actions.shape[1]}"
            )
        dist = self._actor_distribution(observations, base_actions)
        if deterministic:
            actions = dist.mode()
        else:
            if rng is None:
                raise ValueError("rng is required for stochastic RLT sampling")
            actions = dist.sample(seed=rng)
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create RLT over a flattened low-level action chunk."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        config = copy.deepcopy(config)
        action_dim = ex_actions.shape[-1]
        n_base_samples = int(cls.reference_count)
        if n_base_samples < 1:
            raise ValueError("n_base_samples must be positive")
        if not 0.0 <= config["base_action_dropout"] <= 1.0:
            raise ValueError("base_action_dropout must be in [0, 1]")
        if config["action_chunking"] or config["chunk_size"] != 1:
            raise ValueError(
                "RLT replay stores each flattened action chunk as one transition; set action_chunking=false and chunk_size=1"
            )
        expected_action_dim = config["action_horizon"] * config["raw_action_dim"]
        if action_dim != expected_action_dim:
            raise ValueError(
                f"flattened action dim {action_dim} != action_horizon * raw_action_dim ({config['action_horizon']} * {config['raw_action_dim']})"
            )
        if config["target_entropy"] is None:
            config["target_entropy"] = -config["target_entropy_multiplier"] * action_dim
        config["action_dim"] = action_dim
        actor_head = TanhNormal(
            partial(
                MLP,
                hidden_dims=config["actor_hidden_dims"],
                activate_final=True,
                layer_norm=config["actor_layer_norm"],
            ),
            action_dim,
        )
        ex_reference_actions = (
            jnp.zeros(
                (ex_actions.shape[0], n_base_samples * action_dim),
                dtype=ex_actions.dtype,
            )
            if cls.reference_passthrough
            else None
        )
        network = build_actor_critic(
            config,
            ex_observations,
            ex_actions,
            actor_head=actor_head,
            ex_actor_actions=ex_reference_actions,
            tx=optax.adam(learning_rate=config["lr"]),
            init_rng=init_rng,
        )
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def build_actor_critic(
    config, ex_observations, ex_actions, actor_head, ex_actor_actions, tx, init_rng
):
    """Pixel-encoder critic ensemble, target critic, actor and alpha as one TrainState.

    Shared by RLT, EXPO and DICE-RL: the critic is ``Value(num_qs)`` over
    (encoded observation, flattened chunk); the actor is ``actor_head`` over the
    concatenation of the encoded observation and ``ex_actor_actions`` (the
    reference actions the actor conditions on).  Both encoders are separate copies.
    """
    encoder_def = None
    if config["use_encoder"]:
        encoder_def = PixelStateEncoder(
            features=config["cnn_features"],
            strides=config["cnn_strides"],
            padding=config["cnn_padding"],
            use_bottleneck=config["use_bottleneck"],
            bottleneck_dim=config["bottleneck_dim"],
        )
    critic_def = ObservationActionMultiplexer(
        network=Value(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["value_layer_norm"],
            num_ensembles=config["num_qs"],
        ),
        obs_encoder=copy.deepcopy(encoder_def),
        action_encoder=None,
    )
    actor_def = ObservationActionMultiplexer(
        network=actor_head,
        obs_encoder=copy.deepcopy(encoder_def),
        action_encoder=None,
        do_combine_encoding=True,
    )
    network_info = {
        "critic": (critic_def, (ex_observations, ex_actions)),
        "target_critic": (copy.deepcopy(critic_def), (ex_observations, ex_actions)),
        "actor": (actor_def, (ex_observations, ex_actor_actions)),
        "alpha": (LogParam(), ()),
    }
    network_def = ModuleDict({key: value[0] for key, value in network_info.items()})
    network_params = network_def.init(
        init_rng, **{key: value[1] for key, value in network_info.items()}
    )["params"]
    network = TrainState.create(network_def, network_params, tx=tx)
    network.params["modules_target_critic"] = network.params["modules_critic"]
    return network


def get_config():
    """SAC defaults plus the RLT behavior-constraint settings."""
    config = get_sac_config()
    config.update(
        ml_collections.ConfigDict(
            dict(
                agent_name="cached_behavior_regularized",
                bc_loss_weight=10.0,
                base_action_dropout=0.5,
                critic_target_min=None,
                critic_target_max=None,
                action_chunking=False,
                chunk_size=1,
                action_horizon=ml_collections.config_dict.placeholder(int),
                raw_action_dim=ml_collections.config_dict.placeholder(int),
            )
        )
    )
    return config
