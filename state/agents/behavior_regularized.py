import copy
from functools import partial
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from agents.sac import SACAgent
from utils.flax_utils import ModuleDict, TrainState
from utils.networks import Actor, GeqLagrangeMultiplier, Value


class BehaviorRegularizedAgent(SACAgent):
    """Shared actor/critic training for frozen-policy behavior constraints."""

    base_policy: any

    def _action_bounds(self):
        raw_bounds = (
            jnp.full((self.config["action_dim"],), self.config["action_clip"])
            .at[-1]
            .set(self.config["gripper_action_clip"])
        )
        return jnp.tile(raw_bounds, self.config["horizon_length"])

    def _clip_actions(self, actions):
        bounds = self._action_bounds()
        return jnp.clip(actions, -bounds, bounds)

    def _scale_actor_actions(self, actions, log_probs=None):
        bounds = self._action_bounds()
        actions = BehaviorRegularizedAgent._clip_actions(self, actions * bounds)
        if log_probs is None:
            return actions
        return (actions, log_probs - jnp.log(bounds).sum())

    def _sample_base_actions_n(self, observations, rng):
        if self.reference_count == 1:
            actions = self.base_policy.sample_actions(observations, seed=rng)
            return self._clip_actions(actions.reshape(observations.shape[0], -1))[None]
        keys = jax.random.split(rng, self.reference_count)
        return jax.vmap(
            lambda key: self._clip_actions(
                self.base_policy.sample_actions(observations, seed=key).reshape(
                    observations.shape[0], -1
                )
            )
        )(keys)

    def _network_ref_actions(self, base_actions_n):
        return base_actions_n

    def _reference_action_dropout(self):
        return self.config["base_action_dropout"]

    def _compute_next_actions(
        self, batch, rng, return_base_actions=False, next_base_actions_n=None
    ):
        """Sample next actions from the actor conditioned on base-policy chunks."""
        base_rng, sample_rng = jax.random.split(rng)
        if self.reference_passthrough:
            if next_base_actions_n is None:
                next_base_actions_n = self._sample_base_actions_n(
                    batch["next_observations"], base_rng
                )
        elif return_base_actions and next_base_actions_n is None:
            next_base_actions_n = self._sample_base_actions_n(
                batch["next_observations"], base_rng
            )
        next_dist = self.network.select("actor")(
            batch["next_observations"],
            temperature=self.config["policy_temperature"],
            ref_actions=self._network_ref_actions(next_base_actions_n)
            if self.reference_passthrough
            else None,
        )
        next_actions, next_log_probs = self._scale_actor_actions(
            *next_dist.sample_and_log_prob(seed=sample_rng)
        )
        return (next_actions, next_log_probs, next_base_actions_n)

    def _critic(self, observations, actions, params=None, base_actions_n=None):
        qs = self.network.select("critic")(
            observations,
            actions,
            ref_actions=self._network_ref_actions(base_actions_n)
            if base_actions_n is not None
            else None,
            params=params,
        )
        if qs.ndim == 1:
            qs = qs[None, :]
        return qs

    def _target_critic(self, observations, actions, base_actions_n=None):
        qs = self.network.select("target_critic")(
            observations,
            actions,
            ref_actions=self._network_ref_actions(base_actions_n)
            if base_actions_n is not None
            else None,
        )
        if qs.ndim == 1:
            qs = qs[None, :]
        return qs

    def _action_diversity(self, actions_n):
        per_state = jnp.sqrt(jnp.mean(jnp.var(actions_n, axis=0), axis=-1) + 1e-08)
        return per_state

    def _nearest_ref_deviation(self, actions, base_actions_n):
        per_candidate_mse = jnp.square(actions[None] - base_actions_n).mean(axis=-1)
        nearest_mse = per_candidate_mse.min(axis=0)
        return jnp.sqrt(nearest_mse + 1e-08)

    def _reference_bc_loss(self, actions, base_actions_n):
        raise NotImplementedError

    def _actor_diagnostic_info(self, actions, base_actions_n):
        ref_diversity = self._action_diversity(base_actions_n)
        ref_deviation = self._nearest_ref_deviation(actions, base_actions_n)
        return {
            "actor/ref_action_diversity": ref_diversity.mean(),
            "actor/ref_action_diversity_max": ref_diversity.max(),
            "actor/rl_ref_deviation": ref_deviation.mean(),
            "actor/rl_ref_deviation_max": ref_deviation.max(),
        }

    def critic_loss(self, batch, grad_params, rng, next_base_actions_n=None):
        """Compute SAC critic loss with optional reference-action conditioning."""
        batch_size = batch["rewards"].shape[0]
        rng, sample_rng, current_ref_rng, subsample_rng = jax.random.split(rng, 4)
        condition_critic = False
        next_actions, next_log_probs, next_base_actions_n = self._compute_next_actions(
            batch,
            sample_rng,
            return_base_actions=condition_critic,
            next_base_actions_n=next_base_actions_n,
        )
        target_next_qs = self._target_critic(
            batch["next_observations"],
            next_actions,
            base_actions_n=next_base_actions_n if condition_critic else None,
        )
        target_next_q = target_next_qs.mean(axis=0) - self.config[
            "critic_target_rho"
        ] * target_next_qs.std(axis=0)
        target_q = (
            batch["rewards"] + self._discount(batch) * batch["masks"] * target_next_q
        )
        if (
            self.config["critic_target_min"] is not None
            or self.config["critic_target_max"] is not None
        ):
            target_q = jnp.clip(
                target_q,
                self.config["critic_target_min"],
                self.config["critic_target_max"],
            )
        base_actions_n = None
        predicted_qs = self._critic(
            batch["observations"],
            batch["actions"],
            params=grad_params,
            base_actions_n=base_actions_n if condition_critic else None,
        )
        target_qs = jnp.broadcast_to(
            target_q, (self.config["critic_ensemble_size"], batch_size)
        )
        critic_loss = jnp.mean((predicted_qs - target_qs) ** 2)
        return (
            critic_loss,
            {
                "critic/critic_loss": critic_loss,
                "critic/predicted_qs": predicted_qs.mean(),
                "critic/target_qs": target_qs.mean(),
                "critic/rewards": batch["rewards"].mean(),
            },
        )

    def actor_loss(self, batch, grad_params, rng, base_actions_n=None):
        """Compute SAC actor and base-policy imitation losses."""
        rng, base_rng, sample_rng, dropout_rng = jax.random.split(rng, 4)
        if base_actions_n is None:
            base_actions_n = self._sample_base_actions_n(
                batch["observations"], base_rng
            )
        actor_ref_actions = None
        full_actor_ref_actions = None
        ref_action_dropout = self._reference_action_dropout()
        if self.reference_passthrough:
            full_actor_ref_actions = self._network_ref_actions(base_actions_n)
            keep = jax.random.bernoulli(
                dropout_rng,
                p=1.0 - ref_action_dropout,
                shape=(full_actor_ref_actions.shape[1], 1),
            )
            actor_ref_actions = full_actor_ref_actions * keep[None].astype(
                full_actor_ref_actions.dtype
            )
        dist = self.network.select("actor")(
            batch["observations"],
            temperature=self.config["policy_temperature"],
            ref_actions=actor_ref_actions,
            params=grad_params,
        )
        actions, log_probs = self._scale_actor_actions(
            *dist.sample_and_log_prob(seed=sample_rng)
        )
        predicted_qs = self._critic(batch["observations"], actions, base_actions_n=None)
        predicted_q = predicted_qs.mean(axis=0)
        q_abs_mean = jax.lax.stop_gradient(jnp.abs(predicted_q).mean())
        q_loss_scale = jnp.asarray(1.0)
        q_loss = -predicted_q.mean() * q_loss_scale
        temperature = self._temperature()
        entropy_loss = jnp.asarray(0.0)
        sac_actor_loss = q_loss + entropy_loss
        bc_mse_loss, constraint_info = self._reference_bc_loss(actions, base_actions_n)
        total_loss = sac_actor_loss + self.config["bc_loss_weight"] * bc_mse_loss
        action_std = dist._distribution.stddev().mean()
        info = {
            "actor/total_loss": total_loss,
            "actor/actor_loss": sac_actor_loss,
            "actor/q_loss": q_loss,
            "actor/q_abs_mean": q_abs_mean,
            "actor/q_loss_scale": q_loss_scale,
            "actor/entropy_loss": entropy_loss,
            "actor/bc_loss": self.config["bc_loss_weight"] * bc_mse_loss,
            "actor/bc_mse_loss": bc_mse_loss,
            "actor/bc_loss_weight": self.config["bc_loss_weight"],
            "actor/temperature": temperature,
            "actor/entropy": -log_probs.mean(),
            "actor/std": action_std,
            "actor/q": predicted_q.mean(),
            "actor/base_action_magnitude": jnp.abs(base_actions_n).mean(),
            "actor/ref_action_dropout": ref_action_dropout,
            "actor/n_base_samples": jnp.asarray(self.reference_count),
        }
        info.update(self._actor_diagnostic_info(actions, base_actions_n))
        info.update(constraint_info)
        return (total_loss, info)

    @partial(jax.jit, static_argnames=["update_critic", "update_actor", "update_flow"])
    def update(
        self,
        batch,
        update_critic=True,
        update_actor=True,
        update_flow=False,
    ):

        rng, loss_rng = jax.random.split(self.rng)
        base_policy, base_info = (self.base_policy, {})
        if update_flow:
            base_policy, base_info = base_policy.update(batch)
        current = self.replace(base_policy=base_policy)
        losses = {}
        batch = current._normalize_batch(batch)
        if update_critic:
            losses["critic"] = partial(current.critic_loss, batch)
        if update_actor:
            losses["actor"] = partial(current.actor_loss, batch)
        if not losses:
            return (current.replace(rng=rng), base_info)
        network, metrics = current.network.apply_loss_fns(
            losses, rng=loss_rng, has_aux=True
        )
        if update_critic:
            current.target_update(network, "critic")
        info = {k: v for group in metrics.values() for k, v in group.items()}
        return (current.replace(network=network, rng=rng), {**info, **base_info})

    @partial(jax.jit, static_argnames=["argmax"])
    def sample_actions(self, observations, seed=None, temperature=1.0, argmax=False):
        """Return a flattened action chunk of shape (..., H*A)."""
        base_rng, action_rng, _ = jax.random.split(seed, 3)
        observations = jnp.asarray(observations)
        unbatched = observations.ndim == 1
        if unbatched:
            observations = observations[None]
        actor_ref_actions = None
        if self.reference_passthrough:
            base_actions_n = self._sample_base_actions_n(observations, base_rng)
            actor_ref_actions = self._network_ref_actions(base_actions_n)
        dist = self.network.select("actor")(
            observations, temperature=temperature, ref_actions=actor_ref_actions
        )
        actions = dist.mode() if argmax else dist.sample(seed=action_rng)
        actions = self._scale_actor_actions(actions)
        return actions[0] if unbatched else actions

    @partial(jax.jit, static_argnames=["argmax"])
    def sample_actions_with_diagnostics(
        self, observations, seed=None, temperature=1.0, argmax=False
    ):
        actions = self.sample_actions(
            observations, seed=seed, temperature=temperature, argmax=argmax
        )
        q = self._critic(jnp.atleast_2d(observations), jnp.atleast_2d(actions)).mean(
            axis=0
        )
        return (actions, {"q": q[0] if observations.ndim == 1 else q})

    @jax.jit
    def sample_base_actions_only(self, observations, seed=None):
        """Sample a chunk from the base policy only, ignoring the actor.

        Used to roll out BC for replay-buffer prefill before any actor training.
        """
        return self.base_policy.sample_actions(observations, seed=seed)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, base_policy):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        config = copy.deepcopy(config)
        raw_action_dim = ex_actions.shape[-1]
        horizon_length = config["horizon_length"]
        chunk_action_dim = horizon_length * raw_action_dim
        assert config["critic_target_rho"] >= 0
        if (
            config["critic_target_min"] is not None
            and config["critic_target_max"] is not None
        ):
            assert config["critic_target_min"] < config["critic_target_max"]
        assert 0.0 <= config["base_action_dropout"] <= 1.0
        ex_chunk_actions = jnp.zeros((ex_observations.shape[0], chunk_action_dim))
        config["horizon_length"] = horizon_length
        config["raw_action_dim"] = raw_action_dim
        config["chunk_action_dim"] = chunk_action_dim
        config["action_dim"] = raw_action_dim
        if config["target_entropy"] is None:
            config["target_entropy"] = (
                -config["target_entropy_multiplier"] * chunk_action_dim
            )
        critic_def = Value(
            network_class=config["value_network_class"],
            network_kwargs=config["value_network_kwargs"],
            num_ensembles=config["critic_ensemble_size"],
        )
        actor_def = Actor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=chunk_action_dim,
            layer_norm=config["actor_layer_norm"],
            tanh_squash=True,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=config["actor_fc_scale"],
        )
        temperature_def = GeqLagrangeMultiplier(
            init_value=config["temperature_init"], constraint_shape=()
        )
        ex_base_actions_n = jnp.zeros(
            (cls.reference_count, ex_observations.shape[0], chunk_action_dim),
            dtype=ex_observations.dtype,
        )
        ex_actor_args = (
            dict(observations=ex_observations, ref_actions=ex_base_actions_n)
            if cls.reference_passthrough
            else (ex_observations,)
        )
        ex_critic_args = (ex_observations, ex_chunk_actions)
        network_info = dict(
            critic=(critic_def, ex_critic_args),
            target_critic=(copy.deepcopy(critic_def), ex_critic_args),
            actor=(actor_def, ex_actor_args),
            temperature=(temperature_def, ()),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        network_def = ModuleDict(networks)
        network_tx = {
            "critic": optax.adam(learning_rate=config["lr"]),
            "actor": optax.adam(learning_rate=config["lr"]),
            "temperature": optax.adam(learning_rate=config["lr"]),
        }
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)
        network.params["modules_target_critic"] = network.params["modules_critic"]
        return cls(
            rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            base_policy=base_policy,
        )

    @partial(jax.jit, static_argnames=["update_critic", "update_actor"])
    def total_loss(
        self,
        batch,
        grad_params,
        rng=None,
        update_critic=True,
        update_actor=True,
    ):

        a_rng, q_rng = jax.random.split(self.rng if rng is None else rng)
        batch = self._normalize_batch(batch)
        loss, info = (0.0, {})
        if update_critic:
            value, metrics = self.critic_loss(batch, grad_params, q_rng)
            loss += value
            info.update(metrics)
        if update_actor:
            value, metrics = self.actor_loss(batch, grad_params, a_rng)
            loss += value
            info.update(metrics)
        return (loss, info)


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="reference",
            actor_update_period=1,
            lr=0.0003,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_network_class="MLP",
            value_network_kwargs=dict(
                hidden_dims=(512, 512, 512, 512), layer_norm=True
            ),
            actor_layer_norm=True,
            discount=0.99,
            soft_target_update_rate=0.005,
            target_entropy=ml_collections.config_dict.placeholder(float),
            target_entropy_multiplier=0.5,
            temperature_init=1.0,
            actor_fc_scale=1.0,
            critic_ensemble_size=10,
            critic_target_rho=0.5,
            critic_target_min=ml_collections.config_dict.placeholder(float),
            critic_target_max=ml_collections.config_dict.placeholder(float),
            policy_temperature=1.0,
            bc_loss_weight=1.0,
            dataset_action_clip_eps=1e-05,
            action_clip=1.0,
            gripper_action_clip=1.0,
            base_action_dropout=0.5,
            horizon_length=ml_collections.config_dict.placeholder(int),
            raw_action_dim=ml_collections.config_dict.placeholder(int),
            chunk_action_dim=ml_collections.config_dict.placeholder(int),
            action_dim=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
