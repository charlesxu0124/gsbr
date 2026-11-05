"""DSRL noise-as-action with an externally restored, frozen BC flow.

Q_a uses TD on dataset/replay action chunks. Q_z distills Q_a at Gaussian
prior latents; only Q_z supplies the SAC actor gradient. Latent coordinates
are physical z throughout, while SAC entropy is measured before scaling
(u = z / noise_scale), matching the official implementation's convention.
"""

import copy
from functools import partial
from typing import Any
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from agents.expo import EXPOAgent
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, GeqLagrangeMultiplier, Value


class DSRLNAAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()
    base_policy: Any
    _normalize_batch = EXPOAgent._normalize_batch
    _discount = EXPOAgent._discount
    _critic = EXPOAgent._critic
    _temperature = EXPOAgent._temperature
    _action_bounds = EXPOAgent._action_bounds
    _clip_actions = EXPOAgent._clip_actions
    target_update = EXPOAgent.target_update
    sample_base_actions_only = EXPOAgent.sample_base_actions_only

    def _expand_latent(self, z):
        return z

    def _decode(self, observations, z):
        actions = self.base_policy.sample_actions(
            observations, seed=self.rng, init_noise=self._expand_latent(z)
        )
        return self._clip_actions(actions.reshape(observations.shape[0], -1))

    def _sample_latent(
        self, observations, rng, params=None, argmax=False, temperature=1.0
    ):
        dist = self.network.select("actor")(
            observations, params=params, temperature=jnp.maximum(temperature, 1e-06)
        )
        u, logp = dist.sample_and_log_prob(seed=rng)
        if argmax:
            u = dist.mode()
        u = jnp.where(jnp.asarray(temperature) == 0, dist.mode(), u)
        return (self.config["noise_scale"] * u, logp)

    def _aggregate(self, qs):
        kind = "mean_std"
        if kind == "min":
            return qs.min(axis=0)
        if kind == "mean_std":
            return qs.mean(axis=0) - self.config["critic_target_rho"] * qs.std(axis=0)
        return qs.mean(axis=0)

    def _td_target(self, batch, rng):
        z, logp = self._sample_latent(batch["next_observations"], rng)
        actions = self._decode(batch["next_observations"], z)
        q = self._aggregate(
            self._critic(
                batch["next_observations"], actions, critic_name="target_critic"
            )
        )
        q = q - self._temperature() * logp
        target = batch["rewards"] + self._discount(batch) * batch["masks"] * q
        if (
            self.config["critic_target_min"] is not None
            or self.config["critic_target_max"] is not None
        ):
            target = jnp.clip(
                target,
                self.config["critic_target_min"],
                self.config["critic_target_max"],
            )
        return jax.lax.stop_gradient(target)

    def critic_loss(self, batch, grad_params, rng):
        target = self._td_target(batch, rng)
        qs = self._critic(batch["observations"], batch["actions"], params=grad_params)
        loss = jnp.square(qs - target[None]).mean()
        return (
            loss,
            {
                "critic/critic_loss": loss,
                "critic/target_qs": target.mean(),
                "critic/predicted_qs": qs.mean(),
            },
        )

    def noise_critic_loss(self, batch, grad_params, rng):
        z = jax.random.normal(
            rng, (batch["observations"].shape[0], self.config["latent_dim"])
        )
        actions = self._decode(batch["observations"], z)
        teacher = jax.lax.stop_gradient(self._critic(batch["observations"], actions))
        qs = self._critic(
            batch["observations"], z, params=grad_params, critic_name="noise_critic"
        )
        loss = jnp.square(qs - teacher).mean()
        return (
            loss,
            {"noise_critic/loss": loss, "noise_critic/teacher_q": teacher.mean()},
        )

    def actor_loss(self, batch, grad_params, rng):
        z, logp = self._sample_latent(batch["observations"], rng, params=grad_params)
        qs = self._critic(batch["observations"], z, critic_name="noise_critic")
        q = self._aggregate(qs)
        loss = (self._temperature() * logp - q).mean()
        return (
            loss,
            {
                "actor/actor_loss": loss,
                "actor/q": q.mean(),
                "actor/entropy": -logp.mean(),
                "actor/latent_abs": jnp.abs(z).mean(),
            },
        )

    def temperature_loss(self, batch, grad_params, rng):
        _, logp = self._sample_latent(batch["observations"], rng)
        entropy = jax.lax.stop_gradient(-logp.mean())
        loss = self.network.select("temperature")(
            lhs=entropy, rhs=self.config["target_entropy"], params=grad_params
        )
        return (
            loss,
            {"temperature/loss": loss, "temperature/value": self._temperature()},
        )

    def _losses(self, batch, update_critic, update_actor):
        result = {}
        if update_critic:
            result.update(
                critic=partial(self.critic_loss, batch),
                noise_critic=partial(self.noise_critic_loss, batch),
            )
        if update_actor:
            result.update(
                actor=partial(self.actor_loss, batch),
                temperature=partial(self.temperature_loss, batch),
            )
        return result

    @partial(jax.jit, static_argnames=["update_critic", "update_actor"])
    def total_loss(
        self,
        batch,
        grad_params,
        rng=None,
        update_critic=True,
        update_actor=True,
    ):

        losses = self._losses(self._normalize_batch(batch), update_critic, update_actor)
        keys = jax.random.split(self.rng if rng is None else rng, len(losses))
        total, info = (jnp.array(0.0), {})
        for fn, key in zip(losses.values(), keys):
            loss, metrics = fn(grad_params, key)
            total += loss
            info.update(metrics)
        return (total, info)

    @partial(jax.jit, static_argnames=["update_critic", "update_actor"])
    def update(self, batch, update_critic=True, update_actor=True):

        new_rng, rng = jax.random.split(self.rng)
        losses = self._losses(self._normalize_batch(batch), update_critic, update_actor)
        if not losses:
            return (self.replace(rng=new_rng), {})
        active = self.network.replace(
            tx={k: self.network.tx[k] for k in losses},
            opt_state={k: self.network.opt_state[k] for k in losses},
        )
        updated, metrics = active.apply_loss_fns(losses, rng=rng, has_aux=True)
        network = updated.replace(
            tx=self.network.tx,
            opt_state={**self.network.opt_state, **updated.opt_state},
        )
        if update_critic:
            self.target_update(network, "critic")
        return (
            self.replace(network=network, rng=new_rng),
            {k: v for info in metrics.values() for k, v in info.items()},
        )

    @partial(jax.jit, static_argnames=["argmax"])
    def sample_actions(self, observations, seed=None, temperature=1.0, argmax=False):
        observations = jnp.asarray(observations)
        unbatched = observations.ndim == 1
        observations = observations[None] if unbatched else observations
        z, _ = self._sample_latent(
            observations,
            self.rng if seed is None else seed,
            argmax=argmax,
            temperature=temperature,
        )
        actions = self._decode(observations, z)
        return actions[0] if unbatched else actions

    @partial(jax.jit, static_argnames=["argmax"])
    def sample_actions_with_diagnostics(
        self, observations, seed=None, temperature=1.0, argmax=False
    ):
        actions = self.sample_actions(observations, seed, temperature, argmax)
        obs = jnp.atleast_2d(observations)
        q = self._critic(obs, jnp.atleast_2d(actions)).mean(axis=0)
        return (actions, {"q": q[0] if observations.ndim == 1 else q})

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, base_policy):
        config = copy.deepcopy(dict(config))
        assert config["noise_scale"] > 0
        assert config["critic_target_rho"] >= 0
        assert base_policy.config["horizon_length"] == config["horizon_length"]
        config["action_dim"] = ex_actions.shape[-1]
        chunk_dim = config["action_dim"] * config["horizon_length"]
        config["latent_dim"] = chunk_dim if "full" == "full" else config["action_dim"]
        rng, init_rng = jax.random.split(jax.random.PRNGKey(seed))
        critic = Value(
            network_class=config["value_network_class"],
            network_kwargs=config["value_network_kwargs"],
            num_ensembles=config["critic_ensemble_size"],
        )
        actor = Actor(
            hidden_dims=config["actor_hidden_dims"],
            action_dim=config["latent_dim"],
            layer_norm=config["actor_layer_norm"],
            tanh_squash=True,
            state_dependent_std=True,
            const_std=False,
            final_fc_init_scale=1.0,
        )
        modules = ModuleDict(
            dict(
                critic=critic,
                target_critic=copy.deepcopy(critic),
                noise_critic=copy.deepcopy(critic),
                actor=actor,
                temperature=GeqLagrangeMultiplier(
                    init_value=config["temperature_init"], constraint_shape=()
                ),
            )
        )
        chunk = jnp.zeros((ex_observations.shape[0], chunk_dim))
        latent = jnp.zeros((ex_observations.shape[0], config["latent_dim"]))
        params = modules.init(
            init_rng,
            critic=(ex_observations, chunk),
            target_critic=(ex_observations, chunk),
            noise_critic=(ex_observations, latent),
            actor=(ex_observations,),
            temperature=(),
        )["params"]
        params["modules_target_critic"] = params["modules_critic"]
        tx = {
            k: optax.adam(config["lr"])
            for k in ("critic", "noise_critic", "actor", "temperature")
        }
        return cls(
            rng=rng,
            network=TrainState.create(modules, params, tx=tx),
            config=flax.core.FrozenDict(config),
            base_policy=base_policy,
        )


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="dsrl_na",
            actor_update_period=1,
            lr=0.0003,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            actor_layer_norm=True,
            value_network_class="MLP",
            value_network_kwargs=dict(
                hidden_dims=(512, 512, 512, 512), layer_norm=True
            ),
            discount=0.99,
            soft_target_update_rate=0.005,
            critic_ensemble_size=10,
            critic_target_rho=0.5,
            critic_target_min=ml_collections.config_dict.placeholder(float),
            critic_target_max=ml_collections.config_dict.placeholder(float),
            noise_scale=1.5,
            target_entropy=0.0,
            temperature_init=1.0,
            action_clip=1.0,
            gripper_action_clip=1.0,
            horizon_length=ml_collections.config_dict.placeholder(int),
        )
    )
