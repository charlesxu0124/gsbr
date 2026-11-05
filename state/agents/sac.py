"""Continuous soft actor-critic, independently trainable without a base policy."""

import copy
from functools import partial
from typing import Any
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, GeqLagrangeMultiplier, Value


class SACAgent(flax.struct.PyTreeNode):
    """Tanh Gaussian policy, clipped double-Q backup, and learned entropy weight.

    Actions use the benchmarks' symmetric normalized bounds. By default this
    is ordinary single-step, twin-critic SAC; chunked batches are also accepted.
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def _temperature(self, grad_params=None):
        return self.network.select("temperature")(params=grad_params)

    def _discount(self, batch):
        """Bootstrap discount per row: ``discount ** n_steps`` (env steps covered by
        the sampled window, see ``Dataset.sample_sequence``) when the batch carries
        ``n_steps``, else the fixed chunk discount ``discount ** horizon_length``."""
        if "n_steps" in batch:
            return self.config["discount"] ** batch["n_steps"]
        return self.config["discount"] ** self.config.get("horizon_length", 1)

    def _normalize_batch(self, batch):
        """Convert sequence/action-chunk batches into flat-chunk transitions."""
        batch = dict(batch)
        if batch["actions"].ndim == batch["observations"].ndim + 1:
            b = batch["actions"].shape[0]
            batch["actions"] = batch["actions"].reshape(b, -1)
            batch["rewards"] = batch["rewards"][..., -1]
            if batch["masks"].ndim > 1:
                batch["masks"] = batch["masks"][..., -1]
            if "terminals" in batch and batch["terminals"].ndim > 1:
                batch["terminals"] = batch["terminals"][..., -1]
            if batch["next_observations"].ndim == batch["observations"].ndim + 1:
                batch["next_observations"] = batch["next_observations"][..., -1, :]
        if "is_offline" not in batch:
            batch["is_offline"] = jnp.zeros(batch["actions"].shape[0])
        return batch

    def target_update(self, network, module_name):
        """Polyak update a target module."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: (
                p * self.config["soft_target_update_rate"]
                + tp * (1 - self.config["soft_target_update_rate"])
            ),
            network.params[f"modules_{module_name}"],
            network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    def _action_bounds(self):
        bounds = jnp.full((self.config["action_dim"],), self.config["action_clip"])
        if self.config["gripper_action_clip"] is not None:
            bounds = bounds.at[-1].set(self.config["gripper_action_clip"])
        return jnp.tile(bounds, self.config["horizon_length"])

    def _sample_actions_and_log_probs(self, observations, rng, params=None):
        dist = self.network.select("actor")(observations, params=params)
        actions, log_probs = dist.sample_and_log_prob(seed=rng)
        bounds = self._action_bounds()
        return actions * bounds, log_probs - jnp.log(bounds).sum()

    def _critic(self, observations, actions, params=None, target=False):
        return self.network.select("target_critic" if target else "critic")(
            observations, actions, params=params
        )

    def _td_target(self, batch, rng):
        actions, log_probs = self._sample_actions_and_log_probs(
            batch["next_observations"], rng
        )
        next_q = self._critic(batch["next_observations"], actions, target=True).min(
            axis=0
        )
        target = batch["rewards"] + self._discount(batch) * batch["masks"] * (
            next_q - self._temperature() * log_probs
        )
        return jax.lax.stop_gradient(target)

    def critic_loss(self, batch, grad_params, rng):
        target = self._td_target(batch, rng)
        qs = self._critic(batch["observations"], batch["actions"], params=grad_params)
        loss = jnp.square(qs - target[None]).mean()
        return loss, {
            "critic/loss": loss,
            "critic/q": qs.mean(),
            "critic/target_q": target.mean(),
        }

    def actor_loss(self, batch, grad_params, rng):
        actions, log_probs = self._sample_actions_and_log_probs(
            batch["observations"], rng, grad_params
        )
        qs = self._critic(batch["observations"], actions).min(axis=0)
        loss = (self._temperature() * log_probs - qs).mean()
        return loss, {
            "actor/loss": loss,
            "actor/entropy": -log_probs.mean(),
            "actor/q": qs.mean(),
        }

    def temperature_loss(self, batch, grad_params, rng):
        _, log_probs = self._sample_actions_and_log_probs(batch["observations"], rng)
        entropy = jax.lax.stop_gradient(-log_probs.mean())
        loss = self.network.select("temperature")(
            lhs=entropy, rhs=self.config["target_entropy"], params=grad_params
        )
        return loss, {
            "temperature/loss": loss,
            "temperature/value": self._temperature(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        batch = self._normalize_batch(batch)
        keys = jax.random.split(self.rng if rng is None else rng, 3)
        loss, info = 0.0, {}
        for fn, key in zip(
            (self.critic_loss, self.actor_loss, self.temperature_loss), keys
        ):
            value, metrics = fn(batch, grad_params, key)
            loss += value
            info.update(metrics)
        return loss, info

    @partial(jax.jit, static_argnames=["update_critic", "update_actor"])
    def update(self, batch, update_critic=True, update_actor=True):
        batch = self._normalize_batch(batch)
        rng, loss_rng = jax.random.split(self.rng)
        losses = {}
        if update_critic:
            losses["critic"] = partial(self.critic_loss, batch)
        if update_actor:
            losses["actor"] = partial(self.actor_loss, batch)
            losses["temperature"] = partial(self.temperature_loss, batch)
        if not losses:
            return self.replace(rng=rng), {}
        network, metrics = self.network.apply_loss_fns(
            losses, rng=loss_rng, has_aux=True
        )
        if update_critic:
            self.target_update(network, "critic")
        info = {
            key: value for group in metrics.values() for key, value in group.items()
        }
        return self.replace(network=network, rng=rng), info

    @partial(jax.jit, static_argnames=["argmax"])
    def sample_actions(self, observations, seed=None, temperature=1.0, argmax=False):
        observations = jnp.asarray(observations)
        # Keep the distribution nonsingular during deterministic evaluation.
        dist = self.network.select("actor")(
            observations, temperature=jnp.maximum(temperature, 1e-6)
        )
        if argmax:
            actions = dist.mode()
        else:
            sampled = dist.sample(seed=self.rng if seed is None else seed)
            actions = jnp.where(jnp.asarray(temperature) == 0, dist.mode(), sampled)
        return actions * self._action_bounds()

    @jax.jit
    def sample_actions_with_diagnostics(self, observations, seed=None, temperature=1.0):
        actions = self.sample_actions(observations, seed=seed, temperature=temperature)
        qs = self._critic(observations, actions).min(axis=0)
        return actions, {"q": qs}

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        config = copy.deepcopy(dict(config))
        config["action_dim"] = ex_actions.shape[-1]
        chunk_dim = config["action_dim"] * config["horizon_length"]
        if config["horizon_length"] < 1 or config["critic_ensemble_size"] < 2:
            raise ValueError(
                "SAC requires a positive action horizon and at least two critics"
            )
        if config["action_clip"] <= 0 or (
            config["gripper_action_clip"] is not None
            and config["gripper_action_clip"] <= 0
        ):
            raise ValueError("SAC action bounds must be positive")
        if config["target_entropy"] is None:
            config["target_entropy"] = -float(chunk_dim)
        if config["temperature_init"] <= 0:
            raise ValueError("temperature_init must be positive")
        actor = Actor(
            config["actor_hidden_dims"],
            chunk_dim,
            layer_norm=config["actor_layer_norm"],
            tanh_squash=True,
            state_dependent_std=True,
            const_std=False,
        )
        critic = Value(
            network_class="MLP",
            network_kwargs=config["value_network_kwargs"],
            num_ensembles=config["critic_ensemble_size"],
        )
        modules = ModuleDict(
            dict(
                actor=actor,
                critic=critic,
                target_critic=copy.deepcopy(critic),
                temperature=GeqLagrangeMultiplier(
                    init_value=config["temperature_init"], constraint_shape=()
                ),
            )
        )
        rng, key = jax.random.split(jax.random.PRNGKey(seed))
        chunk_actions = jnp.zeros((ex_observations.shape[0], chunk_dim))
        params = modules.init(
            key,
            actor=(ex_observations,),
            critic=(ex_observations, chunk_actions),
            target_critic=(ex_observations, chunk_actions),
            temperature=(),
        )["params"]
        params["modules_target_critic"] = params["modules_critic"]
        network = TrainState.create(
            modules,
            params,
            tx={
                name: optax.adam(config["lr"])
                for name in ("actor", "critic", "temperature")
            },
        )
        return cls(rng=rng, network=network, config=flax.core.FrozenDict(config))


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name="sac",
            actor_update_period=1,
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(256, 256),
            actor_layer_norm=False,
            value_network_kwargs=dict(hidden_dims=(256, 256), layer_norm=False),
            critic_ensemble_size=2,
            discount=0.99,
            soft_target_update_rate=0.005,
            temperature_init=0.1,
            target_entropy=ml_collections.config_dict.placeholder(float),
            horizon_length=1,
            action_clip=1.0,
            gripper_action_clip=ml_collections.config_dict.placeholder(float),
            dataset_action_clip_eps=1e-5,
        )
    )
