"""DICE-RL (Sun & Song, ICML 2026): residual RL finetuning of a frozen flow policy.

Port of ``dice-rl/dice_jax/dice.py`` (the state-based ``ft_distill_residual_flow_mlp``
configs) onto this repository's agent interface, using the repository's ``BCAgent``
flow as the frozen base policy:

* action = stop_grad(base_flow(s, z)) + residual(s, z); the same noise ``z`` drives the
  flow sample and the residual MLP (the base flow's own Euler steps and action clipping
  apply, DICE's flow is unclipped and uses 10 steps).
* critic: ensemble of ``critic_ensemble_size`` Q(s, action chunk).  TD target from
  ``num_next_noise_samples`` next actions (online actor on fresh noise), aggregated
  over the ensemble (``critic_target_aggregation``, min in DICE) then averaged over
  the noise samples, discounted per row by ``discount ** n_steps`` (n-step chunk
  windows come from ``Dataset.sample_sequence`` via ``--n_step``), optionally clipped
  to ``[critic_target_min, critic_target_max]``.  Per-member MSE, summed over members
  (``critic_loss_reduction="sum"``, DICE) or averaged.
* actor: ``num_multi_z_for_actor_loss`` noises per state.  Q term = -mean over the
  batch of the mean min-Q over noises, zeroed on offline rows when
  ``disable_q_loss_for_expert_data`` (``batch["is_offline"]`` from main.py) and divided
  by the detached mean |Q| when ``use_q_normalization``; plus ``bc_loss_weight`` times
  the mean squared distance between the total action and the frozen-flow action.
* rollouts: best of ``num_exploration_samples`` candidates by min-Q (``max_q_min``),
  a single random candidate during the first ``replay_flow_warmup_steps`` gradient
  steps; evaluation uses ``eval_strategy`` (``rollout`` calls
  ``sample_actions_with_diagnostics``).  Executed actions are clipped to the env bounds.
* delayed actor/target updates: main.py gates ``update_actor`` with
  ``actor_update_period`` and ``--actor_update_block``; the target critic follows the
  actor cadence unless ``target_update_every_step``.  A skipped actor step leaves the
  actor and its optimizer untouched (``TrainState.apply_gradients``).
* optimizer: ``make_optimizer`` = grad clip -> coupled L2 -> Adam -> cosine warm
  restarts (torch ``Adam(weight_decay=)`` + ``CosineAnnealingWarmupRestarts``).
"""

import copy
from functools import partial
from typing import Any
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
from utils.flax_utils import (
    ModuleDict,
    TrainState,
    make_optimizer,
    module_mask,
    nonpytree_field,
)
from utils.networks import MLP, Value

ACTIVATIONS = {
    "gelu": partial(nn.gelu, approximate=False),
    "gelu_tanh": nn.gelu,
    "relu": nn.relu,
}


class DICERLAgent(flax.struct.PyTreeNode):
    """DICE-RL: a residual actor on a frozen flow base policy (see module docstring)."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()
    base_policy: Any

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
            batch["is_offline"] = jnp.zeros(batch["rewards"].shape[0])
        return batch

    def _discount(self, batch):
        """Per-row ``discount ** n_steps`` when the batch carries ``n_steps``, else
        the fixed chunk discount (see ``SACAgent._discount``)."""
        if "n_steps" in batch:
            return self.config["discount"] ** batch["n_steps"]
        return self.config["discount"] ** self.config["horizon_length"]

    def _critic(self, observations, actions, params=None, critic_name="critic"):
        qs = self.network.select(critic_name)(observations, actions, params=params)
        if qs.ndim == 1:
            qs = qs[None, :]
        return qs

    def _aggregate_target(self, qs):
        """Reduce an (E, ...) ensemble of target Qs as ``critic_target_aggregation``."""
        return qs.min(axis=0)
        return qs.mean(axis=0)

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

    def _tile(self, x, n):
        """(B, ...) -> (n * B, ...), sample-major: row k * B + b is sample k of b."""
        return jnp.repeat(x[None], n, axis=0).reshape((n * x.shape[0],) + x.shape[1:])

    def _get_action(self, observations, noise, params=None):
        """Frozen-flow action from ``noise`` plus the residual, both (M, H*A).

        Returns ``(actions, base_actions)``; the base action carries no gradient.
        """
        base_actions = self.base_policy.sample_actions(observations, init_noise=noise)
        if base_actions.ndim < noise.ndim:
            base_actions = base_actions[None]
        base_actions = jax.lax.stop_gradient(base_actions)
        residual = self.network.select("actor")(
            jnp.concatenate([observations, noise], axis=-1), params=params
        )
        return (base_actions + residual, base_actions)

    def critic_loss(self, batch, grad_params, rng):
        """TD loss with ``num_next_noise_samples`` next actions per state."""
        batch_size = batch["rewards"].shape[0]
        n_next = self.config["num_next_noise_samples"]
        next_observations = self._tile(batch["next_observations"], n_next)
        next_noise = jax.random.normal(
            rng, (n_next * batch_size, self.config["chunk_action_dim"])
        )
        next_actions, _ = self._get_action(next_observations, next_noise)
        target_next_qs = self._critic(
            next_observations, next_actions, critic_name="target_critic"
        )
        target_next_q = (
            self._aggregate_target(target_next_qs)
            .reshape(n_next, batch_size)
            .mean(axis=0)
        )
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
        target_q = jax.lax.stop_gradient(target_q)
        predicted_qs = self._critic(
            batch["observations"], batch["actions"], params=grad_params
        )
        per_member_loss = jnp.square(predicted_qs - target_q[None]).mean(axis=1)
        critic_loss = per_member_loss.sum()
        return (
            critic_loss,
            {
                "critic/critic_loss": critic_loss,
                "critic/predicted_qs": predicted_qs.mean(),
                "critic/target_qs": target_q.mean(),
                "critic/rewards": batch["rewards"].mean(),
            },
        )

    def actor_loss(self, batch, grad_params, rng):
        """DICE actor loss: normalized Q term (online rows) + multi-z distillation."""
        batch_size = batch["observations"].shape[0]
        n_z = self.config["num_multi_z_for_actor_loss"]
        observations = self._tile(batch["observations"], n_z)
        noise = jax.random.normal(
            rng, (n_z * batch_size, self.config["chunk_action_dim"])
        )
        actions, base_actions = self._get_action(
            observations, noise, params=grad_params
        )
        q_current = (
            self._critic(observations, actions).min(axis=0).reshape(n_z, batch_size)
        )
        q_base = jax.lax.stop_gradient(
            self._critic(observations, base_actions).min(axis=0)
        ).reshape(n_z, batch_size)
        q_weight = jnp.ones(batch_size)
        q_loss = -(q_current.mean(axis=0) * q_weight).mean()
        weighted_count = q_weight.sum() * n_z
        q_abs_mean = jnp.where(
            weighted_count > 0,
            (jnp.abs(q_current) * q_weight[None]).sum()
            / jnp.maximum(weighted_count, 1.0),
            jnp.abs(q_current).mean(),
        )
        q_abs_mean = jax.lax.stop_gradient(q_abs_mean)
        q_loss_scale = jnp.asarray(1.0)
        if self.config["use_q_normalization"]:
            q_loss_scale = jnp.where(
                q_abs_mean > 1e-08, 1.0 / jnp.maximum(q_abs_mean, 1e-08), 1.0
            )
        q_loss = q_loss * q_loss_scale
        squared_residual = jnp.square(actions - base_actions)
        bc_mse_loss = squared_residual.mean()
        total_loss = q_loss + self.config["bc_loss_weight"] * bc_mse_loss
        advantage = q_current - q_base
        return (
            total_loss,
            {
                "actor/total_loss": total_loss,
                "actor/actor_loss": q_loss,
                "actor/q_loss_scale": q_loss_scale,
                "actor/q_abs_mean": q_abs_mean,
                "actor/bc_mse_loss": bc_mse_loss,
                "actor/bc_loss": self.config["bc_loss_weight"] * bc_mse_loss,
                "actor/bc_loss_weight": jnp.asarray(self.config["bc_loss_weight"]),
                "actor/q": q_current.mean(),
                "actor/base_q": q_base.mean(),
                "actor/q_advantage": advantage.mean(),
                "actor/better_than_base_frac": (advantage > 0).mean(),
                "actor/residual_norm": jnp.sqrt(squared_residual.mean()),
                "actor/base_action_magnitude": jnp.abs(base_actions).mean(),
                "actor/action_magnitude": jnp.abs(actions).mean(),
                "actor/online_frac": (1.0 - batch["is_offline"]).mean(),
            },
        )

    def loss_fns(self, batch):
        return {
            "critic": partial(self.critic_loss, batch),
            "actor": partial(self.actor_loss, batch),
        }

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

    @partial(jax.jit, static_argnames=["update_critic", "update_actor"])
    def total_loss(
        self,
        batch,
        grad_params,
        rng=None,
        update_critic=True,
        update_actor=True,
    ):
        """Combined critic and actor loss (validation batches)."""

        info = {}
        rng = rng if rng is not None else self.rng
        rng, critic_rng, actor_rng = jax.random.split(rng, 3)
        batch = self._normalize_batch(batch)
        critic_loss = 0.0
        if update_critic:
            critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
            info.update(critic_info)
        actor_loss = 0.0
        if update_actor:
            actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
            info.update(actor_info)
        return (critic_loss + actor_loss, info)

    @partial(jax.jit, static_argnames=["update_critic", "update_actor"])
    def update(self, batch, update_critic=True, update_actor=True):
        """One gradient step; the target critic follows the actor cadence."""

        new_rng, rng = jax.random.split(self.rng)
        batch = self._normalize_batch(batch)
        loss_fns = self.loss_fns(batch)
        if not update_critic:
            loss_fns.pop("critic")
        if not update_actor:
            loss_fns.pop("actor")
        new_network, info_tree = self.network.apply_loss_fns(
            loss_fns, rng=rng, has_aux=True
        )
        info = {}
        for loss_info in info_tree.values():
            info.update(loss_info)
        if update_critic and (update_actor or False):
            self.target_update(new_network, "critic")
        return (self.replace(network=new_network, rng=new_rng), info)

    def _select_actions(self, observations, rng, strategy):
        """Best of ``num_exploration_samples`` candidates by min-Q (``max_q_min``), or
        one random candidate (``standard`` and during the exploration warmup).

        Returns the chosen actions (B, H*A), their base actions (B, H*A) and their
        ensemble Q values (E, B).
        """
        batch_size = observations.shape[0]
        n = self.config["num_exploration_samples"] if strategy == "max_q_min" else 1
        obs_tiled = self._tile(observations, n)
        noise = jax.random.normal(
            rng, (n * batch_size, self.config["chunk_action_dim"])
        )
        actions, base_actions = self._get_action(obs_tiled, noise)
        qs = self._critic(obs_tiled, actions).reshape(-1, n, batch_size)
        selected = jnp.argmax(qs.min(axis=0), axis=0)
        selected = jnp.where(
            self.network.step <= self.config["replay_flow_warmup_steps"], 0, selected
        )
        rows = jnp.arange(batch_size)
        return (
            actions.reshape(n, batch_size, -1)[selected, rows],
            base_actions.reshape(n, batch_size, -1)[selected, rows],
            qs[:, selected, rows],
        )

    @partial(jax.jit, static_argnames=["argmax"])
    def sample_actions(self, observations, seed=None, temperature=1.0, argmax=False):
        """Rollout actions (``explore_strategy``), clipped to the env bounds."""
        del temperature, argmax
        observations = jnp.asarray(observations)
        unbatched = observations.ndim == 1
        if unbatched:
            observations = observations[None]
        actions, _, _ = self._select_actions(observations, seed, "max_q_min")
        actions = self._clip_actions(actions)
        return actions[0] if unbatched else actions

    @partial(jax.jit, static_argnames=["argmax"])
    def sample_actions_with_diagnostics(
        self, observations, seed=None, temperature=1.0, argmax=False
    ):
        """Evaluation actions (``eval_strategy``) with per-state diagnostics."""
        del temperature, argmax
        observations = jnp.asarray(observations)
        unbatched = observations.ndim == 1
        if unbatched:
            observations = observations[None]
        actions, base_actions, qs = self._select_actions(
            observations, seed, "max_q_min"
        )
        actions = self._clip_actions(actions)
        diagnostics = {
            "q": qs.mean(axis=0),
            "q_min": qs.min(axis=0),
            "q_std": qs.std(axis=0),
            "residual_norm": jnp.sqrt(
                jnp.square(actions - base_actions).mean(axis=-1) + 1e-08
            ),
        }
        if unbatched:
            actions = actions[0]
            diagnostics = jax.tree_util.tree_map(lambda x: x[0], diagnostics)
        return (actions, diagnostics)

    @jax.jit
    def sample_base_actions_only(self, observations, seed=None):
        """One chunk from the frozen base policy (prefill and base-policy eval)."""
        return self.base_policy.sample_actions(observations, seed=seed)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, base_policy):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)
        config = copy.deepcopy(config)
        assert config["activation"] in ACTIVATIONS
        assert config["actor_update_period"] >= 1
        if (
            config["critic_target_min"] is not None
            and config["critic_target_max"] is not None
        ):
            assert config["critic_target_min"] < config["critic_target_max"]
        raw_action_dim = ex_actions.shape[-1]
        horizon_length = config["horizon_length"]
        chunk_action_dim = horizon_length * raw_action_dim
        config["raw_action_dim"] = raw_action_dim
        config["chunk_action_dim"] = chunk_action_dim
        config["action_dim"] = raw_action_dim
        activation = ACTIVATIONS[config["activation"]]
        critic_def = Value(
            network_class="MLP",
            network_kwargs=dict(
                hidden_dims=config["critic_hidden_dims"],
                layer_norm=config["layer_norm"],
                activation=activation,
            ),
            num_ensembles=config["critic_ensemble_size"],
        )
        actor_def = MLP(
            (*config["actor_hidden_dims"], chunk_action_dim),
            activation=activation,
            layer_norm=config["layer_norm"],
        )
        ex_chunk_actions = jnp.zeros((ex_observations.shape[0], chunk_action_dim))
        ex_actor_inputs = jnp.concatenate([ex_observations, ex_chunk_actions], axis=-1)
        network_info = dict(
            critic=(critic_def, (ex_observations, ex_chunk_actions)),
            target_critic=(
                copy.deepcopy(critic_def),
                (ex_observations, ex_chunk_actions),
            ),
            actor=(actor_def, (ex_actor_inputs,)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}
        optimizer_kwargs = dict(
            max_grad_norm=config["max_grad_norm"],
            weight_decay=config["weight_decay"],
            lr_schedule=config["lr_schedule"],
            lr_first_cycle_steps=config["lr_first_cycle_steps"],
            lr_warmup_steps=config["lr_warmup_steps"],
            lr_min=config["lr_min"],
            lr_cycle_gamma=config["lr_cycle_gamma"],
        )
        network_def = ModuleDict(networks)
        network_tx = {
            "critic": make_optimizer(
                config["critic_lr"],
                weight_decay_mask=module_mask("critic"),
                **optimizer_kwargs,
            ),
            "actor": make_optimizer(
                config["actor_lr"],
                weight_decay_mask=module_mask("actor"),
                **optimizer_kwargs,
            ),
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


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="dice_rl",
            batch_size=256,
            discount=0.99,
            bc_loss_weight=100.0,
            use_q_normalization=True,
            num_multi_z_for_actor_loss=16,
            critic_ensemble_size=10,
            critic_target_rho=0.5,
            critic_target_min=ml_collections.config_dict.placeholder(float),
            critic_target_max=ml_collections.config_dict.placeholder(float),
            num_next_noise_samples=4,
            soft_target_update_rate=0.01,
            actor_update_period=2,
            num_exploration_samples=16,
            replay_flow_warmup_steps=40000,
            actor_lr=0.0001,
            critic_lr=0.0001,
            weight_decay=1e-05,
            max_grad_norm=1.0,
            lr_schedule="cosine_warm_restarts",
            lr_first_cycle_steps=1000,
            lr_warmup_steps=10,
            lr_min=1e-06,
            lr_cycle_gamma=1.0,
            actor_hidden_dims=(1024, 1024, 1024),
            critic_hidden_dims=(1024, 1024, 1024),
            activation="gelu",
            layer_norm=True,
            dataset_action_clip_eps=1e-05,
            action_clip=1.0,
            gripper_action_clip=1.0,
            horizon_length=ml_collections.config_dict.placeholder(int),
            raw_action_dim=ml_collections.config_dict.placeholder(int),
            chunk_action_dim=ml_collections.config_dict.placeholder(int),
            action_dim=ml_collections.config_dict.placeholder(int),
        )
    )
    return config
