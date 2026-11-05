"""DICE-RL (Sun & Song, ICML 2026) over cached samples of a frozen VLA.

Pixel-observation port of ``agents/dice_rl.py`` in the behavior-constraint
repository: a deterministic residual actor on top of a frozen base policy, trained
with a Q-normalised objective plus ``bc_loss_weight`` times the mean squared
residual; min-of-ensemble critic targets; best-of-N exploration by min-Q; the actor
and the target critic move every ``actor_update_period``-th gradient step; Adam with
coupled weight decay, gradient clipping and cosine warm restarts.

Adaptation to the VLA setting: DICE conditions the residual on the flow noise ``z``
and re-runs the flow in every loss.  pi0.5's noise is not stored, so the residual is
conditioned on the cached base chunk itself, ``r = MLP(enc(s), a_base)``, and every
"fresh z" of the reference (TD target, actor loss, rollout candidates) is another of
the 16 chunks the outer policy cached with the transition.  pi0.5 is never queried in
``update``.
"""

import copy
from functools import partial
import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from nicerl.agents.cached_behavior_regularized import (
    CachedBehaviorRegularizedAgent,
    build_actor_critic,
)
from nicerl.agents.sac import get_config as get_sac_config
from nicerl.utils.networks import MLP


def cosine_warm_restarts(peak_lr, min_lr, first_cycle_steps, warmup_steps):
    """DICE-RL's CosineAnnealingWarmupRestarts (cycle_mult=1, gamma=1) as an optax schedule."""

    def schedule(count):
        step_in_cycle = count % first_cycle_steps
        warmup = min_lr + (peak_lr - min_lr) * step_in_cycle / warmup_steps
        progress = (step_in_cycle - warmup_steps) / (first_cycle_steps - warmup_steps)
        cosine = min_lr + (peak_lr - min_lr) * (1 + jnp.cos(jnp.pi * progress)) / 2
        return jnp.where(step_in_cycle < warmup_steps, warmup, cosine)

    return schedule


def make_optimizer(config):
    """Global-norm clip -> coupled L2 (torch ``Adam(weight_decay=)``) -> Adam -> cosine warm restarts."""
    return optax.chain(
        optax.clip_by_global_norm(config["max_grad_norm"]),
        optax.add_decayed_weights(config["weight_decay"]),
        optax.scale_by_adam(),
        optax.scale_by_schedule(
            cosine_warm_restarts(
                config["lr"],
                config["lr_min"],
                config["lr_first_cycle_steps"],
                config["lr_warmup_steps"],
            )
        ),
        optax.scale(-1.0),
    )


class DICERLAgent(CachedBehaviorRegularizedAgent):
    """Deterministic residual actor over cached base chunks with min-Q targets."""

    def _residual_actions(self, observations, base_actions, params=None):
        """``(base + residual, residual)`` for base chunks (B, A) or (B, K, A); unclipped
        inside the losses as in DICE, clipped at execution."""
        residual = self.network.select("actor")(
            observations, base_actions, params=params
        )
        return (base_actions + residual, residual)

    def critic_loss(self, batch, grad_params, rng):
        next_base_actions = self._subsample_base_actions(
            batch["next_base_actions"], rng, n=self.config["num_next_base_samples"]
        )
        next_actions, _ = self._residual_actions(
            batch["next_observations"], next_base_actions
        )
        next_qs = self.network.select("target_critic")(
            batch["next_observations"], next_actions
        )
        next_q = self._aggregate_target_q(next_qs).mean(axis=1)
        target_q = self._td_target(batch, next_q)
        qs = self.network.select("critic")(
            batch["observations"], batch["actions"], params=grad_params
        )
        per_member_loss = jnp.square(qs - target_q[None]).mean(axis=1)
        critic_loss = per_member_loss.sum()
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
        base_actions = self._subsample_base_actions(
            batch["base_actions"], rng, n=self.config["num_actor_base_samples"]
        )
        actions, residual = self._residual_actions(
            batch["observations"], base_actions, params=grad_params
        )
        q = self.network.select("critic")(batch["observations"], actions).min(axis=0)
        q_base = self.network.select("critic")(batch["observations"], base_actions).min(
            axis=0
        )
        q_abs_mean = jax.lax.stop_gradient(jnp.abs(q).mean())
        q_loss_scale = jnp.asarray(1.0)
        q_loss_scale = jnp.where(
            q_abs_mean > 1e-08, 1.0 / jnp.maximum(q_abs_mean, 1e-08), 1.0
        )
        q_loss = -q.mean() * q_loss_scale
        bc_mse_loss = jnp.square(residual).mean()
        total_loss = q_loss + self.config["bc_loss_weight"] * bc_mse_loss
        advantage = q - q_base
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
                "actor/q": q.mean(),
                "actor/base_q": q_base.mean(),
                "actor/q_advantage": advantage.mean(),
                "actor/better_than_base_frac": (advantage > 0).mean(),
                "actor/residual_norm": jnp.sqrt(bc_mse_loss),
                "actor/base_action_magnitude": jnp.abs(base_actions).mean(),
                "actor/action_magnitude": jnp.abs(actions).mean(),
                "actor/n_base_samples": jnp.asarray(self.config["n_base_samples"]),
            },
        )

    def target_update(self, network, module_name):
        """Polyak only on the steps where the actor optimizer is unmasked (DICE)."""
        do_update = self.network.step % self.config["actor_update_period"] == 0
        tau = jnp.where(do_update, self.config["tau"], 0.0)
        network.params[f"modules_target_{module_name}"] = jax.tree_util.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            network.params[f"modules_{module_name}"],
            network.params[f"modules_target_{module_name}"],
        )

    @partial(jax.jit, static_argnames=("deterministic",))
    def sample_actions(self, observations, base_actions, rng=None, deterministic=False):
        """Best of the N given base chunks plus residual by online-critic min-Q
        (DICE's ``max_q_min``); the residual is deterministic, so ``rng`` and
        ``deterministic`` are unused."""
        del rng, deterministic
        if base_actions.shape[1] != self.config["n_base_samples"]:
            raise ValueError(
                f"sample_actions expects {self.config['n_base_samples']} base chunks, got {base_actions.shape[1]}"
            )
        actions, _ = self._residual_actions(observations, base_actions)
        actions = jnp.clip(actions, -1.0, 1.0)
        q_min = self.network.select("critic")(observations, actions).min(axis=0)
        best = jnp.argmax(q_min, axis=1)
        best = jnp.where(
            self.network.step <= self.config["replay_flow_warmup_steps"], 0, best
        )
        return jnp.take_along_axis(actions, best[:, None, None], axis=1)[:, 0]

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        config = copy.deepcopy(config)
        action_dim = ex_actions.shape[-1]
        for key in (
            "n_base_samples",
            "num_next_base_samples",
            "num_actor_base_samples",
        ):
            if config[key] < 1:
                raise ValueError(f"{key} must be positive")
        if config["critic_target_aggregation"] not in ("mean_std", "min"):
            raise ValueError("critic_target_aggregation must be 'mean_std' or 'min'")
        if config["actor_update_period"] < 1:
            raise ValueError("actor_update_period must be >= 1")
        if config["action_chunking"] or config["chunk_size"] != 1:
            raise ValueError("DICE-RL stores each flattened chunk as one transition")
        expected_action_dim = config["action_horizon"] * config["raw_action_dim"]
        if action_dim != expected_action_dim:
            raise ValueError(
                f"flattened action dim {action_dim} != action_horizon * raw_action_dim ({config['action_horizon']} * {config['raw_action_dim']})"
            )
        if config["target_entropy"] is None:
            config["target_entropy"] = -config["target_entropy_multiplier"] * action_dim
        config["action_dim"] = action_dim
        actor_head = MLP(
            (*config["actor_hidden_dims"], action_dim),
            layer_norm=config["actor_layer_norm"],
        )
        period = int(config["actor_update_period"])
        tx = optax.multi_transform(
            {
                "actor": optax.transforms.conditionally_mask(
                    make_optimizer(config), lambda count: (count + 1) % period == 0
                ),
                "critic": make_optimizer(config),
                "frozen": optax.set_to_zero(),
            },
            lambda params: {
                key: {"modules_actor": "actor", "modules_critic": "critic"}.get(
                    key, "frozen"
                )
                for key in params
            },
        )
        network = build_actor_critic(
            config,
            ex_observations,
            ex_actions,
            actor_head=actor_head,
            ex_actor_actions=jnp.zeros_like(ex_actions),
            tx=tx,
            init_rng=init_rng,
        )
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """SAC defaults plus DICE-RL's recipe (behavior-constraint's dice_rl.py names)."""
    config = get_sac_config()
    config.update(
        ml_collections.ConfigDict(
            dict(
                agent_name="dice_rl",
                lr=0.0001,
                weight_decay=1e-05,
                max_grad_norm=1.0,
                lr_first_cycle_steps=1000,
                lr_warmup_steps=10,
                lr_min=1e-06,
                actor_hidden_dims=(1024, 1024, 1024),
                value_hidden_dims=(1024, 1024, 1024),
                actor_layer_norm=True,
                value_layer_norm=True,
                tau=0.01,
                critic_target_aggregation="min",
                bc_loss_weight=100.0,
                n_base_samples=8,
                num_next_base_samples=4,
                num_actor_base_samples=8,
                actor_update_period=2,
                replay_flow_warmup_steps=40000,
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
