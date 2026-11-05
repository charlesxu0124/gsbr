"""EXPO (arXiv 2507.07986) over cached samples of a frozen VLA.

Pixel-observation port of ``agents/expo.py`` in the behavior-constraint repository.
A residual (edit) actor ``e(s, a_base)`` proposes ``clip(a_base + scale * e)``; the
policy scores the N cached base chunks plus the M edited chunks with the critic
ensemble and executes the argmax of the min-Q.  The residual policy of the paper
table is EXPO with N = M = 1 (candidates = {base, edited}).

The reference resamples the base policy inside the losses; here every base sample
is drawn from the 16 chunks that the outer policy cached with the transition
(``base_actions`` / ``next_base_actions``), so pi0.5 is never queried in ``update``.
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
from nicerl.utils.networks import MLP, TanhNormal


class EXPOAgent(CachedBehaviorRegularizedAgent):
    """Residual edit actor plus best-of-(N + M) selection over cached base chunks."""

    def _edit(self, observations, base_actions, rng, params=None, deterministic=False):
        """Edit base chunks (B, A) or (B, M, A) with the residual actor.

        Returns ``(edited, residual, dist)``; the residual is tanh-squashed to (-1, 1)
        and scaled by ``residual_action_scale`` before being added.
        """
        dist = self.network.select("actor")(observations, base_actions, params=params)
        residual = dist.mode() if deterministic else dist.sample(seed=rng)
        edited = jnp.clip(
            base_actions + self.config["residual_action_scale"] * residual, -1.0, 1.0
        )
        return (edited, residual, dist)

    def _candidates(self, observations, base_actions, rng, deterministic=False):
        """(B, N, A) cached base chunks -> (B, N + M, A) candidates."""
        m = self.config["n_residual_samples"]
        edited, _, _ = self._edit(
            observations, base_actions[:, :m], rng, deterministic=deterministic
        )
        return jnp.concatenate([base_actions, edited], axis=1)

    def _best_candidate(self, observations, candidates, critic_name):
        """Argmax over candidates of the ensemble min-Q: (B, A) and the (B, K) min-Qs."""
        q_min = self.network.select(critic_name)(observations, candidates).min(axis=0)
        best = jnp.argmax(q_min, axis=1)
        best_actions = jnp.take_along_axis(candidates, best[:, None, None], axis=1)[
            :, 0
        ]
        return (best_actions, q_min)

    def critic_loss(self, batch, grad_params, rng):
        subset_rng, edit_rng = jax.random.split(rng)
        next_base_actions = self._subsample_base_actions(
            batch["next_base_actions"], subset_rng
        )
        candidates = self._candidates(
            batch["next_observations"], next_base_actions, edit_rng
        )
        next_actions, candidate_q = self._best_candidate(
            batch["next_observations"], candidates, "target_critic"
        )
        next_qs = self.network.select("target_critic")(
            batch["next_observations"], next_actions
        )
        target_q = self._td_target(batch, self._aggregate_target_q(next_qs))
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
                "critic/target_candidate_q": candidate_q.max(axis=1).mean(),
            },
        )

    def actor_loss(self, batch, grad_params, rng):
        reference_rng, edit_rng = jax.random.split(rng)
        reference = self._subsample_base_actions(
            batch["base_actions"], reference_rng, n=1
        )[:, 0]
        edited, residual, dist = self._edit(
            batch["observations"], reference, edit_rng, params=grad_params
        )
        q = self.network.select("critic")(batch["observations"], edited).mean(axis=0)
        actor_loss = -q.mean()
        scaled_residual = self.config["residual_action_scale"] * residual
        return (
            actor_loss,
            {
                "actor/total_loss": actor_loss,
                "actor/actor_loss": actor_loss,
                "actor/q": q.mean(),
                "actor/action_std": dist._distribution.stddev().mean(),
                "actor/residual_magnitude": jnp.abs(scaled_residual).mean(),
                "actor/rl_ref_deviation": jnp.sqrt(
                    jnp.square(edited - reference).mean(axis=-1) + 1e-08
                ).mean(),
                "actor/reference_action_magnitude": jnp.abs(reference).mean(),
                "actor/residual_action_scale": jnp.asarray(
                    self.config["residual_action_scale"]
                ),
                "actor/n_base_samples": jnp.asarray(self.config["n_base_samples"]),
                "actor/n_residual_samples": jnp.asarray(
                    self.config["n_residual_samples"]
                ),
            },
        )

    @partial(jax.jit, static_argnames=("deterministic",))
    def sample_actions(self, observations, base_actions, rng=None, deterministic=False):
        """Best of the N given base chunks and their M edits by target-critic min-Q."""
        if base_actions.shape[1] != self.config["n_base_samples"]:
            raise ValueError(
                f"sample_actions expects {self.config['n_base_samples']} base chunks, got {base_actions.shape[1]}"
            )
        if rng is None and (not deterministic):
            raise ValueError("rng is required for stochastic EXPO sampling")
        candidates = self._candidates(observations, base_actions, rng, deterministic)
        actions, _ = self._best_candidate(observations, candidates, "target_critic")
        return jnp.clip(actions, -1.0, 1.0)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)
        config = copy.deepcopy(config)
        action_dim = ex_actions.shape[-1]
        if config["n_base_samples"] < 1:
            raise ValueError("n_base_samples must be positive")
        if not 1 <= config["n_residual_samples"] <= config["n_base_samples"]:
            raise ValueError("n_residual_samples must be in [1, n_base_samples]")
        if config["residual_action_scale"] <= 0:
            raise ValueError("residual_action_scale must be positive")
        if config["critic_target_aggregation"] not in ("mean_std", "min"):
            raise ValueError("critic_target_aggregation must be 'mean_std' or 'min'")
        if config["action_chunking"] or config["chunk_size"] != 1:
            raise ValueError("EXPO stores each flattened chunk as one transition")
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
        network = build_actor_critic(
            config,
            ex_observations,
            ex_actions,
            actor_head=actor_head,
            ex_actor_actions=jnp.zeros_like(ex_actions),
            tx=optax.adam(learning_rate=config["lr"]),
            init_rng=init_rng,
        )
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    """SAC defaults plus the EXPO settings (behavior-constraint's expo.py names)."""
    config = get_sac_config()
    config.update(
        ml_collections.ConfigDict(
            dict(
                agent_name="expo",
                n_base_samples=8,
                n_residual_samples=8,
                residual_action_scale=0.1,
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
