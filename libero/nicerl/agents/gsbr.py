"""GSBR's final LIBERO loss: a union of tubes with seven RMS-distance radii."""

from typing import ClassVar
import jax.numpy as jnp
from nicerl.agents.cached_behavior_regularized import (
    CachedBehaviorRegularizedAgent,
    get_config as behavior_regularized_config,
)


class GSBRAgent(CachedBehaviorRegularizedAgent):
    reference_count: ClassVar[int] = 8
    reference_passthrough: ClassVar[bool] = True

    def _reference_bc_loss(self, actions, base_actions):
        # Match neighbors by whole chunk; calibrate one radius per physical dimension.
        horizon, dimensions = (
            self.config["action_horizon"],
            self.config["raw_action_dim"],
        )
        chunks = base_actions.reshape(*base_actions.shape[:2], horizon, dimensions)
        pairwise_mse = jnp.square(
            base_actions[:, :, None] - base_actions[:, None, :]
        ).mean(axis=-1)
        diagonal = jnp.eye(self.reference_count, dtype=bool)[None]
        neighbors = jnp.where(diagonal, jnp.inf, pairwise_mse).argmin(axis=-1)
        neighbor_chunks = jnp.take_along_axis(
            chunks, neighbors[..., None, None], axis=1
        )
        neighbor_dim_mse = jnp.square(chunks - neighbor_chunks).mean(axis=-2)
        radius = jnp.quantile(
            jnp.sqrt(neighbor_dim_mse + 1e-8), self.config["tube_quantile"], axis=1
        )
        actor_chunks = actions.reshape(actions.shape[0], horizon, dimensions)
        distance = jnp.sqrt(
            jnp.square(actor_chunks[:, None] - chunks).mean(axis=-2) + 1e-8
        )
        violations = jnp.square(jnp.maximum(distance - radius[:, None], 0.0))
        # One coherent reference is chosen AFTER aggregating dimension penalties.
        candidate_losses = violations.mean(axis=-1)
        chosen = candidate_losses.argmin(axis=1, keepdims=True)
        losses = jnp.take_along_axis(candidate_losses, chosen, axis=1)[:, 0]
        info = {"tube_radius": radius.mean(), "tube_active_frac": (losses > 0).mean()}
        for d in range(dimensions):
            info[f"tube_radius_dim_{d}"] = radius[:, d].mean()
        return losses.mean(), info


def get_config():
    config = behavior_regularized_config()
    config.agent_name = "gsbr"
    config.tube_quantile = 0.9
    config.base_action_dropout = 0.0
    return config
