"""GSBR with the paper's scalar RMS-distance support radius for state tasks."""

from typing import ClassVar
import jax.numpy as jnp
from agents.behavior_regularized import (
    BehaviorRegularizedAgent,
    get_config as behavior_regularized_config,
)


class GSBRAgent(BehaviorRegularizedAgent):
    reference_count: ClassVar[int] = 8
    reference_passthrough: ClassVar[bool] = True

    def _reference_bc_loss(self, actions, base_actions_n):
        pairwise_mse = jnp.square(
            base_actions_n[:, None] - base_actions_n[None, :]
        ).mean(axis=-1)
        diagonal = jnp.eye(self.reference_count, dtype=bool)[..., None]
        neighbor_mse = jnp.where(diagonal, jnp.inf, pairwise_mse).min(axis=1)
        radius = jnp.quantile(
            jnp.sqrt(neighbor_mse + 1e-08), self.config["tube_quantile"], axis=0
        )
        candidate_mse = jnp.square(actions[None] - base_actions_n).mean(axis=-1)
        nearest = candidate_mse.argmin(axis=0, keepdims=True)
        nearest_mse = jnp.take_along_axis(candidate_mse, nearest, axis=0)[0]
        distance = jnp.sqrt(nearest_mse + 1e-08)
        loss = jnp.square(jnp.maximum(distance - radius, 0.0)).mean()
        return (
            loss,
            {
                "actor/tube_radius": radius.mean(),
                "actor/tube_active_frac": (distance > radius).mean(),
            },
        )


def get_config():
    config = behavior_regularized_config()
    config.agent_name = "gsbr"
    config.tube_quantile = 0.9
    return config
