"""Vanilla RL Token with one reference and passthrough."""

from typing import ClassVar
import jax.numpy as jnp
from nicerl.agents.cached_behavior_regularized import (
    CachedBehaviorRegularizedAgent,
    get_config as behavior_regularized_config,
)


class RLTAgent(CachedBehaviorRegularizedAgent):
    reference_count: ClassVar[int] = 1
    reference_passthrough: ClassVar[bool] = True

    def _reference_bc_loss(self, actions, base_actions):
        loss = jnp.square(actions - base_actions[:, 0]).mean()
        return loss, {"reference_mse": loss}


def get_config():
    config = behavior_regularized_config()
    config.agent_name = "rlt"
    return config
