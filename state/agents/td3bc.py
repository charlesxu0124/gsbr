"""Paper TD3+BC baseline: one frozen-policy target, observation-only actor."""

from typing import ClassVar
import jax.numpy as jnp
from agents.behavior_regularized import (
    BehaviorRegularizedAgent,
    get_config as behavior_regularized_config,
)


class TD3BCAgent(BehaviorRegularizedAgent):
    reference_count: ClassVar[int] = 1
    reference_passthrough: ClassVar[bool] = False

    def _reference_bc_loss(self, actions, base_actions_n):
        loss = jnp.square(actions - base_actions_n[0]).mean()
        return (loss, {"actor/reference_mse": loss})


def get_config():
    config = behavior_regularized_config()
    config.agent_name = "td3bc"
    return config
