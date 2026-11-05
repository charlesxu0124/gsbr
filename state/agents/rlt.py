"""Vanilla RL Token: one base-policy sample and reference passthrough."""

from typing import ClassVar
import jax.numpy as jnp
from agents.behavior_regularized import (
    BehaviorRegularizedAgent,
    get_config as behavior_regularized_config,
)


class RLTAgent(BehaviorRegularizedAgent):
    reference_count: ClassVar[int] = 1
    reference_passthrough: ClassVar[bool] = True

    def _reference_bc_loss(self, actions, base_actions_n):
        loss = jnp.square(actions - base_actions_n[0]).mean()
        return (loss, {"actor/reference_mse": loss})


def get_config():
    config = behavior_regularized_config()
    config.agent_name = "rlt"
    return config
