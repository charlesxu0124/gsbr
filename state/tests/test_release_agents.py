import importlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from agents import agents
from agents.bc import BCAgent, get_config as bc_config
from agents.rlt import RLTAgent
from agents.td3bc import TD3BCAgent
from envs.robomimic_observations import get_low_dim_obs_keys

ROOT = Path(__file__).resolve().parents[1]


def make_agent(name):
    obs, actions = jnp.ones((3, 4)), jnp.zeros((3, 2))
    bc = bc_config()
    bc.update(hidden_dims=(16, 16), horizon_length=2, action_chunking=True)
    base = BCAgent.create(0, obs, actions, bc)
    config = importlib.import_module("agents." + name).get_config()
    config.horizon_length = 2
    if "actor_hidden_dims" in config:
        config.actor_hidden_dims = (16, 16)
    if "value_network_kwargs" in config:
        config.value_network_kwargs.hidden_dims = (16, 16)
    else:
        config.critic_hidden_dims = (16, 16)
    return agents[name].create(0, obs, actions, config, base_policy=base)


@pytest.mark.parametrize(
    "name", ["rlt", "gsbr", "td3bc", "expo", "bfn", "dsrl_na", "dice_rl"]
)
def test_update_is_finite_and_base_stays_frozen(name):
    agent = make_agent(name)
    obs = jnp.ones((3, 4))
    batch = dict(
        observations=obs,
        actions=jnp.zeros((3, 4)),
        next_observations=obs,
        rewards=jnp.zeros(3),
        masks=jnp.ones(3),
    )
    updated, info = agent.update(batch)
    assert all(np.isfinite(np.asarray(v)).all() for v in info.values())
    for before, after in zip(
        jax.tree_util.tree_leaves(agent.base_policy.policy.params),
        jax.tree_util.tree_leaves(updated.base_policy.policy.params),
    ):
        np.testing.assert_array_equal(before, after)
    assert agent.sample_actions(obs, seed=jax.random.PRNGKey(1)).shape == (3, 4)


@pytest.mark.parametrize("name", ["rlt", "gsbr", "td3bc", "expo", "dsrl_na", "dice_rl"])
def test_delayed_actor_update_preserves_parameters_and_optimizer(name):
    agent = make_agent(name)
    batch = dict(
        observations=jnp.ones((3, 4)),
        next_observations=jnp.zeros((3, 4)),
        actions=jnp.zeros((3, 4)),
        rewards=jnp.ones(3),
        masks=jnp.ones(3),
    )
    # Populate Adam momentum before skipping, to catch accidental optimizer steps.
    agent, _ = agent.update(batch)
    updated, metrics = agent.update(batch, update_actor=False)
    assert all(np.isfinite(np.asarray(value)).all() for value in metrics.values())
    for module in ["actor", "temperature"]:
        key = f"modules_{module}"
        if key in agent.network.params:
            for old, new in zip(
                jax.tree_util.tree_leaves(agent.network.params[key]),
                jax.tree_util.tree_leaves(updated.network.params[key]),
            ):
                np.testing.assert_array_equal(old, new)
        if module in agent.network.opt_state:
            for old, new in zip(
                jax.tree_util.tree_leaves(agent.network.opt_state[module]),
                jax.tree_util.tree_leaves(updated.network.opt_state[module]),
            ):
                np.testing.assert_array_equal(old, new)
    assert any(
        not np.array_equal(old, new)
        for old, new in zip(
            jax.tree_util.tree_leaves(agent.network.params["modules_critic"]),
            jax.tree_util.tree_leaves(updated.network.params["modules_critic"]),
        )
    )


def test_rlt_single_sample_passthrough_and_td3bc_independence():
    obs = jnp.ones((3, 4))
    for name, cls in [("rlt", RLTAgent), ("td3bc", TD3BCAgent)]:
        agent = make_agent(name)
        refs = agent._sample_base_actions_n(obs, jax.random.PRNGKey(1))
        assert refs.shape == (1, 3, 4)
        assert "n_base_samples" not in agent.config
        assert "ref_action_passthrough" not in agent.config
        assert agent.reference_passthrough == (name == "rlt")
        a = agent.network.select("actor")(
            obs,
            ref_actions=jnp.zeros_like(refs) if agent.reference_passthrough else None,
        ).mode()
        b = agent.network.select("actor")(
            obs,
            ref_actions=jnp.ones_like(refs) if agent.reference_passthrough else None,
        ).mode()
        assert bool(jnp.allclose(a, b)) == (name == "td3bc")
        loss, _ = agent._reference_bc_loss(jnp.ones((3, 4)), jnp.zeros_like(refs))
        assert float(loss) == 1.0


def test_gsbr_distance_hinge_and_zero_gradient_inside_support():
    agent = make_agent("gsbr")
    refs = jnp.arange(8, dtype=float)[:, None, None] * jnp.ones((8, 1, 4))
    # Every leave-one-out neighbor is distance 1: radius=1 for any quantile.
    loss, info = agent._reference_bc_loss(jnp.full((1, 4), 9.0), refs)
    np.testing.assert_allclose(loss, 1.0, rtol=1e-6)
    np.testing.assert_allclose(info["actor/tube_radius"], 1.0, rtol=1e-6)
    grad = jax.grad(lambda a: agent._reference_bc_loss(a, refs)[0])(
        jnp.full((1, 4), 3.4)
    )
    np.testing.assert_array_equal(grad, 0.0)


def test_transport_layout_and_selected_recipe():
    keys = get_low_dim_obs_keys("transport-mh-low_dim")
    dims = dict(
        robot0_eef_pos=3,
        robot0_eef_quat=4,
        robot0_gripper_qpos=2,
        object=41,
        robot1_eef_pos=3,
        robot1_eef_quat=4,
        robot1_gripper_qpos=2,
    )
    assert sum(dims[k] for k in keys) == 59
    assert keys[-3:] == ("robot1_eef_pos", "robot1_eef_quat", "robot1_gripper_qpos")
    cfg = json.loads((ROOT / "configs/transport-mh-low_dim/gsbr.json").read_text())
    assert cfg["agent"]["agent_name"] == "gsbr"
    assert cfg["agent"]["discount"] == 0.998
    assert "robomimic_transport_full_observation" not in cfg["training"]


def test_all_selected_recipes_load():
    import recipe
    import base_recipe
    import main

    flags = main.FLAGS

    for path in (ROOT / "configs").glob("*/*.json"):
        config = recipe.get_config(path)
        base = base_recipe.get_config(path)
        assert config.agent_name in agents and base.agent_name == "bc"
        settings = json.loads(path.read_text())
        assert (
            type(settings["training"]["n_step"]) is int
            and settings["training"]["n_step"] >= 1
        )
        assert not (set(settings["training"]) - set(flags))
