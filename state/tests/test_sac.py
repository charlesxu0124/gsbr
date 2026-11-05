from pathlib import Path
from types import SimpleNamespace
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from agents.sac import SACAgent, get_config
from utils.flax_utils import save_agent, restore_agent
from train import command


ROOT = Path(__file__).resolve().parents[1]


def make_agent(**overrides):
    config = get_config()
    config.update(
        actor_hidden_dims=(16, 16),
        value_network_kwargs=dict(hidden_dims=(16, 16), layer_norm=False),
    )
    config.update(overrides)
    return SACAgent.create(0, jnp.ones((4, 3)), jnp.zeros((4, 2)), config)


def batch():
    return dict(
        observations=jnp.ones((4, 3)),
        next_observations=jnp.zeros((4, 3)),
        actions=jnp.zeros((4, 2)),
        rewards=jnp.array([1.0, 2.0, 3.0, 4.0]),
        masks=jnp.array([0.0, 1.0, 0.0, 1.0]),
    )


def same_tree(left, right):
    return all(
        np.array_equal(a, b)
        for a, b in zip(
            jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)
        )
    )


def test_sac_updates_actor_critics_temperature_and_polyak_targets(tmp_path):
    agent = make_agent()
    updated, metrics = agent.update(batch())
    assert all(np.isfinite(np.asarray(v)).all() for v in metrics.values())
    for name in ["actor", "critic", "temperature"]:
        assert not same_tree(
            agent.network.params[f"modules_{name}"],
            updated.network.params[f"modules_{name}"],
        )
    expected = jax.tree_util.tree_map(
        lambda old, new: 0.995 * old + 0.005 * new,
        agent.network.params["modules_target_critic"],
        updated.network.params["modules_critic"],
    )
    for a, b in zip(
        jax.tree_util.tree_leaves(expected),
        jax.tree_util.tree_leaves(updated.network.params["modules_target_critic"]),
    ):
        np.testing.assert_allclose(a, b, atol=1e-7)
    target = agent._td_target(batch(), jax.random.PRNGKey(2))
    np.testing.assert_array_equal(target[::2], [1.0, 3.0])

    # A skipped actor step also preserves its Adam moments and entropy optimizer.
    critic_only, _ = updated.update(batch(), update_actor=False)
    for name in ["actor", "temperature"]:
        assert same_tree(
            updated.network.params[f"modules_{name}"],
            critic_only.network.params[f"modules_{name}"],
        )
        assert same_tree(
            updated.network.opt_state[name], critic_only.network.opt_state[name]
        )

    save_agent(updated, tmp_path, 1)
    restored = restore_agent(make_agent(), str(tmp_path), 1)
    assert same_tree(updated, restored)


@pytest.mark.parametrize("target, direction", [(100.0, 1), (-100.0, -1)])
def test_temperature_moves_toward_entropy_target(target, direction):
    agent = make_agent(target_entropy=target)
    updated, _ = agent.update(batch())
    assert direction * float(updated._temperature() - agent._temperature()) > 0


def test_scaled_chunk_actions_and_deterministic_evaluation():
    agent = make_agent(horizon_length=2, action_clip=5.0, gripper_action_clip=1.0)
    obs = jnp.ones(3)
    actions = agent.sample_actions(obs, seed=jax.random.PRNGKey(1))
    assert actions.shape == (4,)
    assert np.all(np.abs(actions) <= [5, 1, 5, 1])
    np.testing.assert_array_equal(
        agent.sample_actions(obs, seed=jax.random.PRNGKey(1), temperature=0),
        agent.sample_actions(obs, seed=jax.random.PRNGKey(2), temperature=0),
    )
    data = batch()
    data.update(
        actions=jnp.zeros((4, 2, 2)),
        rewards=jnp.ones((4, 2)),
        masks=jnp.ones((4, 2)),
        next_observations=jnp.zeros((4, 2, 3)),
        n_steps=jnp.full(4, 2),
    )
    _, info = agent.update(data)
    assert all(np.isfinite(np.asarray(v)).all() for v in info.values())


def test_sac_task_commands_need_no_bc_checkpoint():
    for task in (ROOT / "configs").iterdir():
        if not task.is_dir():
            continue
        args = SimpleNamespace(
            task=task.name, method="sac", seed=0, wandb=False, data_dir=None
        )
        cmd = command(args)
        assert "--agent=agents/sac.py" in cmd
        assert "--horizon_length=1" in cmd and "--prefill_episodes=0" in cmd
        assert not any(
            "restore_base_policy" in value or "--base_policy=" in value for value in cmd
        )


def test_sac_real_trainer_from_scratch_and_resume(tmp_path):
    args = [
        sys.executable,
        str(ROOT / "tests/trainer_smoke.py"),
        "--agent=agents/sac.py",
        "--agent.actor_update_period=2",
        "--agent.actor_hidden_dims=(16,16)",
        "--agent.value_network_kwargs.hidden_dims=(16,16)",
        "--agent.batch_size=2",
        "--env_name=smoke",
        "--horizon_length=1",
        "--offline_steps=0",
        "--online_steps=4",
        "--online_start_training=2",
        "--random_steps=2",
        "--balanced_sampling=false",
        "--prefill_episodes=0",
        "--debug=true",
        "--wandb_log_code=false",
        "--render_videos=false",
        "--video_episodes=0",
        "--buffer_size=64",
        "--online_log_interval=1",
        "--online_save_interval=2",
        "--online_eval_interval=2",
        "--eval_batch_size=1",
        "--eval_episodes=1",
        "--wandb_run_group=sac",
        f"--save_dir={tmp_path}",
    ]
    env = dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1")
    for extra in [[], ["--online_steps=6"], ["--online_steps=6", "--eval_only=true"]]:
        result = subprocess.run(
            args + extra, cwd=ROOT, env=env, text=True, capture_output=True, timeout=120
        )
        assert result.returncode == 0, result.stdout + result.stderr
    run_dir = next((tmp_path / "sac").glob("*/params_6.pkl")).parent
    assert (run_dir / "eval.csv").exists() and (
        run_dir / "replay_buffer_6.pkl"
    ).exists()
