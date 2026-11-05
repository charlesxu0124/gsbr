import os
import pickle
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import jax
import numpy as np
from train import command

ROOT = Path(__file__).resolve().parents[1]


def test_cli_keeps_bc_and_rl_budgets_distinct():
    args = SimpleNamespace(
        task="transport-mh-low_dim",
        method="bc",
        bc_for="gsbr",
        bc_steps=None,
        offline_steps=0,
        online_steps=None,
        seed=0,
        wandb=False,
        data_dir=None,
        dry_run=True,
        base_policy_checkpoint=None,
        base_policy_step=1000000,
    )
    cmd = command(args)
    assert "--bc_steps=1000000" in cmd
    assert "--offline_steps=0" in cmd and "--online_steps=0" in cmd
    args.method, args.bc_steps = "gsbr", 2
    args.offline_steps, args.online_steps = 3, 4
    cmd = command(args)
    assert all(
        f"--{key}={value}" in cmd
        for key, value in [("bc_steps", 2), ("offline_steps", 3), ("online_steps", 4)]
    )
    assert not any("restore_base_policy" in flag for flag in cmd)
    args.method, args.bc_steps, args.online_steps = "sac", 0, 0
    cmd = command(args)
    assert "--offline_steps=3" in cmd and "--online_steps=0" in cmd


def run(tmp_path, *flags):
    args = [
        sys.executable,
        str(ROOT / "tests/trainer_smoke.py"),
        "--env_name=smoke",
        "--debug=true",
        "--wandb_log_code=false",
        "--render_videos=false",
        "--video_episodes=0",
        "--buffer_size=64",
        "--log_interval=1",
        "--online_log_interval=1",
        "--save_interval=1",
        "--online_save_interval=1",
        "--online_eval_interval=1",
        "--eval_interval=1",
        "--eval_batch_size=1",
        "--eval_episodes=1",
        "--horizon_length=2",
        "--wandb_run_group=phases",
        f"--save_dir={tmp_path}",
        *flags,
    ]
    result = subprocess.run(
        args,
        cwd=ROOT,
        env=dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def load(path):
    with path.open("rb") as handle:
        return pickle.load(handle)["agent"]


def test_bc_offline_and_online_budgets_and_resume_at_phase_boundary(tmp_path):
    flags = [
        "--agent=agents/gsbr.py",
        "--agent.actor_hidden_dims=(16,16)",
        "--agent.value_network_kwargs.hidden_dims=(16,16)",
        "--agent.batch_size=2",
        "--base_policy.hidden_dims=(16,16)",
        "--base_policy.batch_size=2",
        "--base_policy.denoise_steps=2",
        "--bc_steps=2",
        "--offline_steps=3",
        "--online_steps=0",
        "--prefill_episodes=1",
        "--balanced_sampling=true",
    ]
    output = run(tmp_path, *flags)
    assert "Training environment steps: 0" in output
    assert "Rolling out" not in output
    directory = next((tmp_path / "phases").glob("*/params_5.pkl")).parent
    bc_end = load(directory / "params_2.pkl")
    offline_end = load(directory / "params_5.pkl")
    assert bc_end["base_policy"]["policy"]["step"] == 3  # exactly two BC updates
    assert bc_end["network"]["step"] == 1
    assert offline_end["network"]["step"] == 4  # exactly three offline RL updates
    assert not list(directory.glob("replay_buffer_*.pkl"))

    output = run(tmp_path, *flags, "--online_steps=2")
    assert "Rolling out 1 BC episodes" in output
    assert "Training environment steps: 8" in output  # six prefill + two online
    online_end = load(directory / "params_7.pkl")
    assert online_end["network"]["step"] == 6
    for earlier in [offline_end, online_end]:
        for expected, actual in zip(
            jax.tree_util.tree_leaves(bc_end["base_policy"]),
            jax.tree_util.tree_leaves(earlier["base_policy"]),
        ):
            np.testing.assert_array_equal(expected, actual)

    # Once online collection has started, resume loads replay and skips prefill.
    output = run(tmp_path, *flags, "--online_steps=3")
    assert "Restored replay buffer from step 7" in output
    assert "Rolling out" not in output
    assert "Training environment steps: 1" in output


def test_offline_sac_never_collects_online_transitions(tmp_path):
    output = run(
        tmp_path,
        "--agent=agents/sac.py",
        "--agent.actor_hidden_dims=(16,16)",
        "--agent.value_network_kwargs.hidden_dims=(16,16)",
        "--agent.batch_size=2",
        "--offline_steps=3",
        "--online_steps=0",
        "--balanced_sampling=false",
    )
    assert "Training environment steps: 0" in output
    checkpoint = next((tmp_path / "phases").glob("*/params_3.pkl"))
    assert load(checkpoint)["network"]["step"] == 4
