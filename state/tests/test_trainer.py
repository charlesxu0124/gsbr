import os
import pickle
import json
from pathlib import Path
import subprocess
import sys
import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_bc_checkpoint_then_gsbr_training_and_evaluation(tmp_path):
    common = [
        sys.executable,
        str(ROOT / "tests/trainer_smoke.py"),
        f"--save_dir={tmp_path}",
        "--env_name=smoke",
        "--debug=true",
        "--wandb_log_code=false",
        "--auto_resume=false",
        "--render_videos=false",
        "--video_episodes=0",
        "--eval_batch_size=1",
        "--eval_episodes=1",
        "--horizon_length=2",
        "--buffer_size=64",
        "--log_interval=1",
        "--online_log_interval=1",
        "--save_interval=2",
        "--save_online_buffer=false",
        "--online_eval_interval=2",
        "--eval_interval=2",
        "--sparse=false",
    ]
    env = dict(os.environ, JAX_PLATFORMS="cpu", OMP_NUM_THREADS="1")

    def run(args):
        result = subprocess.run(
            common + args,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
        assert result.returncode == 0, result.stdout
        return result.stdout

    run(
        [
            "--agent=agents/bc.py",
            "--agent.hidden_dims=(16,16)",
            "--agent.batch_size=2",
            "--agent.denoise_steps=2",
            "--bc_steps=2",
            "--online_steps=0",
            "--wandb_run_group=bc",
        ]
    )
    checkpoint = next((tmp_path / "bc").glob("*/params_2.pkl")).parent
    rl_args = [
        "--agent=agents/gsbr.py",
        "--agent.actor_hidden_dims=(16,16)",
        "--agent.value_network_kwargs.hidden_dims=(16,16)",
        "--agent.batch_size=2",
        # These deliberately conflict with the saved BC model; metadata wins.
        "--base_policy.hidden_dims=(8,8)",
        "--base_policy.denoise_steps=3",
        "--horizon_length=3",
        f"--restore_base_policy_path={checkpoint}",
        "--restore_base_policy_epoch=2",
        "--offline_steps=0",
        "--online_steps=2",
        "--prefill_episodes=1",
        "--balanced_sampling=true",
        "--wandb_run_group=gsbr",
        "--save_online_buffer=true",
        "--online_save_interval=2",
    ]
    run(rl_args)
    assert list((tmp_path / "gsbr").glob("*/params_2.pkl"))
    assert list((tmp_path / "gsbr").glob("*/eval.csv"))
    assert list((tmp_path / "bc").glob("*/eval.csv"))
    assert not list(tmp_path.rglob("base_policy_eval.csv"))
    rl_dir = next((tmp_path / "gsbr").glob("*/params_2.pkl")).parent
    saved_flags = json.loads((rl_dir / "flags.json").read_text())
    assert saved_flags["base_policy"]["hidden_dims"] == [16, 16]
    assert saved_flags["base_policy"]["denoise_steps"] == 2
    assert saved_flags["horizon_length"] == saved_flags["agent"]["horizon_length"] == 2
    with (checkpoint / "params_2.pkl").open("rb") as handle:
        bc = pickle.load(handle)["agent"]["policy"]["params"]
    with (rl_dir / "params_2.pkl").open("rb") as handle:
        base = pickle.load(handle)["agent"]["base_policy"]["policy"]["params"]
    for expected, actual in zip(
        jax.tree_util.tree_leaves(bc), jax.tree_util.tree_leaves(base)
    ):
        np.testing.assert_array_equal(expected, actual)

    # Resume must restore the learner/replay without repeating base-policy prefill.
    output = run(rl_args + ["--online_steps=4", "--auto_resume=true"])
    assert "Restored replay buffer from step 2" in output
    assert "Rolling out 1 BC episodes" not in output
    assert (rl_dir / "params_4.pkl").exists()
    output = run(
        rl_args + [f"--restore_path={rl_dir}", "--restore_epoch=4", "--eval_only=true"]
    )
    assert "Rolling out 1 BC episodes" not in output
