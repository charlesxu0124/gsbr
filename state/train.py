"""Run one selected paper recipe on a local CPU/GPU; no scheduler required."""

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
METHODS = (
    "bc",
    "sac",
    "gsbr",
    "rlt",
    "td3bc",
    "expo",
    "residual",
    "q_chunking",
    "dsrl",
    "dice_rl",
)


def command(args, overrides=()):
    recipe = (
        ROOT
        / "configs"
        / args.task
        / f"{args.bc_for if args.method == 'bc' else 'gsbr' if args.method == 'sac' else args.method}.json"
    )
    saved = json.loads(recipe.read_text())
    settings = dict(saved["training"])
    settings.update(
        seed=args.seed,
        wandb_run_group=f"{args.method}_{args.task}",
        wandb_project="gsbr",
        debug=not args.wandb,
        wandb_log_code=False,
        bc_steps=getattr(args, "bc_steps", None) or 0,
        offline_steps=getattr(args, "offline_steps", 0),
    )
    if getattr(args, "online_steps", None) is not None:
        settings["online_steps"] = args.online_steps
    cmd = [sys.executable, str(ROOT / "main.py")]
    if args.method == "bc":
        if settings["offline_steps"] or getattr(args, "online_steps", None):
            raise ValueError(
                "--method bc uses --bc-steps; RL budgets belong to an RL method"
            )
        settings.update(
            bc_steps=args.bc_steps if args.bc_steps is not None else 1000000,
            offline_steps=0,
            online_steps=0,
            balanced_sampling=False,
            prefill_episodes=0,
            n_step=1,
            chunk_discount="step",
        )
        cmd.append(f"--agent=base_recipe.py:{recipe}")
    elif args.method == "sac":
        # Reuse only the task's environment/evaluation settings. SAC defaults
        # are a standalone training option, not a selected paper recipe.
        settings.update(
            balanced_sampling=False,
            prefill_episodes=0,
            horizon_length=1,
            n_step=1,
            chunk_discount="step",
            utd=1,
            actor_update_block=1,
            online_start_training=1000,
            random_steps=1000,
        )
        cmd.append("--agent=agents/sac.py")
        if "-cpu-" in args.task:
            cmd += [
                f"--agent.action_clip={settings['ocbench_action_norm_clip']}",
                "--agent.gripper_action_clip=1.0",
            ]
    else:
        if (
            not args.base_policy_checkpoint
            and not settings["bc_steps"]
            and not args.dry_run
        ):
            raise ValueError(
                "Provide --base-policy-checkpoint or train a fresh base with --bc-steps"
            )
        if args.base_policy_checkpoint or not settings["bc_steps"]:
            settings.update(
                restore_base_policy_path=args.base_policy_checkpoint
                or "PATH_TO_BC_CHECKPOINT",
                restore_base_policy_epoch=args.base_policy_step,
            )
        cmd += [
            f"--agent=recipe.py:{recipe}",
            f"--base_policy=base_recipe.py:{recipe}",
        ]
    if args.data_dir:
        key = "ocbench_dataset_dir" if "-cpu-" in args.task else "ogbench_dataset_dir"
        settings[key] = str(Path(args.data_dir).resolve())
    for key, value in settings.items():
        if value is not None:
            cmd.append(
                f"--{key}={str(value).lower() if isinstance(value, bool) else value}"
            )
    return cmd + list(overrides)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        required=True,
        choices=sorted(p.name for p in (ROOT / "configs").iterdir() if p.is_dir()),
    )
    parser.add_argument("--method", choices=METHODS, default="gsbr")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-policy-checkpoint")
    parser.add_argument("--base-policy-step", type=int, default=1000000)
    parser.add_argument(
        "--bc-steps",
        type=int,
        default=None,
        help="BC gradient updates; defaults to 1M for --method bc, otherwise 0",
    )
    parser.add_argument(
        "--offline-steps", type=int, default=0, help="Offline RL gradient updates"
    )
    parser.add_argument(
        "--online-steps",
        type=int,
        default=None,
        help="Online environment steps; defaults to the task recipe",
    )
    parser.add_argument(
        "--bc-for",
        choices=[method for method in METHODS[1:] if method != "sac"],
        default="gsbr",
        help="Use this RL method's matching base-policy recipe for BC",
    )
    parser.add_argument(
        "--data-dir",
        help="Local OGBench/OCBench dataset directory; RoboMimic uses ~/.robomimic",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    cmd = command(args, overrides)
    if args.dry_run:
        print(shlex.join(cmd))
    else:
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
