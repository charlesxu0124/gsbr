"""Run directories, checkpoint restoration, agents, and experiment tracking."""

import glob
import hashlib
import json
import os
import re
from agents import (
    agents,
    BehaviorRegularizedAgent,
    BFNAgent,
    EXPOAgent,
    DICERLAgent,
    DSRLNAAgent,
    SACAgent,
)
from agents.bc import get_config as bc_config
from envs.benchmark import configure_agent_observations
from utils.flax_utils import restore_agent, _resolve_checkpoint_dir_from_glob
from utils.log_utils import get_flag_dict, setup_wandb


def needs_base_policy(config):
    return issubclass(
        agents[config["agent_name"]],
        (BehaviorRegularizedAgent, BFNAgent, EXPOAgent, DICERLAgent, DSRLNAAgent),
    )


def match_base_checkpoint_config(options, config):
    """Resolve BC metadata before constructing networks, datasets, or run names."""
    if not options.restore_base_policy_path:
        return
    directory = _resolve_checkpoint_dir_from_glob(options.restore_base_policy_path)
    options.restore_base_policy_path = directory
    if options.restore_base_policy_epoch is None:
        steps = [
            int(re.fullmatch(r"params_(\d+)\.pkl", os.path.basename(path))[1])
            for path in glob.glob(os.path.join(directory, "params_*.pkl"))
            if re.fullmatch(r"params_(\d+)\.pkl", os.path.basename(path))
        ]
        if not steps:
            raise FileNotFoundError(f"No BC checkpoints found in {directory}")
        options.restore_base_policy_epoch = max(steps)
    checkpoint = os.path.join(
        directory, f"params_{options.restore_base_policy_epoch}.pkl"
    )
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(checkpoint)
    metadata_path = os.path.join(directory, "flags.json")
    if not os.path.isfile(metadata_path):
        print(
            f"No BC metadata at {metadata_path}; using the supplied base-policy configuration.",
            flush=True,
        )
        return
    with open(metadata_path) as handle:
        saved = json.load(handle)
    saved_agent = saved["agent"]
    if saved_agent.get("agent_name") != "bc":
        raise ValueError("The base checkpoint must contain a standalone BC policy")
    if saved.get("env_name", options.env_name) != options.env_name:
        raise ValueError(
            "BC checkpoint and experiment must use the same benchmark task"
        )
    if options.env_name.startswith("transport-"):
        if saved.get("robomimic_transport_full_observation") is False:
            raise ValueError(
                "Legacy 50D Transport checkpoints cannot be used; a full 59D BC policy is required"
            )
        expected = {}
        configure_agent_observations(options.env_name, expected)
        if "observation_keys" in saved_agent and tuple(
            saved_agent["observation_keys"]
        ) != tuple(expected["observation_keys"]):
            raise ValueError(
                "BC checkpoint does not use the required 59D Transport observation layout"
            )
    base = bc_config()
    base.update({key: value for key, value in saved_agent.items() if key in base})
    horizon = saved.get("horizon_length", base.horizon_length)
    if horizon < 1 or not base.action_chunking:
        raise ValueError(
            "The RL trainer requires a chunked BC policy with a positive horizon"
        )
    if (
        options.horizon_length > 1
        and options.actor_update_block == options.horizon_length
    ):
        options.actor_update_block = horizon
    base.horizon_length = options.horizon_length = horizon
    options.base_policy = base
    for key in ("action_clip", "gripper_action_clip"):
        if key in config:
            config[key] = base[key]
    if "frame_stack" in saved:
        options.frame_stack = saved["frame_stack"]
    if "-cpu-" in options.env_name:
        for key in ("ocbench_action_norm_percentile", "ocbench_action_norm_clip"):
            if key in saved:
                setattr(options, key, saved[key])
    print(f"Matched base-policy configuration to {checkpoint}", flush=True)


def prepare_run(options, config):
    if needs_base_policy(config):
        match_base_checkpoint_config(options, config)
    if options.actor_update_block < 1 or config.get("actor_update_period", 1) < 1:
        raise ValueError("Actor update block and period must be positive")
    if min(options.bc_steps, options.offline_steps, options.online_steps) < 0:
        raise ValueError("Training step budgets must be nonnegative")
    if config["agent_name"] == "bc" and (options.offline_steps or options.online_steps):
        raise ValueError(
            "The BC agent uses --bc_steps; offline_steps and online_steps are for RL"
        )
    if config["agent_name"] == "sac" and options.bc_steps:
        raise ValueError("Standalone SAC has no BC base policy; use bc_steps=0")
    if (
        options.video_log_interval < 0
        or options.video_fps <= 0
        or options.video_max_size < 0
    ):
        raise ValueError("Invalid video logging settings")
    config["horizon_length"] = options.horizon_length
    configure_agent_observations(options.env_name, config)
    agent_name = config["agent_name"]
    explicit_restore_path = options.restore_path
    env_short = re.sub("-(singletask|v0|v1|v2)", "", options.env_name)
    identity = config.to_dict()
    if needs_base_policy(config):
        identity = dict(
            agent=identity,
            base_policy=options.base_policy.to_dict(),
            base_checkpoint=options.restore_base_policy_path,
            base_checkpoint_step=options.restore_base_policy_epoch,
        )
    exp_hash = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:8]
    exp_name = f"{options.wandb_run_group}_{agent_name}_{env_short}_seed{options.seed:02d}_{exp_hash}"
    save_subpath = [options.wandb_run_group]
    if options.dataset_proportion < 1.0:
        save_subpath.append(f"dataset_prop_{options.dataset_proportion:g}")
    save_subpath.append(exp_name)
    options.save_dir = os.path.join(options.save_dir, *save_subpath)
    os.makedirs(options.save_dir, exist_ok=True)
    resume_id = None
    if options.auto_resume:
        pkl_files = glob.glob(os.path.join(options.save_dir, "params_*.pkl"))
        if pkl_files:
            saved_steps = [
                int(re.search("params_(\\d+)\\.pkl", f).group(1)) for f in pkl_files
            ]
            latest_step = max(saved_steps)
            options.restore_path = options.save_dir
            options.restore_epoch = latest_step
            resume_id_file = os.path.join(options.save_dir, "wandb_run_id.txt")
            try:
                with open(resume_id_file, "r") as f:
                    resume_id = f.read().strip() or None
            except FileNotFoundError:
                resume_id = None
            print("*" * 100)
            print(f"Found checkpoint step {latest_step} at {options.restore_path}")
            print(f"Resume ID: {resume_id}")
            print("*" * 100)
    if (
        resume_id is None
        and options.resume_wandb_from_restore_path
        and (explicit_restore_path is not None)
    ):
        resume_id_file = os.path.join(explicit_restore_path, "wandb_run_id.txt")
        with open(resume_id_file, "r") as f:
            resume_id = f.read().strip()
        if not resume_id:
            raise ValueError(f"Empty W&B run ID in {resume_id_file}")
        print(f"Resuming W&B run {resume_id} from {explicit_restore_path}")
    return exp_name, resume_id


def create_agent(options, config, example_batch):
    agent_name = config["agent_name"]
    if needs_base_policy(config):
        bp_config = options.base_policy
        bp_config.update(
            dict(horizon_length=options.horizon_length, action_chunking=True)
        )
        base_policy = agents[bp_config["agent_name"]].create(
            options.seed,
            example_batch["observations"],
            example_batch["actions"],
            bp_config,
        )
        if options.restore_base_policy_path:
            base_policy = restore_agent(
                base_policy,
                options.restore_base_policy_path,
                options.restore_base_policy_epoch,
            )
        agent = agents[agent_name].create(
            options.seed,
            example_batch["observations"],
            example_batch["actions"],
            config,
            base_policy=base_policy,
        )
    else:
        agent = agents[agent_name].create(
            options.seed,
            example_batch["observations"],
            example_batch["actions"],
            config,
        )
    if options.n_step > 1:
        assert isinstance(agent, (SACAgent, BFNAgent, EXPOAgent, DICERLAgent)), (
            f"--n_step > 1 is not supported by agent {agent_name}"
        )
    if options.restore_path:
        agent = restore_agent(agent, options.restore_path, options.restore_epoch)
        print(
            f"Restored agent from {options.restore_path} at epoch {options.restore_epoch}",
            flush=True,
        )
    return agent


def initialize_tracking(options, config, exp_name, resume_id):
    run = setup_wandb(
        project=options.wandb_project,
        group=options.wandb_run_group,
        tags=options.wandb_tags,
        name=exp_name,
        hyperparam_dict=config.to_dict(),
        mode="disabled"
        if options.debug
        else "offline"
        if options.wandb_offline
        else "online",
        log_code=options.wandb_log_code,
        resume_id=resume_id,
        resume_mode="allow" if resume_id is not None else None,
    )
    with open(os.path.join(options.save_dir, "wandb_run_id.txt"), "w") as f:
        f.write(run.id)
    with open(os.path.join(options.save_dir, "flags.json"), "w") as f:
        json.dump(get_flag_dict(), f)
    return run
