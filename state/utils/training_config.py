"""Command-line configuration for BC training and RL fine-tuning."""

from absl import flags
from ml_collections import config_flags

FLAGS = flags.FLAGS
flags.DEFINE_string("wandb_run_group", "gsbr", "Run group.")
flags.DEFINE_string("wandb_project", "gsbr", "Wandb project name.")
flags.DEFINE_boolean("wandb_offline", False, "Whether to run wandb in offline mode.")
flags.DEFINE_boolean("wandb_log_code", True, "Whether to log code to wandb.")
flags.DEFINE_multi_string("wandb_tags", None, "Wandb tags.")
flags.DEFINE_boolean("debug", False, "Whether to run in debug mode.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string(
    "env_name", "cube-triple-play-singletask-task1-v0", "Environment (dataset) name."
)
flags.DEFINE_float("reward_scale", 1.0, "Reward scale.")
flags.DEFINE_float("reward_bias", 1.0, "Reward bias.")
flags.DEFINE_boolean(
    "sparse",
    True,
    "If True, offline train data and online env rewards use sparse shaping: -1 where dense reward was non-zero, else 0 (before reward_scale/bias).",
)
flags.DEFINE_float("p_aug", None, "Probability of applying image augmentation.")
flags.DEFINE_integer("frame_stack", None, "Number of frames to stack.")
flags.DEFINE_string("save_dir", "exp/", "Save directory.")
flags.DEFINE_string("restore_path", None, "Restore path.")
flags.DEFINE_integer("restore_epoch", 0, "Restore epoch.")
flags.DEFINE_string(
    "restore_base_policy_path", None, "Restore RLT base flow policy path."
)
flags.DEFINE_integer(
    "restore_base_policy_epoch", None, "Restore RLT base flow policy epoch."
)
flags.DEFINE_string(
    "override_dataset_path",
    None,
    "Path to saved replay buffer to use as train_dataset.",
)
flags.DEFINE_integer("buffer_size", 5000000, "Replay buffer size.")
flags.DEFINE_integer(
    "bc_steps", 0, "Number of behavior-cloning gradient updates before RL."
)
flags.DEFINE_integer(
    "offline_steps", 0, "Number of offline RL gradient updates from the dataset."
)
flags.DEFINE_integer("log_interval", 5000, "Logging interval.")
flags.DEFINE_integer("eval_interval", 100000, "Evaluation interval.")
flags.DEFINE_integer("save_interval", 100000, "Saving interval.")
flags.DEFINE_integer(
    "eval_episodes", 25, "Number of episodes for ordinary evaluations."
)
flags.DEFINE_integer(
    "million_step_eval_episodes",
    50,
    "Number of evaluation episodes at each million online environment steps.",
)
flags.DEFINE_integer("eval_batch_size", 5, "Evaluation batch size.")
flags.DEFINE_boolean(
    "render_videos",
    True,
    "Master switch for rendering and logging videos. When false, video_episodes is ignored.",
)
flags.DEFINE_integer("video_episodes", 1, "Number of video episodes for each task.")
flags.DEFINE_integer("video_frame_skip", 3, "Frame skip for videos.")
flags.DEFINE_integer(
    "video_log_interval",
    20000,
    "Render diagnostic videos every this many online environment steps. 0 renders only at full online evaluations.",
)
flags.DEFINE_integer("video_fps", 5, "Playback frame rate for logged videos.")
flags.DEFINE_integer(
    "video_max_size",
    640,
    "Maximum width or height of the compressed W&B video. 0 keeps the original size.",
)
flags.DEFINE_integer(
    "horizon_length",
    5,
    "Sequence length for n-step returns and/or action chunking (1 = standard single-step).",
)
flags.DEFINE_integer("online_steps", 500000, "Number of online steps.")
flags.DEFINE_integer(
    "online_start_training",
    0,
    "Online environment steps collected before gradient updates begin.",
)
flags.DEFINE_integer("online_eval_interval", 100000, "Online evaluation interval.")
flags.DEFINE_integer("online_save_interval", 100000, "Online saving interval.")
flags.DEFINE_integer("online_log_interval", 500, "Online logging interval.")
flags.DEFINE_float(
    "utd",
    1.0,
    "Gradient updates per online env step.  Fractional values accumulate (e.g. 0.3125 = DICE-RL's 10 gradient steps per 4 envs x 8-step chunk).",
)
flags.DEFINE_integer(
    "random_steps",
    0,
    "Initial environment steps collected with uniform random actions.",
)
flags.DEFINE_integer(
    "actor_update_block",
    1,
    "Global online env steps per actor-update decision, preserved across episode resets and resumes. The actor updates on every period-th block. DICE-RL updates the actor on every 2nd chunk: block = horizon_length, period 2.",
)
flags.DEFINE_integer(
    "n_step",
    1,
    "Chunks per return window for the chunked (n-step) TD target: rewards are summed over n_step * horizon_length env steps and the critic bootstraps from the state after them.  1 = plain chunk transitions.  Only agents with `_normalize_batch` (SAC/RLT/EXPO/DICE families) accept n_step > 1.",
)
flags.DEFINE_enum(
    "chunk_discount",
    "step",
    ["step", "chunk"],
    "Discounting inside a return window: 'step' applies agent.discount per env step; 'chunk' uses DICE-RL's undiscounted chunk sums with agent.discount ** horizon_length per chunk (see Dataset.sample_sequence).",
)
flags.DEFINE_boolean(
    "bootstrap_on_truncation",
    True,
    "Keep mask = 1 at a time-limit truncation so the critic bootstraps from the last state.  False folds truncation into done (no bootstrap), as DICE-RL's multi-step wrapper does.",
)
flags.DEFINE_boolean(
    "truncate_dataset_at_first_success",
    False,
    "Cut every Robomimic demo at the first step whose cumulative reward reaches 1 (DICE-RL's ph_finetune data), so offline episodes end with one success exactly like online episodes.",
)
flags.DEFINE_boolean(
    "balanced_sampling",
    False,
    "Whether to use balanced sampling for online fine-tuning.",
)
flags.DEFINE_boolean(
    "save_online_buffer", True, "Whether to save the online replay buffer."
)
flags.DEFINE_boolean(
    "eval_only",
    False,
    "Run a single evaluation pass and exit (no training). Requires --restore_path.",
)
flags.DEFINE_integer(
    "online_buffer_save_interval",
    5000,
    "Interval (in steps) to save the online replay buffer.",
)
flags.DEFINE_integer(
    "prefill_episodes",
    0,
    "Number of episodes to prefill the replay buffer with the base flow policy before online training. 0 disables prefill.",
)
flags.DEFINE_string(
    "prefill_cache_path",
    None,
    "Path to a pickle of cached prefill transitions. If unset, collect fresh transitions without caching.",
)
config_flags.DEFINE_config_file("agent", "agents/gsbr.py", lock_config=False)
config_flags.DEFINE_config_file("base_policy", "agents/bc.py", lock_config=False)
flags.DEFINE_boolean(
    "auto_resume",
    True,
    "Automatically resume from latest checkpoint and W&B run in save dir.",
)
flags.DEFINE_boolean(
    "resume_wandb_from_restore_path",
    False,
    "Resume the W&B run ID stored in an explicitly supplied --restore_path. Useful when code or config changes produce a new save-dir hash.",
)
flags.DEFINE_string(
    "ogbench_dataset_dir", None, "Directory of OGBench 100M dataset .npz slices."
)
flags.DEFINE_string(
    "ocbench_dataset_dir", None, "Directory of OCBench dataset .npz shards."
)
flags.DEFINE_float(
    "ocbench_action_norm_percentile",
    1.0,
    "Lower/upper percentile used to normalize OCBench actions.",
)
flags.DEFINE_float(
    "ocbench_action_norm_clip", 5.0, "Normalized OCBench non-gripper action limit."
)
flags.DEFINE_boolean(
    "ocbench_load_all_shards",
    False,
    "Load every OCBench train shard at once (concatenated; the action normalizer is still fitted on the first shard) instead of one shard rotated every dataset_replace_interval steps. Rotation happens while train_dataset is sampled (offline steps, or online with balanced_sampling), so an online-only run without balanced sampling otherwise trains on the first shard for its whole duration.",
)
flags.DEFINE_integer(
    "dataset_replace_interval",
    1000,
    "Steps between dataset slice swaps (0 = disabled).",
)
flags.DEFINE_float(
    "dataset_proportion", 1.0, "Fraction of slice files to use (1.0 = all)."
)
