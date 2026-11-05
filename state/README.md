# State benchmarks

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/). Install from this directory:

```bash
uv sync --locked --extra test
# NVIDIA GPU and RoboMimic:
uv sync --locked --extra cuda --extra robomimic --extra test
```

Activate `.venv` or use `.venv/bin/python` below. For headless rendered evaluation, configure the host's MuJoCo rendering backend (typically `MUJOCO_GL=egl` on NVIDIA machines). `--render_videos=false --video_episodes=0` disables videos. Unit and synthetic training tests run on CPU:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

## Data

**RoboMimic.** The recipes use `can-mh-low_dim`, `square-mh-low_dim`, `transport-mh-low_dim`, and `tool_hang-ph-low_dim`. Install the `robomimic` extra above, then download the simulator-v1.5 low-dimensional datasets into `~/.robomimic`:

```bash
.venv/bin/python -m robomimic.scripts.download_datasets \
  --download_dir ~/.robomimic --tasks can square transport \
  --dataset_types mh --hdf5_types low_dim
.venv/bin/python -m robomimic.scripts.download_datasets \
  --download_dir ~/.robomimic --tasks tool_hang \
  --dataset_types ph --hdf5_types low_dim
```

Each task/dataset folder must contain `low_dim_v15.hdf5`. Transport concatenates `robot0_eef_pos`, `robot0_eef_quat`, `robot0_gripper_qpos`, `object`, `robot1_eef_pos`, `robot1_eef_quat`, and `robot1_gripper_qpos`, in that order: 59 dimensions. A checkpoint trained on the older 50D state is incompatible.

**Cube Triple.** Recipes are `cube-triple-play-singletask-task{1,2,3,4,5}-v0`. The selected RL runs use OGBench's default dataset loader and its `~/.ogbench/data` cache (`cube-triple-play-v0.npz` and `cube-triple-play-v0-val.npz`). Missing files are downloaded by OGBench. An optional `--data-dir` supports task-specific local shards when using an equivalent dataset; it is not needed for these recipes.

### OCBench

We use OCBench's **CPU simulator** (`-cpu-` task IDs) for compatiablity with training on TPU.
OCBench is installed with the state environment above.

#### Download datasets

Use the official OCBench downloader to fetch the training and validation shards
from [Hugging Face](https://huggingface.co/datasets/seohongpark/ocbench):

```bash
.venv/bin/python - <<'PYTHON'
from pathlib import Path
import ocbench

tasks = [
    "block-cpu-lite-double-task1-v0",
    "block-cpu-lite-double-task2-v0",
    "chamber-cpu-lite-easy-task1-v0",
    "chamber-cpu-lite-easy-task2-v0",
    "chamber-cpu-lite-easy-task3-v0",
]
for task in tasks:
    train_paths, _ = ocbench.download_datasets(
        task, dataset_root="~/.cache/ocbench/datasets"
    )
    print(task, "--data-dir", Path(train_paths[0]).parent)
PYTHON
```

Remove tasks from the list to download only what you need. Pass the printed
`--data-dir` for that task to both BC and RL commands. For example:

```bash
.venv/bin/python train.py --task block-cpu-lite-double-task1-v0 \
  --method bc --seed 0 \
  --data-dir "$HOME/.cache/ocbench/datasets/lite/block-lite-double-task1-v0"
```

CPU and MJWarp variants share the released datasets, so download directory names
omit `-cpu-`. Keep each training `.npz` alongside its matching `-val.npz` file.
This trainer accepts dense observations (`observation_interval=1`); the released
shards for the five tasks above have this format. For other tasks and dataset
generation, see the [official dataset instructions](https://github.com/seohongpark/ocbench#datasets).

#### Try the MJWarp GPU simulator

For GPU acclerated simulation, use an NVIDIA GPU and follow the
[OCBench setup](https://github.com/seohongpark/ocbench#quick-start) and
[MJWarp requirements](https://github.com/google-deepmind/mujoco_warp#getting-started).
Create a separate environment for this example:

```bash
uv venv --python 3.11 .venv-mjwarp
uv pip install --python .venv-mjwarp/bin/python "ocbench[mjwarp]==1.0.2"
.venv-mjwarp/bin/python - <<'PYTHON'
import ocbench

env = ocbench.make("block-lite-double-task1-v0", nworld=64)
try:
    observations, info = env.reset()
    actions = env.action_space.sample()  # One action per parallel world.
    observations, rewards, terminated, truncated, info = env.step(actions)
    print("Parallel worlds:", env.nworld)
finally:
    env.close()
PYTHON
```

Omitting `-cpu-` selects MJWarp; `nworld` sets the parallel batch size.
**The GSBR `train.py` entrypoint currently supports the CPU environments only.**
The example above exercises the GPU simulator, not GSBR training. Integrating
MJWarp into this trainer requires batched collection, per-world episode/reset
handling, replay insertion, and evaluation. Enforce the episode limit returned
by `ocbench.parse_env_spec` when writing a rollout loop. See OCBench's
[batched usage example](https://github.com/seohongpark/ocbench#usage) and
[reference trainers](https://github.com/seohongpark/ocbench#reference-implementations)
for a starting point; those trainers implement upstream algorithms rather than GSBR.

## Train

Run these commands from `state/` after downloading the dataset. Each command
pretrains the method's base policy and then runs online RL. Replace the task and
seed as needed. For OCBench, also add `--data-dir /path/to/downloaded/task`.

```bash
# GSBR
.venv/bin/python train.py --task can-mh-low_dim --method gsbr --seed 0 \
  --bc-steps 1000000

# RL Token
.venv/bin/python train.py --task can-mh-low_dim --method rlt --seed 0 \
  --bc-steps 1000000

# TD3+BC
.venv/bin/python train.py --task can-mh-low_dim --method td3bc --seed 0 \
  --bc-steps 1000000

# EXPO
.venv/bin/python train.py --task can-mh-low_dim --method expo --seed 0 \
  --bc-steps 1000000

# Residual Policy
.venv/bin/python train.py --task can-mh-low_dim --method residual --seed 0 \
  --bc-steps 1000000

# Q-Chunking
.venv/bin/python train.py --task can-mh-low_dim --method q_chunking --seed 0 \
  --bc-steps 1000000

# DSRL
.venv/bin/python train.py --task can-mh-low_dim --method dsrl --seed 0 \
  --bc-steps 1000000

# DICE-RL
.venv/bin/python train.py --task can-mh-low_dim --method dice_rl --seed 0 \
  --bc-steps 1000000
```

To train BC alone or SAC from scratch:

```bash
# Behavioral cloning
.venv/bin/python train.py --task can-mh-low_dim --method bc --seed 0

# SAC (no base policy required)
.venv/bin/python train.py --task can-mh-low_dim --method sac --seed 0
```

To reuse an existing base policy, replace `--bc-steps` with its checkpoint
arguments. Use a checkpoint trained for the same task and matching method:

```bash
.venv/bin/python train.py --task can-mh-low_dim --method gsbr --seed 0 \
  --base-policy-checkpoint /path/to/bc/run --base-policy-step 1000000
```

Outputs and checkpoints are saved under `exp/`. Repeating the same command
resumes an existing run. Add `--dry-run` to preview a command or `--wandb` to
enable experiment logging.
