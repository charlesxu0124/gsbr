# LIBERO with pi0.5

The learner/environment and the OpenPI server use **separate environments**. The learner's pinned JAX/Flax versions match the experiments; OpenPI brings its own model dependencies. Both processes can run on the same host or communicate over HTTP.

## Learner and environment

From this directory, with Python 3.11 and uv:

```bash
uv sync --locked --extra cuda --extra test
mkdir -p submodule
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git submodule/liberoenv
git -C submodule/liberoenv checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
uv pip install --python .venv/bin/python --no-deps -e submodule/liberoenv
```

Omit `--extra cuda` for CPU tests. Use a separate environment from the state trainer. Initialize LIBERO's asset paths on first import and ensure its BDDL files, assets, and initial states are available. The outer learner trains from online episodes; it does not require demonstration downloads.

```bash
.venv/bin/python -c 'from libero.libero import get_libero_path; print(get_libero_path("bddl_files"))'
```

For headless rendering on NVIDIA machines, set `MUJOCO_GL=egl`. A working MuJoCo renderer is required for image observations.

## Frozen pi0.5 server

Create an OpenPI checkout separately (for example `/path/to/openpi`) at the recorded revision:

```bash
git clone https://github.com/Physical-Intelligence/openpi.git /path/to/openpi
git -C /path/to/openpi checkout 215abfb217dbac7d5f1273282331b9b1866c0479
cd /path/to/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11
uv pip install --python .venv/bin/python -r /path/to/gsbr/libero/server-requirements.txt
```

Return to this release's `libero/` directory and launch using the **server environment's Python**:

```bash
/path/to/openpi/.venv/bin/python -m src.vla.pi_server
```

The server downloads the public `pi05_libero` checkpoint from `gs://openpi-assets/checkpoints/pi05_libero` through OpenPI. It listens on `127.0.0.1:8400` by default. It preserves the checkpoint's input/output transforms, samples 10-step chunks, and returns the seven physical action dimensions. Batched requests use the final shared-prefix JAX sampler: one observation prefill per request and independent noise suffixes for the action samples. The KV cache is never reused across observations. Model weights are not bundled with this repository. To use a verified local copy, append `checkpoint_path=/path/to/pi05_libero` to the server command.

To change the address, append `host=... port=...` and set the learner's `inner_policy.host_ip` / `inner_policy.host_port` to match.

## Train and evaluate

In another terminal, from the release's `libero/` directory:

```bash
MUJOCO_GL=egl .venv/bin/python -m src.train --config-name=gsbr_t14 seed=0
```

Configs follow `<method>_t<task-id>.yaml`. Methods are `gsbr`, `rlt`, `td3bc`, `expo`, `residual`, `q_chunking`, `dsrl`, and `dice_rl`. Tasks are 14, 29, 54, 60, and 64 in **LIBERO-90**. Use the full task config to preserve its selected hyperparameters, rather than overriding only `task_id`.

Recipes train for 1,000 episodes with 25 initial base-policy episodes, the selected update ratio, checkpointing, and a final 50-episode evaluation. All methods with a base-action replay cache store 16 chunks per state. GSBR trains with eight references drawn from that cache, zero reference dropout, and seven tube radii (one per physical action dimension). RLT caches 16 chunks per state, resamples one reference per learner update, and conditions its actor on that reference; evaluation uses one fresh reference. TD3BC caches 16 chunks per state, resamples one reference per learner update, and uses no actor reference-action input; evaluation uses one reference. W&B is disabled by default; enable it with `wandb.mode=online` and your own project/entity.

Outputs live in `results/<PolicyClass>/<exp_name>/`: resolved config, metrics, rollout/video files, and checkpoints. `do_restore=true` resumes the policy, replay, counters, and performance tracker. Use a new `exp_name` for a fresh experiment.

Evaluate the frozen base policy alone:

```bash
MUJOCO_GL=egl .venv/bin/python -m src.rollout --config-name=rollout_libero
```

For another selected task, set `env.env.task_id=29` (the wrapper's task ID interpolates from it). For a saved outer learner, rerun its original training command with the same `results_dir` and `exp_name`; when the saved episode count reaches `train.max_num_trajs`, the trainer restores it and performs the final evaluation without collecting more training episodes.

## Tests

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
```

Tests exercise cached collection, frozen-base inference counts, pixel GSBR updates, RLT passthrough, DSRL latent-action handling, and construction of all 40 recipes. These tests use a synthetic base-policy client and do not download model weights.
