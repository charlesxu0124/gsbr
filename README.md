# GSBR

**Geometric Support Behavior Regularization for Online RL**

[Project website](https://charlesxu0124.github.io/gsbr/) · [Paper](https://arxiv.org/abs/2611.05295)
<!-- Replace the website and paper URLs before publishing. -->

Train and evaluate GSBR, compare it with baseline methods, or build on the agents and training tools for your own experiments.

The repository supports state-based control on **Cube, RoboMimic, and OCBench**, and vision-based control on **LIBERO** with a frozen pi0.5 policy. Baselines include RL Token, TD3+BC, EXPO, Residual Policy, Q-Chunking, DSRL, and DICE-RL.

## Getting started

Choose the guide for your benchmark. Each trainer has its own environment and dependency lockfile.

- **[State experiments](state/README.md):** installation, datasets, base-policy training, and online RL.
- **[LIBERO experiments](libero/README.md):** installation, pi0.5 serving, training, and evaluation.

### State experiments

Install [uv](https://docs.astral.sh/uv/), then set up the environment:

```bash
cd state
uv sync --locked --extra cuda --extra robomimic
```

Omit `--extra cuda` for CPU use. The `robomimic` extra is needed only for RoboMimic tasks. Follow the [data setup](state/README.md#data) for your chosen benchmark before training.

Pretrain a base policy, then use its checkpoint to train GSBR:

```bash
.venv/bin/python train.py --task can-mh-low_dim --method bc --seed 0

.venv/bin/python train.py --task can-mh-low_dim --method gsbr --seed 0 \
  --base-policy-checkpoint /path/to/bc/run --base-policy-step 1000000
```

Replace `/path/to/bc/run` with the base-policy output directory. Change `--task`, `--method`, and `--seed` to select an experiment; add `--dry-run` to preview the command. The [state guide](state/README.md) explains configuration, checkpoint resume, and evaluation.

### LIBERO experiments

Complete the [LIBERO setup](libero/README.md), including starting the pi0.5 server. Then, from `libero/`:

```bash
MUJOCO_GL=egl .venv/bin/python -m src.train --config-name=gsbr_t14 seed=0
```

Choose a task and method using the configurations in `libero/src/cfgs/`. See the [LIBERO guide](libero/README.md#train-and-evaluate) for training and evaluation options.

## Developing with GSBR

- `state/agents/` and `libero/nicerl/`: learning algorithms and shared components.
- `state/configs/` and `libero/src/cfgs/`: experiment configurations.
- `state/tests/` and `libero/tests/`: tests and small examples for checking changes.

Both trainers support checkpointing and resume. Datasets and pretrained weights are obtained separately, as described in the guides.