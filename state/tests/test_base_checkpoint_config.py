import json
from types import SimpleNamespace

import pytest

from agents.bc import get_config as bc_config
from agents.gsbr import get_config as gsbr_config
from envs.robomimic_observations import TRANSPORT_OBSERVATION_KEYS
from utils.training_setup import match_base_checkpoint_config


def checkpoint(tmp_path, env_name, **metadata):
    base = bc_config().to_dict()
    base.update(
        hidden_dims=[12, 24],
        denoise_steps=12,
        horizon_length=5,
        action_clip=5.0,
        gripper_action_clip=1.0,
    )
    saved = dict(
        agent=base,
        env_name=env_name,
        horizon_length=5,
        frame_stack=None,
        ocbench_action_norm_percentile=2.0,
        ocbench_action_norm_clip=5.0,
    )
    saved.update(metadata)
    (tmp_path / "flags.json").write_text(json.dumps(saved))
    (
        tmp_path / "params_7.pkl"
    ).touch()  # Metadata matching does not deserialize weights.
    return SimpleNamespace(
        restore_base_policy_path=str(tmp_path),
        restore_base_policy_epoch=None,
        env_name=env_name,
        base_policy=bc_config(),
        horizon_length=3,
        actor_update_block=3,
        frame_stack=2,
        ocbench_action_norm_percentile=1.0,
        ocbench_action_norm_clip=1.0,
    )


def test_base_metadata_matches_architecture_and_coordinates_not_rl_learning(tmp_path):
    opts = checkpoint(tmp_path, "block-cpu-lite-double-task1-v0")
    config = gsbr_config()
    config.discount, config.lr, config.bc_loss_weight = 0.97, 0.001, 3.0
    match_base_checkpoint_config(opts, config)
    assert opts.restore_base_policy_epoch == 7
    assert list(opts.base_policy.hidden_dims) == [12, 24]
    assert opts.base_policy.denoise_steps == 12
    assert opts.horizon_length == opts.actor_update_block == 5
    assert opts.frame_stack is None and opts.ocbench_action_norm_percentile == 2.0
    assert opts.ocbench_action_norm_clip == config.action_clip == 5.0
    assert (config.discount, config.lr, config.bc_loss_weight) == (0.97, 0.001, 3.0)


def test_checkpoint_task_must_match(tmp_path):
    opts = checkpoint(tmp_path, "can-mh-low_dim")
    opts.env_name = "square-mh-low_dim"
    with pytest.raises(ValueError, match="same benchmark task"):
        match_base_checkpoint_config(opts, gsbr_config())


def test_resmlp_checkpoint_metadata_preserves_depth(tmp_path):
    opts = checkpoint(tmp_path, "block-cpu-lite-double-task1-v0")
    path = tmp_path / "flags.json"
    saved = json.loads(path.read_text())
    saved["agent"].update(mlp_class="resmlp", hidden_dims=[8] * 8)
    path.write_text(json.dumps(saved))
    match_base_checkpoint_config(opts, gsbr_config())
    assert tuple(opts.base_policy.hidden_dims) == (8,) * 8


def test_resmlp_bc_checkpoint_restores_identical_actions(tmp_path):
    import jax
    import jax.numpy as jnp
    import numpy as np

    from agents.bc import BCAgent
    from utils.flax_utils import restore_agent, save_agent

    opts = checkpoint(tmp_path, "block-cpu-lite-double-task1-v0")
    config = bc_config()
    config.update(mlp_class="resmlp", hidden_dims=(8,) * 8, horizon_length=3)
    observations, actions = jnp.ones((2, 5)), jnp.ones((2, 7))
    original = BCAgent.create(0, observations, actions, config)
    save_agent(original, str(tmp_path), 7)
    saved = json.loads((tmp_path / "flags.json").read_text())
    saved["agent"] = config.to_dict()
    saved["horizon_length"] = 3
    (tmp_path / "flags.json").write_text(json.dumps(saved))
    match_base_checkpoint_config(opts, gsbr_config())
    restored = restore_agent(
        BCAgent.create(1, observations, actions, opts.base_policy), str(tmp_path), 7
    )
    key = jax.random.PRNGKey(3)
    np.testing.assert_array_equal(
        original.sample_actions(observations, seed=key),
        restored.sample_actions(observations, seed=key),
    )


def test_transport_metadata_preserves_full_observation_layout(tmp_path):
    opts = checkpoint(tmp_path, "transport-mh-low_dim")
    path = tmp_path / "flags.json"
    saved = json.loads(path.read_text())
    saved["agent"]["observation_keys"] = list(TRANSPORT_OBSERVATION_KEYS)
    path.write_text(json.dumps(saved))
    match_base_checkpoint_config(opts, gsbr_config())
    saved["robomimic_transport_full_observation"] = False
    path.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="50D Transport"):
        match_base_checkpoint_config(opts, gsbr_config())
