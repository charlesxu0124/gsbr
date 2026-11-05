import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.utils import get_class
from omegaconf import OmegaConf

from test_cached_collection import _FakeInnerPolicy, _FakeObsWrangler, _OneStepEnv
from src.outer_policies.base import SamplingMode

ROOT = Path(__file__).resolve().parents[1]


def make_agent(name, pixels=False):
    module = importlib.import_module("nicerl.agents." + name)
    cls = getattr(
        module, {"rlt": "RLTAgent", "gsbr": "GSBRAgent", "td3bc": "TD3BCAgent"}[name]
    )
    config = module.get_config()
    config.update(
        actor_hidden_dims=(16, 16),
        value_hidden_dims=(16, 16),
        use_encoder=pixels,
        action_horizon=2,
        raw_action_dim=7,
    )
    obs = (
        {
            "state": jnp.ones((2, 4)),
            "pixels": jnp.ones((2, 64, 64, 3, 1), dtype=jnp.uint8),
        }
        if pixels
        else jnp.ones((2, 4))
    )
    return cls.create(0, obs, jnp.zeros((2, 14)), config), obs


def test_rlt_and_td3bc_reference_contract():
    for method in ["rlt", "td3bc"]:
        agent, obs = make_agent(method)
        refs = jnp.zeros((2, 1, 14))
        a = agent.sample_actions(obs, refs, deterministic=True)
        b = agent.sample_actions(obs, jnp.ones_like(refs), deterministic=True)
        assert bool(jnp.allclose(a, b)) == (method == "td3bc")
        assert "n_base_samples" not in agent.config
        with pytest.raises(ValueError):
            agent.sample_actions(obs, jnp.zeros((2, 8, 14)), deterministic=True)


def test_gsbr_per_dimension_loss_and_pixel_update():
    agent, obs = make_agent("gsbr", pixels=True)
    refs = jnp.arange(8, dtype=float)[None, :, None] * jnp.ones((2, 8, 14))
    loss, info = agent._reference_bc_loss(jnp.full((2, 14), 9.0), refs)
    np.testing.assert_allclose(loss, 1.0, rtol=1e-6)
    for d in range(7):
        np.testing.assert_allclose(info[f"tube_radius_dim_{d}"], 1.0, rtol=1e-6)
    grad = jax.grad(lambda x: agent._reference_bc_loss(x, refs)[0])(
        jnp.full((2, 14), 3.4)
    )
    np.testing.assert_array_equal(grad, 0.0)
    batch = dict(
        observations=obs,
        next_observations=obs,
        actions=jnp.zeros((2, 14)),
        base_actions=refs / 8,
        next_base_actions=refs / 8,
        rewards=-jnp.ones(2),
        masks=jnp.ones(2),
    )
    updated, metrics = agent.update(batch)
    assert all(np.isfinite(np.asarray(v)).all() for v in metrics.values())
    assert updated.sample_actions(obs, refs / 8, deterministic=True).shape == (2, 14)


def test_all_forty_selected_configs_construct_and_validate():
    with initialize_config_dir(config_dir=str(ROOT / "src/cfgs"), version_base=None):
        configs = list((ROOT / "src/cfgs").glob("*_t*.yaml"))
        assert len(configs) == 40
        for path in configs:
            cfg = compose(config_name=path.stem)
            assert cfg.train.max_num_trajs == 1000 and cfg.env.chunk_size == 10
            assert cfg.env.env.task_id in [14, 29, 54, 60, 64]
            outer = OmegaConf.to_container(cfg.outer_policy, resolve=True)
            cls = get_class(outer.pop("_target_"))
            outer.update(
                inner_policy=_FakeInnerPolicy(), obs_wrangler=_FakeObsWrangler()
            )
            key = next(k for k in ["learner", "rlt", "bfn", "sac"] if k in outer)
            assert outer[key]["num_qs"] == 10
            if "actor_hidden_dims" in outer[key]:
                outer[key]["actor_hidden_dims"] = [16, 16]
            outer[key].update(
                value_hidden_dims=[16, 16],
                use_encoder=False,
                batch_size=1,
            )
            outer["replay_buffer_size"] = 4
            policy = cls(**outer)
            policy.setup(_OneStepEnv().reset()[0])
            if hasattr(policy, "cache_num_base_samples"):
                assert policy.cache_num_base_samples == 16
            if path.stem.startswith("rlt_"):
                assert (
                    policy.cache_num_base_samples == 16 and policy._reference_count == 1
                )
                assert policy._agent.reference_passthrough
            if path.stem.startswith("td3bc_"):
                assert (
                    policy.cache_num_base_samples == 16 and policy._reference_count == 1
                )
                assert not policy._agent.reference_passthrough
            if path.stem.startswith("gsbr_"):
                assert (
                    policy.cache_num_base_samples == 16 and policy._reference_count == 8
                )
                assert policy._agent.config["base_action_dropout"] == 0


def test_dsrl_collect_update_preserves_latent_action_contract():
    from src.outer_policies.dsrl import DSRLPolicy

    policy = DSRLPolicy(
        inner_policy=_FakeInnerPolicy(),
        obs_wrangler=_FakeObsWrangler(),
        sac=dict(
            batch_size=1,
            actor_hidden_dims=[16, 16],
            value_hidden_dims=[16, 16],
            use_encoder=False,
            chunk_size=1,
            action_chunking=False,
        ),
        cameras=[],
        image_resolution=[8, 8],
        replay_buffer_size=4,
        seed=0,
        max_action=2.0,
    )
    policy.setup(_OneStepEnv().reset()[0])
    policy.collect_experience(_OneStepEnv(), SamplingMode.TRAIN)
    metrics = policy.update(1)
    assert all(np.isfinite(np.asarray(v)).all() for v in metrics.values())
