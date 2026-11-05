import tempfile
import unittest

import jax
import numpy as np

from src.outer_policies.base import SamplingMode
from src.outer_policies.cached import CachedPolicy


class _FakeInnerPolicy:
    noise_shape = (2, 3)
    action_horizon = 2
    action_dim = 2

    def __init__(self):
        self.scalar_calls = 0
        self.batch_calls = 0
        self.sample_count = 0

    def sample_action(self, obs, instruction, noise=None, meta=None):
        del obs
        assert instruction == "do the thing"
        assert meta == {"num_steps": 10, "start_step": 0}
        self.scalar_calls += 1
        return np.tanh(noise[:, :2]).astype(np.float32)

    def sample_actions(self, obs, instruction, num_samples, noise=None, meta=None):
        del obs
        assert instruction == "do the thing"
        assert meta == {"num_steps": 10, "start_step": 0}
        assert noise.shape == (num_samples, 2, 3)
        self.batch_calls += 1
        self.sample_count += num_samples
        return np.tanh(noise[:, :, :2]).astype(np.float32)


class _FakeObsWrangler:
    @staticmethod
    def preprocess_rl_obs(obs, cameras, use_state, image_resolution):
        del cameras, use_state, image_resolution
        return np.asarray(obs["state"], dtype=np.float32)


class _OneStepEnv:
    def reset(self):
        return {"task_language": "do the thing", "state": np.zeros(3)}, {}

    def step(self, action):
        assert action.shape == (2, 2)
        return (
            {"task_language": "do the thing", "state": np.ones(3)},
            -1.0,
            False,
            True,
            {"is_success": False},
        )


def _make_policy(inner, agent="rlt", *, cache_num_base_samples=None, **rlt):
    config = {
        "batch_size": 1,
        "actor_hidden_dims": (16, 16),
        "value_hidden_dims": (16, 16),
        "num_qs": 2,
        "use_encoder": False,
    }
    config.update(rlt)
    policy = CachedPolicy(
        inner_policy=inner,
        obs_wrangler=_FakeObsWrangler(),
        cameras=[],
        image_resolution=(8, 8),
        agent=agent,
        cache_num_base_samples=(
            cache_num_base_samples
            if cache_num_base_samples is not None or agent in ("rlt", "td3bc")
            else 16
        ),
        replay_buffer_size=8,
        seed=3,
        rlt=config,
    )
    first_obs, _ = _OneStepEnv().reset()
    policy.setup(first_obs)
    return policy


class RLTOuterPolicyTest(unittest.TestCase):
    def test_every_agent_trains_from_the_cache_without_base_calls(self):
        for agent, n_eval in (
            ("rlt", 1),
            ("gsbr", 8),
            ("td3bc", 1),
            ("expo", 8),
            ("dice_rl", 8),
        ):
            inner = _FakeInnerPolicy()
            policy = _make_policy(inner, agent)
            policy.collect_experience(_OneStepEnv(), SamplingMode.TRAIN)
            calls_before = (inner.batch_calls, inner.sample_count)
            info = policy.update(2)
            self.assertEqual(
                (inner.batch_calls, inner.sample_count), calls_before, agent
            )
            self.assertTrue(np.isfinite(info["actor/total_loss"]), agent)
            first_obs, _ = _OneStepEnv().reset()
            action = policy.sample(first_obs, SamplingMode.EVAL)
            self.assertEqual(action.shape, (2, 2), agent)
            self.assertEqual(inner.sample_count - calls_before[1], n_eval, agent)

    def test_residual_policy_requests_one_base_chunk_at_eval(self):
        inner = _FakeInnerPolicy()
        policy = _make_policy(inner, "expo", n_base_samples=1, n_residual_samples=1)
        policy.collect_experience(_OneStepEnv(), SamplingMode.TRAIN)
        self.assertEqual(inner.sample_count, 32)
        first_obs, _ = _OneStepEnv().reset()
        policy.sample(first_obs, SamplingMode.EVAL)
        self.assertEqual(inner.sample_count, 33)

    def test_unknown_agent_key_is_rejected(self):
        with self.assertRaises(KeyError):
            _make_policy(_FakeInnerPolicy(), "expo", nearest_tube_quantile=0.5)

    def test_rlt_caches_one_current_and_next_chunk(self):
        inner = _FakeInnerPolicy()
        policy = _make_policy(inner, cache_num_base_samples=1)

        rollouts, _ = policy.collect_experience(_OneStepEnv(), SamplingMode.RANDOM)

        self.assertEqual(inner.scalar_calls, 0)
        self.assertEqual(inner.batch_calls, 2)
        self.assertEqual(inner.sample_count, 2)
        self.assertEqual(len(rollouts["rl_rollout"]), 1)
        self.assertEqual(policy._replay_buffer.size, 1)
        self.assertEqual(policy._replay_buffer["actions"][0].shape, (4,))
        self.assertEqual(policy._replay_buffer["base_actions"][0].shape, (1, 4))
        self.assertEqual(policy._replay_buffer["next_base_actions"][0].shape, (1, 4))
        self.assertEqual(
            len(np.unique(policy._replay_buffer["base_actions"][0], axis=0)), 1
        )

        calls_before_update = (inner.batch_calls, inner.sample_count)
        info = policy.update(1)
        self.assertEqual((inner.batch_calls, inner.sample_count), calls_before_update)
        self.assertTrue(np.isfinite(info["actor/total_loss"]))

    def test_td3bc_caches_sixteen_but_resamples_one_reference(self):
        self._check_sixteen_sample_cache("td3bc", passthrough=False)

    def test_rlt_caches_sixteen_but_resamples_one_reference(self):
        self._check_sixteen_sample_cache("rlt", passthrough=True)

    def _check_sixteen_sample_cache(self, method, passthrough):
        inner = _FakeInnerPolicy()
        policy = _make_policy(inner, method)
        policy.collect_experience(_OneStepEnv(), SamplingMode.TRAIN)

        self.assertEqual(policy.cache_num_base_samples, 16)
        self.assertEqual(policy._reference_count, 1)
        self.assertEqual(policy._agent.reference_passthrough, passthrough)
        self.assertEqual(inner.sample_count, 32)
        refs = policy._replay_buffer["base_actions"][:1]
        self.assertEqual(refs.shape, (1, 16, 4))
        self.assertEqual(policy._replay_buffer["next_base_actions"][0].shape, (16, 4))
        draws = [
            np.asarray(policy._agent._subsample_base_actions(refs, jax.random.PRNGKey(i)))
            for i in range(16)
        ]
        self.assertTrue(all(draw.shape == (1, 1, 4) for draw in draws))
        self.assertGreater(len(np.unique(np.concatenate(draws)[:, 0], axis=0)), 1)
        for draw in draws:
            self.assertTrue(np.any(np.all(refs[0] == draw[0, 0], axis=-1)))

        calls = (inner.batch_calls, inner.sample_count)
        metrics = policy.update(2)
        self.assertEqual((inner.batch_calls, inner.sample_count), calls)
        self.assertTrue(np.isfinite(metrics["actor/total_loss"]))

    def test_td3bc_restore_checks_cache_width(self):
        for width in (1, 16):
            with self.subTest(width=width), tempfile.TemporaryDirectory() as root:
                saved = _make_policy(
                    _FakeInnerPolicy(), "td3bc", cache_num_base_samples=width
                )
                saved.collect_experience(_OneStepEnv(), SamplingMode.TRAIN)
                saved.save_state(root)
                restored = _make_policy(_FakeInnerPolicy(), "td3bc")
                if width == 1:
                    with self.assertRaisesRegex(ValueError, "cache has 1 samples"):
                        restored.load_state(root)
                else:
                    restored.load_state(root)
                    np.testing.assert_array_equal(
                        restored._replay_buffer["base_actions"],
                        saved._replay_buffer["base_actions"],
                    )
                    self.assertTrue(np.isfinite(restored.update(1)["actor/total_loss"]))


if __name__ == "__main__":
    unittest.main()
