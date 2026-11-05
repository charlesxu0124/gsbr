from types import SimpleNamespace

import numpy as np
import pytest

from envs import ocbench_utils, ogbench_utils
from envs.benchmark import make_benchmark
from trainer_smoke import TinyEnv
from utils.training_data import prepare_dataset


def options(**overrides):
    settings = dict(
        env_name="block-cpu-lite-double-task1-v0",
        ocbench_dataset_dir=None,
        ogbench_dataset_dir=None,
        dataset_proportion=1.0,
        dataset_replace_interval=10,
        ocbench_load_all_shards=False,
        ocbench_action_norm_percentile=0.0,
        ocbench_action_norm_clip=5.0,
        frame_stack=None,
        sparse=True,
        reward_scale=2.0,
        reward_bias=3.0,
        p_aug=None,
        eval_batch_size=2,
        eval_episodes=2,
        online_steps=0,
        million_step_eval_episodes=2,
    )
    settings.update(overrides)
    return SimpleNamespace(**settings)


def write_shard(path, action_scale):
    actions = np.array([[-action_scale, -1], [action_scale, 1]], dtype=np.float32)
    np.savez(
        path,
        observations=np.zeros((2, 4)),
        next_observations=np.ones((2, 4)),
        actions=actions,
        rewards=np.array([0, 1]),
        masks=np.array([1, 0]),
        terminals=np.array([0, 1]),
    )


def test_ocbench_rotation_retains_first_shard_normalizer(tmp_path, monkeypatch):
    for index, scale in enumerate([2, 4]):
        write_shard(tmp_path / f"data-{index}.npz", scale)
        write_shard(tmp_path / f"data-{index}-val.npz", scale)
    normalizers = []

    def make_eval(*args, **kwargs):
        normalizers.append(kwargs["action_normalizer"])
        return TinyEnv()

    monkeypatch.setattr(ocbench_utils, "make_ocbench_eval_env", make_eval)
    opts = options(ocbench_dataset_dir=str(tmp_path))
    with make_benchmark(opts, {}) as benchmark:
        initial = prepare_dataset(benchmark.train_dataset, opts)
        validation = prepare_dataset(benchmark.val_dataset, opts)
        np.testing.assert_allclose(initial["actions"][:, 0], [-1, 1])
        assert not benchmark.shards.should_rotate(9)
        assert benchmark.shards.should_rotate(10)
        rotated = prepare_dataset(benchmark.shards.load(10), opts)
        # Re-fitting the normalizer to shard 2 would incorrectly give [-1, 1].
        np.testing.assert_allclose(rotated["actions"][:, 0], [-2, 2])
        for data in [initial, validation, rotated]:
            np.testing.assert_allclose(data["rewards"], [3, 1])
        benchmark.make_eval_env().close()
        assert len(normalizers) == 3
        assert all(n is normalizers[0] for n in normalizers)
        assert benchmark.shards.should_rotate(20)
        np.testing.assert_allclose(benchmark.shards.load(20)["actions"][:, 0], [-1, 1])


def test_vector_evaluation_uses_env_only_loader_and_closes_workers(monkeypatch):
    from trainer_smoke import make_env

    calls = []

    def make_default(*args):
        calls.append("dataset")
        return make_env()

    monkeypatch.setattr(ogbench_utils, "make_default_env_and_datasets", make_default)
    monkeypatch.setattr(
        ogbench_utils, "make_eval_env", lambda *args, **kwargs: TinyEnv()
    )
    opts = options(env_name="cube-triple-play-singletask-task1-v0")
    with make_benchmark(opts, {}) as benchmark:
        vector = benchmark.make_vector_eval_env(opts)
        assert vector.reset()[0].shape == (2, 4)
        assert vector.step(np.zeros((2, 2)))[0].shape == (2, 4)
        processes = list(vector.processes)
        assert calls == ["dataset"]
    assert all(not process.is_alive() for process in processes)


def test_ocbench_missing_data_is_reported_before_environment_creation(tmp_path):
    with pytest.raises(FileNotFoundError, match="No training shards"):
        make_benchmark(options(ocbench_dataset_dir=str(tmp_path)), {})
