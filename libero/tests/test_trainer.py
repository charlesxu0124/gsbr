from pathlib import Path
import json
from omegaconf import OmegaConf
from src.train import run_training
from test_cached_collection import _FakeInnerPolicy, _OneStepEnv, _make_policy


class Env(_OneStepEnv):
    def close(self):
        pass


def test_gsbr_train_checkpoint_resume_and_eval(tmp_path):
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1] / "src/cfgs/gsbr_t14.yaml")
    cfg.results_dir = str(tmp_path)
    cfg.exp_name = "smoke"
    cfg.train.max_num_trajs = 2
    cfg.train.learning_starts = 0
    cfg.train.grad_steps_per_env_step = 1
    cfg.train.save_every = 1
    cfg.eval.eval_every = 1
    cfg.eval.eval_num_trajs = 1
    cfg.eval.eval_num_videos = 0
    run_training(cfg, Env(), _make_policy(_FakeInnerPolicy(), "gsbr"))
    run = tmp_path / "CachedPolicy" / "smoke"
    assert (run / "checkpoints/stats.pkl").exists()
    metrics = [
        json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()
    ]
    assert any("training/actor/total_loss" in row for row in metrics)
    # At the completed budget, resume restores the cache/learner and only evaluates.
    run_training(cfg, Env(), _make_policy(_FakeInnerPolicy(), "gsbr"))
