from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ModuleNotFoundError:
    torch = None

from thrawn_nnue.config import TrainConfig
from thrawn_nnue.native import write_fixture_binpack
from thrawn_nnue.training import (
    _create_state,
    _maybe_update_best_checkpoint,
    _normalize_teacher_scores,
    _run_validation,
    train_from_config,
)


@unittest.skipUnless(torch is not None, "PyTorch is required for validation training tests")
class ValidationTrainingTests(unittest.TestCase):
    def test_score_normalization_clips_then_scales(self) -> None:
        values = torch.tensor([[-5000.0], [2000.0], [9000.0]], dtype=torch.float32)
        config = TrainConfig.from_dict(
            {
                "train_datasets": ["/tmp/train.binpack"],
                "score_clip": 4000.0,
                "score_scale": 10.0,
            }
        )
        normalized = _normalize_teacher_scores(values, config, torch)
        expected = torch.tensor([[-400.0], [200.0], [400.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(normalized, expected))

    def test_run_validation_does_not_mutate_weights_or_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            train_path = tmp / "train.binpack"
            valid_path = tmp / "valid.binpack"
            write_fixture_binpack(train_path)
            write_fixture_binpack(valid_path)

            config = TrainConfig.from_dict(
                {
                    "train_datasets": [str(train_path)],
                    "validation_datasets": [str(valid_path)],
                    "validation_steps": 1,
                    "output_dir": str(tmp / "run"),
                    "device": "cpu",
                }
            )
            state = _create_state(config)
            model_before = {k: v.detach().clone() for k, v in state.model.state_dict().items()}
            optimizer_before = state.optimizer.state_dict()

            metrics = _run_validation(state)

            self.assertEqual(metrics["event"], "validation")
            self.assertEqual(metrics["validation_batches"], 1)
            for key, before in model_before.items():
                self.assertTrue(torch.equal(before, state.model.state_dict()[key]))
            self.assertEqual(optimizer_before["param_groups"], state.optimizer.state_dict()["param_groups"])

    def test_best_checkpoint_updates_only_for_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            train_path = tmp / "train.binpack"
            valid_path = tmp / "valid.binpack"
            write_fixture_binpack(train_path)
            write_fixture_binpack(valid_path)

            config = TrainConfig.from_dict(
                {
                    "train_datasets": [str(train_path)],
                    "validation_datasets": [str(valid_path)],
                    "validation_steps": 1,
                    "output_dir": str(tmp / "run"),
                    "device": "cpu",
                }
            )
            state = _create_state(config)
            state.global_step = 5
            self.assertTrue(_maybe_update_best_checkpoint(state, 0.25))
            self.assertTrue((state.run_dir / "checkpoints" / "best.pt").exists())
            self.assertFalse(_maybe_update_best_checkpoint(state, 0.30))
            self.assertTrue(_maybe_update_best_checkpoint(state, 0.20))

    def test_train_with_validation_writes_metrics_and_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            train_path = tmp / "train.binpack"
            valid_path = tmp / "valid.binpack"
            write_fixture_binpack(train_path)
            write_fixture_binpack(valid_path)

            config = TrainConfig.from_dict(
                {
                    "train_datasets": [str(train_path)],
                    "validation_datasets": [str(valid_path)],
                    "output_dir": str(tmp / "run"),
                    "device": "cpu",
                    "batch_size": 2,
                    "steps_per_epoch": 2,
                    "max_epochs": 1,
                    "validation_every": 1,
                    "validation_steps": 1,
                    "checkpoint_every": 10,
                    "log_every": 1,
                }
            )
            train_from_config(config)

            best_path = Path(config.output_dir) / "checkpoints" / "best.pt"
            metrics_path = Path(config.output_dir) / "metrics.jsonl"
            self.assertTrue(best_path.exists())
            self.assertTrue(metrics_path.exists())

            records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(record.get("event") == "train" for record in records))
            self.assertTrue(any(record.get("event") == "validation" for record in records))


if __name__ == "__main__":
    unittest.main()
