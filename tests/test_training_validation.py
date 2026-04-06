from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ModuleNotFoundError:
    torch = None

from thrawn_nnue.checkpoint import load_checkpoint
from thrawn_nnue.config import TrainConfig
from thrawn_nnue.native import inspect_binpack, write_fixture_binpack
from thrawn_nnue.training import (
    _create_state,
    _maybe_update_best_checkpoint,
    _normalize_teacher_scores,
    _run_validation,
    resume_training,
    train_from_config,
)


@unittest.skipUnless(torch is not None, "PyTorch is required for validation training tests")
class ValidationTrainingTests(unittest.TestCase):
    def test_score_normalization_clips_then_scales(self) -> None:
        values = torch.tensor([[-5000.0], [2000.0], [9000.0]], dtype=torch.float32)
        config = TrainConfig.from_dict(
            {
                "train_datasets": ["/tmp/train.binpack"],
                "total_train_positions": 10_000,
                "superbatch_positions": 1_000,
                "score_clip": 4000.0,
                "score_scale": 10.0,
            }
        )
        normalized = _normalize_teacher_scores(values, config, torch)
        expected = torch.tensor([[-400.0], [200.0], [400.0]], dtype=torch.float32)
        self.assertTrue(torch.equal(normalized, expected))

    def test_run_validation_does_not_mutate_weights_or_optimizer_and_reports_new_metrics(self) -> None:
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
                    "validation_positions": 2,
                    "total_train_positions": 10_000,
                    "superbatch_positions": 1_000,
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
            self.assertEqual(metrics["validation_positions"], 2)
            self.assertIn("validation_teacher_loss", metrics)
            self.assertIn("wdl_accuracy", metrics)
            self.assertIn("teacher_result_disagreement_rate", metrics)
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
                    "validation_positions": 2,
                    "total_train_positions": 10_000,
                    "superbatch_positions": 1_000,
                    "output_dir": str(tmp / "run"),
                    "device": "cpu",
                }
            )
            state = _create_state(config)
            state.global_step = 5
            state.positions_seen = 4_096
            self.assertTrue(_maybe_update_best_checkpoint(state, 0.25))
            self.assertTrue((state.run_dir / "checkpoints" / "best.pt").exists())
            self.assertEqual(state.best_validation_positions, 4_096)
            self.assertFalse(_maybe_update_best_checkpoint(state, 0.30))
            state.positions_seen = 8_192
            self.assertTrue(_maybe_update_best_checkpoint(state, 0.20))
            self.assertEqual(state.best_validation_positions, 8_192)

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
                    "total_train_positions": 4,
                    "superbatch_positions": 4,
                    "validation_interval_positions": 2,
                    "validation_positions": 2,
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
            validation_records = [record for record in records if record.get("event") == "validation"]
            self.assertTrue(validation_records)
            self.assertTrue(all("positions_seen" in record for record in records))
            self.assertTrue(all("validation_teacher_loss" in record for record in validation_records))

    def test_validation_runs_on_superbatch_boundary_when_interval_is_zero(self) -> None:
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
                    "total_train_positions": 4,
                    "superbatch_positions": 2,
                    "validation_interval_positions": 0,
                    "validation_positions": 2,
                    "checkpoint_every": 10,
                    "log_every": 1,
                }
            )
            train_from_config(config)

            metrics_path = Path(config.output_dir) / "metrics.jsonl"
            records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
            validation_records = [record for record in records if record.get("event") == "validation"]
            self.assertEqual(len(validation_records), 2)
            self.assertEqual([record["positions_seen"] for record in validation_records], [2, 4])

    def test_zero_validation_positions_consumes_full_validation_stream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            train_path = tmp / "train.binpack"
            valid_path = tmp / "valid.binpack"
            write_fixture_binpack(train_path)
            write_fixture_binpack(valid_path)

            valid_entries = int(inspect_binpack(valid_path)["entries_read"])
            config = TrainConfig.from_dict(
                {
                    "train_datasets": [str(train_path)],
                    "validation_datasets": [str(valid_path)],
                    "validation_positions": 0,
                    "total_train_positions": 10_000,
                    "superbatch_positions": 1_000,
                    "output_dir": str(tmp / "run"),
                    "device": "cpu",
                    "batch_size": 2,
                }
            )

            state = _create_state(config)
            metrics = _run_validation(state)

            self.assertEqual(metrics["validation_batches"], math.ceil(valid_entries / 2))
            self.assertEqual(metrics["validation_positions"], valid_entries)

    def test_validation_position_budget_is_exact_for_partial_last_batch(self) -> None:
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
                    "validation_positions": 1,
                    "total_train_positions": 10_000,
                    "superbatch_positions": 1_000,
                    "output_dir": str(tmp / "run"),
                    "device": "cpu",
                    "batch_size": 2,
                }
            )

            state = _create_state(config)
            metrics = _run_validation(state)

            self.assertEqual(metrics["validation_batches"], 1)
            self.assertEqual(metrics["validation_positions"], 1)

    def test_training_stops_on_position_budget_and_resume_continues_from_checkpoint(self) -> None:
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
                    "total_train_positions": 5,
                    "superbatch_positions": 3,
                    "validation_interval_positions": 10,
                    "validation_positions": 2,
                    "checkpoint_every": 1,
                    "log_every": 1,
                }
            )

            final_checkpoint = train_from_config(config)
            mid_checkpoint = Path(config.output_dir) / "checkpoints" / "step_00000001.pt"
            self.assertTrue(mid_checkpoint.exists())
            resumed_checkpoint = resume_training(mid_checkpoint, console_mode="text")

            final_payload = load_checkpoint(final_checkpoint)
            resumed_payload = load_checkpoint(resumed_checkpoint)
            self.assertEqual(final_payload["positions_seen"], config.total_train_positions)
            self.assertEqual(resumed_payload["positions_seen"], config.total_train_positions)
            self.assertEqual(resumed_payload["global_step"], final_payload["global_step"])

    def test_final_validation_runs_even_if_interval_is_not_reached(self) -> None:
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
                    "total_train_positions": 2,
                    "superbatch_positions": 10,
                    "validation_interval_positions": 10,
                    "validation_positions": 2,
                    "checkpoint_every": 10,
                    "log_every": 1,
                }
            )
            train_from_config(config)

            metrics_path = Path(config.output_dir) / "metrics.jsonl"
            records = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
            validation_records = [record for record in records if record.get("event") == "validation"]
            self.assertEqual(len(validation_records), 1)
            self.assertEqual(validation_records[0]["positions_seen"], 2)

    def test_text_console_mode_prints_position_budget_progress_without_raw_json(self) -> None:
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
                    "console_mode": "text",
                    "batch_size": 2,
                    "total_train_positions": 4,
                    "superbatch_positions": 4,
                    "validation_interval_positions": 2,
                    "validation_positions": 2,
                    "checkpoint_every": 10,
                    "log_every": 1,
                }
            )

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                train_from_config(config)

            output = stdout.getvalue()
            self.assertIn("training start:", output)
            self.assertIn("positions=", output)
            self.assertIn("validation done:", output)
            self.assertNotIn('{"event": "train"', output)
            self.assertNotIn('{"event": "validation"', output)


if __name__ == "__main__":
    unittest.main()
