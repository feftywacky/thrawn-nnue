from __future__ import annotations

import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import matplotlib  # noqa: F401
except ModuleNotFoundError:
    matplotlib = None

from thrawn_nnue.metrics import _checkpoint_diagnostics, generate_run_plots, load_metrics_run, render_summary_text, summarize_run


def _write_metrics(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


class MetricsSummaryTests(unittest.TestCase):
    def test_checkpoint_diagnostics_falls_back_to_stamped_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            checkpoints_dir = run_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            stamped_path = checkpoints_dir / "best_step_00000042.pt"
            stamped_path.write_bytes(b"fixture")

            with patch(
                "thrawn_nnue.checkpoint.load_checkpoint",
                return_value={
                    "best_validation_loss": 0.123,
                    "best_validation_positions": 8192,
                    "config": {"batch_size": 1024},
                    "global_step": 42,
                    "positions_seen": 8192,
                },
            ):
                diagnostics = _checkpoint_diagnostics(run_dir)

            self.assertEqual(diagnostics["best_validation_loss"], 0.123)
            self.assertEqual(diagnostics["global_step"], 42)
            self.assertEqual(diagnostics["positions_seen"], 8192)

    def test_load_and_summarize_train_only_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_metrics(
                run_dir / "metrics.jsonl",
                [
                    {
                        "event": "train",
                        "global_step": 1,
                        "positions_seen": 2048,
                        "epoch_index": 0,
                        "loss": 0.9,
                        "cp_loss": 0.8,
                        "wdl_loss": 1.0,
                        "lr": 0.001,
                    },
                    {
                        "event": "train",
                        "global_step": 2,
                        "positions_seen": 4096,
                        "epoch_index": 0,
                        "loss": 0.7,
                        "cp_loss": 0.6,
                        "wdl_loss": 0.8,
                        "lr": 0.0008,
                    },
                ],
            )
            with patch(
                "thrawn_nnue.metrics._checkpoint_diagnostics",
                return_value={
                    "best_validation_loss": None,
                    "best_validation_positions": None,
                    "config": {
                        "batch_size": 2048,
                        "total_train_positions": 10_000,
                        "epoch_positions": 5_000,
                        "validation_interval_positions": 2_500,
                        "score_clip": 4000.0,
                        "cp_loss_beta": 128.0,
                        "wdl_lambda": 0.1,
                    },
                    "global_step": 2,
                    "positions_seen": 4096,
                },
            ):
                run = load_metrics_run(run_dir)
                summary = summarize_run(run)
            self.assertEqual(summary["status"], "train-only")
            self.assertEqual(summary["train_records"], 2)
            self.assertEqual(summary["validation_records"], 0)
            self.assertEqual(summary["positions_seen"], 4096)
            self.assertEqual(summary["epoch_positions"], 5_000)
            self.assertEqual(summary["latest_epoch_index"], 0)
            self.assertIsNone(summary["latest_validation_positions"])
            self.assertEqual(summary["resume_recommendation"], "insufficient-validation")
            self.assertIn("configured_total_positions", render_summary_text(summary))

    def test_validation_summary_prefers_best_validation_positions_and_material_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_metrics(
                run_dir / "metrics.jsonl",
                [
                    {
                        "event": "train",
                        "global_step": 1,
                        "positions_seen": 1024,
                        "epoch_index": 0,
                        "loss": 0.9,
                        "cp_loss": 0.8,
                        "wdl_loss": 1.0,
                        "lr": 0.001,
                    },
                    {
                        "event": "train",
                        "global_step": 4,
                        "positions_seen": 4096,
                        "epoch_index": 1,
                        "loss": 0.4,
                        "cp_loss": 0.2,
                        "wdl_loss": 0.5,
                        "lr": 0.0007,
                    },
                    {
                        "event": "validation",
                        "global_step": 2,
                        "positions_seen": 2048,
                        "validation_loss": 0.5,
                        "validation_cp_loss": 0.4,
                        "validation_wdl_loss": 0.6,
                        "cp_mae": 120.0,
                        "cp_rmse": 140.0,
                        "cp_corr": 0.51,
                        "wdl_accuracy": 0.55,
                        "teacher_result_disagreement_rate": 0.40,
                        "validation_positions": 1024,
                        "material_sanity": {"ordering_ok": False},
                        "material_ordering_ok": False,
                    },
                    {
                        "event": "validation",
                        "global_step": 4,
                        "positions_seen": 4096,
                        "validation_loss": 0.3,
                        "validation_cp_loss": 0.25,
                        "validation_wdl_loss": 0.35,
                        "cp_mae": 80.0,
                        "cp_rmse": 95.0,
                        "cp_corr": 0.72,
                        "wdl_accuracy": 0.62,
                        "teacher_result_disagreement_rate": 0.33,
                        "validation_positions": 1024,
                        "material_sanity": {"ordering_ok": True},
                        "material_ordering_ok": True,
                    },
                ],
            )
            with patch(
                "thrawn_nnue.metrics._checkpoint_diagnostics",
                return_value={
                    "best_validation_loss": 0.3,
                    "best_validation_positions": 4096,
                    "config": {
                        "batch_size": 1024,
                        "total_train_positions": 8192,
                        "epoch_positions": 4096,
                        "validation_interval_positions": 2048,
                        "score_clip": 4000.0,
                        "cp_loss_beta": 128.0,
                        "wdl_lambda": 0.1,
                    },
                    "global_step": 4,
                    "positions_seen": 4096,
                },
            ):
                run = load_metrics_run(run_dir)
                summary = summarize_run(run)
            self.assertEqual(summary["status"], "validated")
            self.assertEqual(summary["best_validation_positions"], 4096)
            self.assertAlmostEqual(summary["best_validation_loss"], 0.3)
            self.assertEqual(summary["resume_recommendation"], "continue-latest")
            self.assertTrue(summary["best_is_latest_validation"])
            self.assertEqual(summary["epoch_positions"], 4096)
            self.assertEqual(summary["latest_epoch_index"], 1)
            self.assertAlmostEqual(summary["train_validation_gap"], -0.1)
            self.assertAlmostEqual(summary["latest_validation_wdl_accuracy"], 0.62)
            self.assertTrue(summary["latest_material_ordering_ok"])
            text = render_summary_text(summary)
            self.assertIn("best_validation_positions: 4096", text)
            self.assertIn("epoch_positions: 4096", text)
            self.assertIn("Suggestions", text)


@unittest.skipUnless(matplotlib is not None, "matplotlib is required for metrics plotting tests")
class MetricsPlotTests(unittest.TestCase):
    def test_generate_plots_for_train_and_validation_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_metrics(
                run_dir / "metrics.jsonl",
                [
                    {
                        "event": "train",
                        "global_step": 1,
                        "positions_seen": 1024,
                        "epoch_index": 0,
                        "loss": 0.9,
                        "cp_loss": 0.8,
                        "wdl_loss": 1.0,
                        "lr": 0.001,
                    },
                    {
                        "event": "train",
                        "global_step": 2,
                        "positions_seen": 2048,
                        "epoch_index": 1,
                        "loss": 0.7,
                        "cp_loss": 0.6,
                        "wdl_loss": 0.8,
                        "lr": 0.0008,
                    },
                    {
                        "event": "validation",
                        "global_step": 2,
                        "positions_seen": 2048,
                        "validation_loss": 0.5,
                        "validation_cp_loss": 0.45,
                        "validation_wdl_loss": 0.55,
                        "cp_mae": 100.0,
                        "cp_rmse": 120.0,
                        "cp_corr": 0.5,
                        "wdl_accuracy": 0.60,
                        "teacher_result_disagreement_rate": 0.30,
                        "validation_positions": 1024,
                        "material_sanity": {"ordering_ok": True},
                        "material_ordering_ok": True,
                    },
                ],
            )
            with patch(
                "thrawn_nnue.metrics._checkpoint_diagnostics",
                return_value={
                    "best_validation_loss": None,
                    "best_validation_positions": None,
                    "config": {
                        "batch_size": 1024,
                        "total_train_positions": 4096,
                        "epoch_positions": 2048,
                        "validation_interval_positions": 2048,
                    },
                    "global_step": 2,
                    "positions_seen": 2048,
                },
            ):
                run = load_metrics_run(run_dir)
            outputs = generate_run_plots(run)
            names = {path.name for path in outputs}
            self.assertIn("train_loss.png", names)
            self.assertIn("validation_loss.png", names)
            self.assertIn("lr.png", names)
            self.assertIn("loss_overview.png", names)
            for output in outputs:
                self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
