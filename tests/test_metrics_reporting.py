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
    def test_checkpoint_diagnostics_reads_epoch_best_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            checkpoints_dir = run_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            stamped_path = checkpoints_dir / "epoch_0042_best.pt"
            stamped_path.write_bytes(b"fixture")

            with patch(
                "thrawn_nnue.checkpoint.load_checkpoint",
                return_value={
                    "best_checkpoint_metric_name": "score_mae",
                    "best_checkpoint_metric_value": 77.0,
                    "best_checkpoint_positions": 9000,
                    "config": {"batch_size": 1024},
                    "global_step": 42,
                    "positions_seen": 8192,
                },
            ):
                diagnostics = _checkpoint_diagnostics(run_dir)

            self.assertEqual(diagnostics["global_step"], 42)
            self.assertEqual(diagnostics["positions_seen"], 8192)
            self.assertEqual(diagnostics["best_checkpoint_metric_name"], "score_mae")

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
                        "wdl_loss": 1.0,
                        "lr": 0.001,
                    },
                    {
                        "event": "train",
                        "global_step": 2,
                        "positions_seen": 4096,
                        "epoch_index": 0,
                        "loss": 0.7,
                        "wdl_loss": 0.8,
                        "lr": 0.0008,
                    },
                ],
            )
            with patch(
                "thrawn_nnue.metrics._checkpoint_diagnostics",
                return_value={
                    "best_checkpoint_metric_name": None,
                    "best_checkpoint_metric_value": None,
                    "best_checkpoint_positions": None,
                    "config": {
                        "batch_size": 2048,
                        "max_epochs": 2,
                        "epoch_size": 5_000,
                        "start_lambda": 0.1,
                        "end_lambda": 0.1,
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
            self.assertEqual(summary["epoch_size"], 5_000)
            self.assertEqual(summary["latest_epoch_index"], 0)
            self.assertIsNone(summary["latest_validation_positions"])
            self.assertEqual(summary["resume_recommendation"], "insufficient-validation")
            self.assertIn("progress: 4096/10000", render_summary_text(summary))

    def test_validation_summary_prefers_best_mae_positions(self) -> None:
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
                        "wdl_loss": 1.0,
                        "lr": 0.001,
                    },
                    {
                        "event": "train",
                        "global_step": 4,
                        "positions_seen": 4096,
                        "epoch_index": 1,
                        "loss": 0.4,
                        "wdl_loss": 0.5,
                        "lr": 0.0007,
                    },
                    {
                        "event": "validation",
                        "global_step": 2,
                        "positions_seen": 2048,
                        "validation_loss": 0.5,
                        "validation_wdl_loss": 0.6,
                        "score_mae": 120.0,
                        "cp_mae": 120.0,
                        "cp_rmse": 140.0,
                        "cp_corr": 0.51,
                        "wdl_accuracy": 0.55,
                        "teacher_result_disagreement_rate": 0.40,
                        "validation_positions": 1024,
                    },
                    {
                        "event": "validation",
                        "global_step": 4,
                        "positions_seen": 4096,
                        "validation_loss": 0.3,
                        "validation_wdl_loss": 0.35,
                        "score_mae": 80.0,
                        "cp_mae": 80.0,
                        "cp_rmse": 95.0,
                        "cp_corr": 0.72,
                        "wdl_accuracy": 0.62,
                        "teacher_result_disagreement_rate": 0.33,
                        "validation_positions": 1024,
                    },
                ],
            )
            with patch(
                "thrawn_nnue.metrics._checkpoint_diagnostics",
                return_value={
                    "best_checkpoint_metric_name": "score_mae",
                    "best_checkpoint_metric_value": 80.0,
                    "best_checkpoint_positions": 4096,
                    "config": {
                        "batch_size": 1024,
                        "max_epochs": 2,
                        "epoch_size": 4096,
                        "start_lambda": 0.1,
                        "end_lambda": 0.1,
                    },
                    "global_step": 4,
                    "positions_seen": 4096,
                },
            ):
                run = load_metrics_run(run_dir)
                summary = summarize_run(run)
            self.assertEqual(summary["status"], "validated")
            self.assertEqual(summary["best_checkpoint_positions"], 4096)
            self.assertEqual(summary["best_checkpoint_metric_name"], "score_mae")
            self.assertAlmostEqual(summary["best_checkpoint_metric_value"], 80.0)
            self.assertEqual(summary["resume_recommendation"], "continue-latest")
            self.assertTrue(summary["best_is_latest_validation"])
            self.assertEqual(summary["epoch_size"], 4096)
            self.assertEqual(summary["latest_epoch_index"], 1)
            self.assertAlmostEqual(summary["train_validation_gap"], -0.1)
            self.assertAlmostEqual(summary["latest_validation_wdl_accuracy"], 0.62)
            text = render_summary_text(summary)
            self.assertIn("best: score_mae=80.000000 at=4096", text)
            self.assertIn("budget: batch=1024 epoch_size=4096", text)
            self.assertNotIn("Suggestions", text)

    def test_small_validation_gap_continues_latest(self) -> None:
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
                        "loss": 0.3,
                        "wdl_loss": 0.3,
                        "lr": 0.001,
                    },
                    {
                        "event": "validation",
                        "global_step": 1,
                        "positions_seen": 1024,
                        "validation_loss": 0.2,
                        "validation_wdl_loss": 0.2,
                        "score_mae": 10.0,
                        "cp_corr": 0.9,
                    },
                    {
                        "event": "validation",
                        "global_step": 2,
                        "positions_seen": 2048,
                        "validation_loss": 0.20001,
                        "validation_wdl_loss": 0.20001,
                        "score_mae": 10.1,
                        "cp_corr": 0.9,
                    },
                ],
            )
            with patch(
                "thrawn_nnue.metrics._checkpoint_diagnostics",
                return_value={
                    "best_checkpoint_metric_name": None,
                    "best_checkpoint_metric_value": 10.0,
                    "best_checkpoint_positions": 1024,
                    "config": {
                        "batch_size": 1024,
                        "max_epochs": 4,
                        "epoch_size": 1024,
                    },
                    "global_step": 2,
                    "positions_seen": 2048,
                },
            ):
                summary = summarize_run(load_metrics_run(run_dir))

            self.assertEqual(summary["resume_recommendation"], "continue-latest")
            self.assertNotIn("suggestions", summary)

    def test_moving_lambda_adds_loss_caveat(self) -> None:
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
                        "loss": 0.3,
                        "wdl_loss": 0.3,
                        "lambda": 0.74,
                        "lr": 0.001,
                    },
                    {
                        "event": "validation",
                        "global_step": 1,
                        "positions_seen": 1024,
                        "validation_loss": 0.2,
                        "validation_wdl_loss": 0.2,
                        "score_mae": 10.0,
                        "lambda": 0.74,
                        "cp_corr": 0.9,
                    },
                ],
            )
            with patch(
                "thrawn_nnue.metrics._checkpoint_diagnostics",
                return_value={
                    "best_checkpoint_metric_name": None,
                    "best_checkpoint_metric_value": 10.0,
                    "best_checkpoint_positions": 1024,
                    "config": {
                        "batch_size": 1024,
                        "max_epochs": 4,
                        "epoch_size": 1024,
                        "start_lambda": 0.75,
                        "end_lambda": 0.5,
                    },
                    "global_step": 1,
                    "positions_seen": 1024,
                },
            ):
                summary = summarize_run(load_metrics_run(run_dir))
                text = render_summary_text(summary)

            self.assertNotIn("suggestions", summary)
            self.assertIn("lambda=0.750000->0.500000", text)
            self.assertAlmostEqual(summary["latest_validation_lambda"], 0.74)


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
                        "wdl_loss": 1.0,
                        "lr": 0.001,
                    },
                    {
                        "event": "train",
                        "global_step": 2,
                        "positions_seen": 2048,
                        "epoch_index": 1,
                        "loss": 0.7,
                        "wdl_loss": 0.8,
                        "lr": 0.0008,
                    },
                    {
                        "event": "validation",
                        "global_step": 2,
                        "positions_seen": 2048,
                        "validation_loss": 0.5,
                        "validation_wdl_loss": 0.55,
                        "score_mae": 100.0,
                        "cp_mae": 100.0,
                        "cp_rmse": 120.0,
                        "cp_corr": 0.5,
                        "wdl_accuracy": 0.60,
                        "teacher_result_disagreement_rate": 0.30,
                        "validation_positions": 1024,
                    },
                ],
            )
            with patch(
                "thrawn_nnue.metrics._checkpoint_diagnostics",
                return_value={
                    "best_checkpoint_metric_name": None,
                    "best_checkpoint_metric_value": None,
                    "best_checkpoint_positions": None,
                    "config": {
                        "batch_size": 1024,
                        "max_epochs": 2,
                        "epoch_size": 2048,
                    },
                    "global_step": 2,
                    "positions_seen": 2048,
                },
            ):
                run = load_metrics_run(run_dir)
            old_plot = run_dir / "plots" / "train_loss.png"
            old_plot.parent.mkdir(parents=True, exist_ok=True)
            old_plot.write_bytes(b"stale")
            outputs = generate_run_plots(run)
            names = {path.name for path in outputs}
            self.assertEqual(names, {"loss.png", "mae.png"})
            self.assertFalse(old_plot.exists())
            for output in outputs:
                self.assertTrue(output.exists())


if __name__ == "__main__":
    unittest.main()
