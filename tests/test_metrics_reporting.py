from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import matplotlib  # noqa: F401
except ModuleNotFoundError:
    matplotlib = None

from thrawn_nnue.cli import main
from thrawn_nnue.metrics import generate_run_plots, load_metrics_run, render_summary_text, summarize_run


def _write_metrics(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


class MetricsSummaryTests(unittest.TestCase):
    def test_load_and_summarize_train_only_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_metrics(
                run_dir / "metrics.jsonl",
                [
                    {"event": "train", "global_step": 1, "loss": 0.9, "eval_loss": 0.8, "result_loss": 1.0, "lr": 0.001},
                    {"event": "train", "global_step": 2, "loss": 0.7, "eval_loss": 0.6, "result_loss": 0.8, "lr": 0.0008},
                ],
            )
            run = load_metrics_run(run_dir)
            summary = summarize_run(run)
            self.assertEqual(summary["status"], "train-only")
            self.assertEqual(summary["train_records"], 2)
            self.assertEqual(summary["validation_records"], 0)
            self.assertEqual(summary["latest_train_step"], 2)
            self.assertIsNone(summary["latest_validation_step"])
            self.assertEqual(summary["resume_recommendation"], "insufficient-validation")
            self.assertIn("Run Budget", render_summary_text(summary))

    def test_validation_summary_prefers_best_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_metrics(
                run_dir / "metrics.jsonl",
                [
                    {"event": "train", "global_step": 1, "loss": 0.9, "eval_loss": 0.8, "result_loss": 1.0, "lr": 0.001},
                    {"event": "train", "global_step": 4, "loss": 0.4, "eval_loss": 0.2, "result_loss": 0.5, "lr": 0.0007},
                    {"event": "validation", "global_step": 2, "validation_loss": 0.5, "validation_eval_loss": 0.4, "validation_result_loss": 0.6},
                    {"event": "validation", "global_step": 4, "validation_loss": 0.3, "validation_eval_loss": 0.25, "validation_result_loss": 0.35},
                ],
            )
            with patch(
                "thrawn_nnue.metrics._checkpoint_diagnostics",
                return_value={
                    "best_validation_loss": 0.3,
                    "best_validation_step": 4,
                    "config": {
                        "batch_size": 1024,
                        "max_epochs": 2,
                        "steps_per_epoch": 4,
                        "score_clip": 16000.0,
                        "score_scale": 1.0,
                        "wdl_scale": 8000.0,
                        "result_lambda": 0.8,
                    },
                    "global_step": 4,
                },
            ):
                run = load_metrics_run(run_dir)
                summary = summarize_run(run)
            self.assertEqual(summary["status"], "validated")
            self.assertEqual(summary["best_validation_step"], 4)
            self.assertAlmostEqual(summary["best_validation_loss"], 0.3)
            self.assertEqual(summary["resume_recommendation"], "continue-latest")
            self.assertTrue(summary["best_is_latest_validation"])
            self.assertEqual(summary["configured_total_steps"], 8)
            self.assertEqual(summary["samples_seen"], 4096)
            self.assertAlmostEqual(summary["train_validation_gap"], -0.1)
            text = render_summary_text(summary)
            self.assertIn("best_validation_step: 4", text)
            self.assertIn("Suggestions", text)

    def test_summary_flags_export_best_and_eval_signal_collapse(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_metrics(
                run_dir / "metrics.jsonl",
                [
                    {"event": "train", "global_step": 100, "loss": 0.30, "eval_loss": 0.001, "result_loss": 0.100, "lr": 0.0010},
                    {"event": "train", "global_step": 200, "loss": 0.28, "eval_loss": 0.001, "result_loss": 0.095, "lr": 0.0000},
                    {"event": "validation", "global_step": 100, "validation_loss": 0.25, "validation_eval_loss": 0.001, "validation_result_loss": 0.090},
                    {"event": "validation", "global_step": 200, "validation_loss": 0.27, "validation_eval_loss": 0.001, "validation_result_loss": 0.100},
                ],
            )
            with patch(
                "thrawn_nnue.metrics._checkpoint_diagnostics",
                return_value={
                    "best_validation_loss": 0.25,
                    "best_validation_step": 100,
                    "config": {
                        "batch_size": 2048,
                        "max_epochs": 10,
                        "steps_per_epoch": 20,
                        "score_clip": 16000.0,
                        "score_scale": 1.0,
                        "wdl_scale": 8000.0,
                        "result_lambda": 0.8,
                    },
                    "global_step": 100,
                },
            ):
                run = load_metrics_run(run_dir)
                summary = summarize_run(run)
            self.assertEqual(summary["resume_recommendation"], "export-best")
            self.assertEqual(summary["steps_since_best"], 100)
            self.assertTrue(summary["eval_signal_collapsed"])
            self.assertTrue(summary["scheduler_exhausted"])
            self.assertIn("revisit score normalization", " ".join(summary["suggestions"]))


@unittest.skipUnless(matplotlib is not None, "matplotlib is required for metrics plotting tests")
class MetricsPlotTests(unittest.TestCase):
    def test_generate_plots_for_train_and_validation_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_metrics(
                run_dir / "metrics.jsonl",
                [
                    {"event": "train", "global_step": 1, "loss": 0.9, "eval_loss": 0.8, "result_loss": 1.0, "lr": 0.001},
                    {"event": "train", "global_step": 2, "loss": 0.7, "eval_loss": 0.6, "result_loss": 0.8, "lr": 0.0008},
                    {"event": "validation", "global_step": 2, "validation_loss": 0.5, "validation_eval_loss": 0.45, "validation_result_loss": 0.55},
                ],
            )
            run = load_metrics_run(run_dir)
            outputs = generate_run_plots(run)
            names = {path.name for path in outputs}
            self.assertIn("train_loss.png", names)
            self.assertIn("validation_loss.png", names)
            self.assertIn("lr.png", names)
            self.assertIn("loss_overview.png", names)
            for output in outputs:
                self.assertTrue(output.exists())


class MetricsCliTests(unittest.TestCase):
    def test_metrics_cli_prints_summary_or_fails_clearly_without_matplotlib(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_metrics(
                run_dir / "metrics.jsonl",
                [
                    {"event": "train", "global_step": 1, "loss": 0.9, "eval_loss": 0.8, "result_loss": 1.0, "lr": 0.001},
                ],
            )
            stdout = io.StringIO()
            argv = sys.argv
            try:
                sys.argv = ["thrawn-nnue", "metrics", "--run-dir", str(run_dir)]
                with redirect_stdout(stdout):
                    if matplotlib is None:
                        with self.assertRaises(RuntimeError):
                            main()
                    else:
                        main()
            finally:
                sys.argv = argv
            output = stdout.getvalue()
            if matplotlib is not None:
                self.assertIn("run_dir:", output)
                self.assertIn("train_records: 1", output)
                self.assertIn("Run Budget", output)

    def test_metrics_cli_json_output_includes_enriched_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_metrics(
                run_dir / "metrics.jsonl",
                [
                    {"event": "train", "global_step": 1, "loss": 0.9, "eval_loss": 0.8, "result_loss": 1.0, "lr": 0.001},
                ],
            )
            stdout = io.StringIO()
            argv = sys.argv
            try:
                sys.argv = ["thrawn-nnue", "metrics", "--run-dir", str(run_dir), "--json"]
                with redirect_stdout(stdout):
                    if matplotlib is None:
                        with self.assertRaises(RuntimeError):
                            main()
                    else:
                        main()
            finally:
                sys.argv = argv
            if matplotlib is not None:
                payload = json.loads(stdout.getvalue())
                self.assertIn("summary", payload)
                self.assertIn("samples_seen", payload["summary"])
                self.assertIn("plots", payload)

    def test_train_cli_console_mode_override_takes_precedence(self) -> None:
        argv = sys.argv
        try:
            sys.argv = ["thrawn-nnue", "train", "--config", "configs/default.toml", "--console-mode", "text"]
            with (
                redirect_stdout(io.StringIO()),
                patch("thrawn_nnue.cli.load_config") as load_config,
                patch("thrawn_nnue.cli.train_from_config") as train_from_config,
            ):
                config = object()
                load_config.return_value = config
                train_from_config.return_value = Path("/tmp/final.pt")
                main()
                train_from_config.assert_called_once_with(config, console_mode="text")
        finally:
            sys.argv = argv

    def test_resume_cli_console_mode_override_takes_precedence(self) -> None:
        argv = sys.argv
        try:
            sys.argv = ["thrawn-nnue", "resume", "--checkpoint", "/tmp/run.pt", "--console-mode", "text"]
            with redirect_stdout(io.StringIO()), patch("thrawn_nnue.cli.resume_training") as resume_training:
                resume_training.return_value = Path("/tmp/final.pt")
                main()
                resume_training.assert_called_once_with("/tmp/run.pt", console_mode="text")
        finally:
            sys.argv = argv


if __name__ == "__main__":
    unittest.main()
