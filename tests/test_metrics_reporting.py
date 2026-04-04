from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest

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

    def test_validation_summary_prefers_best_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            _write_metrics(
                run_dir / "metrics.jsonl",
                [
                    {"event": "train", "global_step": 1, "loss": 0.9, "eval_loss": 0.8, "result_loss": 1.0, "lr": 0.001},
                    {"event": "validation", "global_step": 2, "validation_loss": 0.5, "validation_eval_loss": 0.4, "validation_result_loss": 0.6},
                    {"event": "validation", "global_step": 4, "validation_loss": 0.3, "validation_eval_loss": 0.25, "validation_result_loss": 0.35},
                ],
            )
            run = load_metrics_run(run_dir)
            summary = summarize_run(run)
            self.assertEqual(summary["status"], "validated")
            self.assertEqual(summary["best_validation_step"], 4)
            self.assertAlmostEqual(summary["best_validation_loss"], 0.3)
            text = render_summary_text(summary)
            self.assertIn("best_validation_step: 4", text)


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


if __name__ == "__main__":
    unittest.main()
