from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thrawn_nnue.calibration import calibrate_scale, _fit_scale_through_origin
from thrawn_nnue.cli import main
from thrawn_nnue.export import ExportedNetwork, _write_export
from thrawn_nnue.native import write_fixture_binpack


def _make_tiny_export(path: Path) -> None:
    num_features = 768
    ft_size = 4
    hidden_size = 2
    output_buckets = 1

    ft_weight = np.zeros((num_features, ft_size), dtype=np.int16)
    for idx in range(num_features):
        ft_weight[idx, 0] = np.int16((idx % 11) - 5)
        ft_weight[idx, 1] = np.int16(((idx * 3) % 13) - 6)
        ft_weight[idx, 2] = np.int16(((idx * 5) % 7) - 3)
        ft_weight[idx, 3] = np.int16(((idx * 7) % 17) - 8)

    exported = ExportedNetwork(
        description="calibration-test",
        num_features=num_features,
        ft_size=ft_size,
        hidden_size=hidden_size,
        output_buckets=output_buckets,
        ft_scale=1024.0,
        dense_scale=256.0,
        wdl_scale=410.0,
        ft_bias=np.zeros(ft_size, dtype=np.int16),
        ft_weight=ft_weight,
        l1_bias=np.zeros(hidden_size, dtype=np.int32),
        l1_weight=np.array(
            [
                [32, -24],
                [-16, 20],
                [12, 8],
                [-10, 14],
                [-8, 12],
                [18, -6],
                [-14, 10],
                [6, -16],
            ],
            dtype=np.int8,
        ),
        out_bias=np.zeros(output_buckets, dtype=np.int32),
        out_weight=np.array([[256], [-128]], dtype=np.int16),
    )

    with path.open("wb") as handle:
        _write_export(handle, exported)


class CalibrationMathTests(unittest.TestCase):
    def test_fit_scale_through_origin_matches_expected_slope(self) -> None:
        raw = np.array([1.0, 2.0, -3.0, 4.0], dtype=np.float64)
        cp = raw * 37.5
        fitted = _fit_scale_through_origin(raw, cp)
        self.assertAlmostEqual(fitted, 37.5, places=9)

    def test_fit_scale_through_origin_rejects_degenerate_outputs(self) -> None:
        raw = np.zeros(4, dtype=np.float64)
        cp = np.array([10.0, -4.0, 3.0, 1.0], dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "all zero"):
            _fit_scale_through_origin(raw, cp)

    def test_quiet_range_fit_excludes_outlier_tail_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nnue_path = root / "tiny.nnue"
            validation_path = root / "fixture.binpack"
            _make_tiny_export(nnue_path)
            write_fixture_binpack(validation_path)

            with patch("thrawn_nnue.calibration.BinpackStream") as stream_cls:
                batch = type("Batch", (), {})()
                batch.score_cp = np.array([100.0, -200.0, 4000.0], dtype=np.float64)
                batch.stm = np.array([1.0, 1.0, 1.0], dtype=np.float64)
                batch.white_counts = np.array([32, 32, 32], dtype=np.int32)
                stream = stream_cls.return_value.__enter__.return_value
                stream.next_batch.side_effect = [batch, None]

                with patch("thrawn_nnue.calibration._ExportEvaluator") as evaluator_cls:
                    evaluator = evaluator_cls.return_value
                    evaluator.eval_batch.return_value = np.array([10.0, -20.0, 10.0], dtype=np.float64)
                    evaluator.eval_fens.return_value = [0.0, 0.0, 0.0]

                    result = calibrate_scale(
                        nnue_path,
                        validation_path,
                        max_positions=3,
                        batch_size=3,
                        threads=1,
                        fit_window_cp=600.0,
                        min_fit_positions=1,
                    )

        self.assertAlmostEqual(float(result["cp_per_raw"]), 10.0, places=8)
        self.assertNotAlmostEqual(float(result["global_cp_per_raw"]), 10.0, places=3)
        self.assertEqual(result["fit_filter"]["within_window"], 2)

    def test_calibration_rejects_zero_slope_when_teacher_cp_is_already_stm_native(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nnue_path = root / "tiny.nnue"
            validation_path = root / "fixture.binpack"
            _make_tiny_export(nnue_path)
            write_fixture_binpack(validation_path)

            with patch("thrawn_nnue.calibration.BinpackStream") as stream_cls:
                batch = type("Batch", (), {})()
                batch.score_cp = np.array([100.0, 100.0], dtype=np.float64)
                batch.stm = np.array([1.0, 0.0], dtype=np.float64)
                batch.white_counts = np.array([32, 32], dtype=np.int32)
                stream = stream_cls.return_value.__enter__.return_value
                stream.next_batch.side_effect = [batch, None]

                with patch("thrawn_nnue.calibration._ExportEvaluator") as evaluator_cls:
                    evaluator = evaluator_cls.return_value
                    evaluator.eval_batch.return_value = np.array([10.0, -10.0], dtype=np.float64)
                    evaluator.eval_fens.return_value = [0.0, 0.0, 0.0]

                    with self.assertRaisesRegex(ValueError, "approximately zero"):
                        calibrate_scale(
                            nnue_path,
                            validation_path,
                            max_positions=2,
                            batch_size=2,
                            threads=1,
                            fit_window_cp=600.0,
                            min_fit_positions=1,
                        )

    def test_quiet_range_fit_requires_minimum_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nnue_path = root / "tiny.nnue"
            validation_path = root / "fixture.binpack"
            _make_tiny_export(nnue_path)
            write_fixture_binpack(validation_path)

            with patch("thrawn_nnue.calibration.BinpackStream") as stream_cls:
                batch = type("Batch", (), {})()
                batch.score_cp = np.array([2000.0, -1900.0, 1700.0], dtype=np.float64)
                batch.stm = np.array([1.0, 1.0, 1.0], dtype=np.float64)
                batch.white_counts = np.array([32, 32, 32], dtype=np.int32)
                stream = stream_cls.return_value.__enter__.return_value
                stream.next_batch.side_effect = [batch, None]

                with patch("thrawn_nnue.calibration._ExportEvaluator") as evaluator_cls:
                    evaluator = evaluator_cls.return_value
                    evaluator.eval_batch.return_value = np.array([2.0, 3.0, 4.0], dtype=np.float64)
                    evaluator.eval_fens.return_value = [0.0, 0.0, 0.0]

                    with self.assertRaisesRegex(ValueError, "Not enough positions in quiet fit window"):
                        calibrate_scale(
                            nnue_path,
                            validation_path,
                            max_positions=3,
                            batch_size=3,
                            threads=1,
                            fit_window_cp=600.0,
                            min_fit_positions=1,
                        )


class CalibrationIntegrationTests(unittest.TestCase):
    def test_calibrate_scale_runs_end_to_end_and_emits_hardcoded_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nnue_path = root / "tiny.nnue"
            validation_path = root / "fixture.binpack"
            _make_tiny_export(nnue_path)
            write_fixture_binpack(validation_path)

            result = calibrate_scale(
                nnue_path,
                validation_path,
                max_positions=3,
                batch_size=2,
                threads=1,
                fit_window_cp=20_000.0,
                min_fit_positions=1,
            )

            self.assertEqual(result["positions_used"], 3)
            self.assertIn("cp_per_raw", result)
            self.assertIn("raw_per_cp", result)
            self.assertTrue(np.isfinite(float(result["cp_per_raw"])))
            self.assertIn("fit_metrics", result)
            self.assertEqual(result["teacher_perspective"], "stm")
            self.assertEqual(result["fit_scope"], "quiet_range")
            self.assertIn("fit_filter", result)
            self.assertGreater(result["fit_filter"]["used_for_fit"], 0)
            self.assertIn("global_cp_per_raw", result)
            self.assertIn("global_fit_metrics", result)
            self.assertIn("normalization_constant", result)
            self.assertIn("normalization_constant_rounded", result)
            self.assertEqual(
                float(result["normalization_constant"]),
                float(100.0 / float(result["cp_per_raw"])),
            )
            self.assertEqual(
                int(result["normalization_constant_rounded"]),
                int(round(float(result["normalization_constant"]))),
            )
            self.assertIn("sanity_flags", result)
            self.assertIn("hardcoded_positions", result)
            self.assertIn("symmetry_checks", result)
            self.assertEqual(len(result["symmetry_checks"]), 3)
            hardcoded = result["hardcoded_positions"]
            self.assertEqual(len(hardcoded), 3)
            self.assertEqual(
                [item["name"] for item in hardcoded],
                ["starting_position", "white_up_pawn", "white_up_knight"],
            )


class CalibrationCliTests(unittest.TestCase):
    def test_calibrate_scale_cli_outputs_json(self) -> None:
        payload = {
            "positions_used": 42,
            "teacher_perspective": "stm",
            "fit_scope": "quiet_range",
            "fit_filter": {"total_sampled": 42, "within_window": 40, "used_for_fit": 40, "window_cp": 600.0},
            "cp_per_raw": 19.5,
            "raw_per_cp": 0.05128205128,
            "fit_metrics": {"mae_cp": 1.1, "rmse_cp": 1.3, "corr": 0.77},
            "global_cp_per_raw": 18.2,
            "global_fit_metrics": {"mae_cp": 2.1, "rmse_cp": 2.3, "corr": 0.55},
            "normalization_constant": 5.128205128,
            "normalization_constant_rounded": 5,
            "sanity_flags": [],
            "symmetry_checks": [
                {
                    "name": "starting_position",
                    "fen_white_stm": "xw",
                    "fen_black_stm": "xb",
                    "white_raw": 0.0,
                    "black_raw": 0.0,
                    "white_scaled_cp": 0.0,
                    "black_scaled_cp": 0.0,
                    "sum_scaled_cp": 0.0,
                }
            ],
            "hardcoded_positions": [
                {"name": "starting_position", "fen": "x", "raw": 0.0, "scaled_cp": 0.0},
                {"name": "white_up_pawn", "fen": "y", "raw": 0.1, "scaled_cp": 1.95},
                {"name": "white_up_knight", "fen": "z", "raw": 0.2, "scaled_cp": 3.9},
            ],
        }

        argv = sys.argv
        stdout = io.StringIO()
        try:
            sys.argv = [
                "thrawn-nnue",
                "calibrate-scale",
                "--nnue",
                "/tmp/model.nnue",
                "--validation-path",
                "/tmp/validation.binpack",
                "--max-positions",
                "123",
                "--batch-size",
                "64",
                "--threads",
                "2",
                "--fit-window-cp",
                "450",
                "--min-fit-positions",
                "333",
            ]
            with redirect_stdout(stdout), patch("thrawn_nnue.cli.calibrate_scale", return_value=payload) as patched:
                main()
                patched.assert_called_once_with(
                    "/tmp/model.nnue",
                    "/tmp/validation.binpack",
                    max_positions=123,
                    batch_size=64,
                    threads=2,
                    fit_window_cp=450.0,
                    min_fit_positions=333,
                )
        finally:
            sys.argv = argv

        decoded = json.loads(stdout.getvalue())
        self.assertEqual(decoded["positions_used"], 42)
        self.assertEqual(decoded["teacher_perspective"], "stm")
        self.assertIn("fit_metrics", decoded)
        self.assertIn("global_fit_metrics", decoded)
        self.assertIn("fit_filter", decoded)
        self.assertIn("hardcoded_positions", decoded)
        self.assertIn("symmetry_checks", decoded)

if __name__ == "__main__":
    unittest.main()
