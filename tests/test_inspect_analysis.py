from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thrawn_nnue.native import inspect_binpack, write_fixture_binpack, _inspect_recommendation, _wdl_scale_diagnostics


class InspectAnalysisTests(unittest.TestCase):
    def test_fixture_inspect_includes_deep_stats_and_recommendation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixture.binpack"
            write_fixture_binpack(path)
            stats = inspect_binpack(path)
            self.assertIn("mean_score", stats)
            self.assertIn("score_percentiles", stats)
            self.assertIn("abs_score_percentiles", stats)
            self.assertIn("ply_percentiles", stats)
            self.assertIn("result_percentages", stats)
            self.assertIn("wdl_scale_diagnostics", stats)
            self.assertIn("recommendation", stats)
            self.assertIn("recommended_wdl_scale", stats["recommendation"])

    def test_recommendation_flags_high_saturation_for_large_score_distribution(self) -> None:
        synthetic = {
            "mean_abs_score": 7920.0,
            "abs_score_percentiles": {"p95": 14000.0, "p99": 20000.0},
        }
        synthetic["wdl_scale_diagnostics"] = _wdl_scale_diagnostics(synthetic)
        recommendation = _inspect_recommendation(synthetic)
        self.assertTrue(recommendation["saturated_at_default_wdl_scale"])
        self.assertGreaterEqual(recommendation["recommended_wdl_scale"], 2000.0)
        self.assertGreater(recommendation["recommended_score_scale"], 1.0)

    def test_recommendation_stays_mild_for_small_score_distribution(self) -> None:
        synthetic = {
            "mean_abs_score": 180.0,
            "abs_score_percentiles": {"p95": 500.0, "p99": 900.0},
        }
        synthetic["wdl_scale_diagnostics"] = _wdl_scale_diagnostics(synthetic)
        recommendation = _inspect_recommendation(synthetic)
        self.assertFalse(recommendation["saturated_at_default_wdl_scale"])
        self.assertEqual(recommendation["recommended_score_clip"], 0.0)
        self.assertEqual(recommendation["recommended_score_scale"], 1.0)


if __name__ == "__main__":
    unittest.main()
