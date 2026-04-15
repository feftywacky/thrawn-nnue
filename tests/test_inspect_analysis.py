from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thrawn_nnue.native import (
    _inspect_recommendation,
    _wdl_scale_diagnostics,
    discover_binpack_files,
    inspect_binpack,
    inspect_binpack_collection,
    write_fixture_binpack,
)


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
            self.assertIn("effective_raw_wdl_scale", stats["recommendation"])

    def test_recommendation_flags_high_saturation_for_large_score_distribution(self) -> None:
        synthetic = {
            "entries_read": 10,
            "mean_abs_score": 7920.0,
            "abs_score_percentiles": {"p95": 14000.0, "p99": 20000.0},
        }
        synthetic["wdl_scale_diagnostics"] = _wdl_scale_diagnostics(synthetic)
        recommendation = _inspect_recommendation(synthetic)
        self.assertTrue(recommendation["saturated_at_default_wdl_scale"])
        self.assertEqual(recommendation["recommended_wdl_scale"], 4000.0)
        self.assertEqual(recommendation["effective_raw_wdl_scale"], 4000.0)
        self.assertFalse(recommendation["teacher_target_collapse_risk"])
        self.assertIn("score clipping", " ".join(recommendation["notes"]))

    def test_recommendation_stays_mild_for_small_score_distribution(self) -> None:
        synthetic = {
            "entries_read": 10,
            "mean_abs_score": 180.0,
            "abs_score_percentiles": {"p95": 500.0, "p99": 900.0},
        }
        synthetic["wdl_scale_diagnostics"] = _wdl_scale_diagnostics(synthetic)
        recommendation = _inspect_recommendation(synthetic)
        self.assertFalse(recommendation["saturated_at_default_wdl_scale"])
        self.assertEqual(recommendation["recommended_score_clip"], 0.0)
        self.assertEqual(recommendation["recommended_wdl_scale"], 410.0)
        self.assertEqual(recommendation["effective_raw_wdl_scale"], 410.0)
        self.assertFalse(recommendation["teacher_target_collapse_risk"])

    def test_recommendation_flags_empty_or_unreadable_dataset(self) -> None:
        synthetic = {
            "entries_read": 0,
            "mean_abs_score": 0.0,
            "abs_score_percentiles": {"p95": 0.0, "p99": 0.0},
        }
        synthetic["wdl_scale_diagnostics"] = _wdl_scale_diagnostics(synthetic)
        recommendation = _inspect_recommendation(synthetic)
        self.assertFalse(recommendation["teacher_target_collapse_risk"])
        self.assertIn("empty or unreadable", " ".join(recommendation["notes"]))

    def test_discover_binpack_files_walks_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "nested"
            nested.mkdir()
            a = root / "a.binpack"
            b = nested / "b.binpack"
            write_fixture_binpack(a)
            write_fixture_binpack(b)

            discovered = discover_binpack_files(root)
            self.assertEqual(discovered, sorted([a.resolve(), b.resolve()]))

    def test_collection_inspect_returns_aggregate_and_per_file_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.binpack"
            second = root / "second.binpack"
            write_fixture_binpack(first)
            write_fixture_binpack(second)

            combined = inspect_binpack_collection([first, second])

            self.assertEqual(combined["file_count"], 2)
            self.assertEqual(len(combined["files"]), 2)
            self.assertEqual(combined["aggregate"]["entries_read"], 6)
            expected_wins = sum(int(item["stats"]["wins"]) for item in combined["files"])
            expected_draws = sum(int(item["stats"]["draws"]) for item in combined["files"])
            expected_losses = sum(int(item["stats"]["losses"]) for item in combined["files"])
            self.assertEqual(combined["aggregate"]["wins"], expected_wins)
            self.assertEqual(combined["aggregate"]["draws"], expected_draws)
            self.assertEqual(combined["aggregate"]["losses"], expected_losses)
            self.assertIn("aggregate_notes", combined["aggregate"])


if __name__ == "__main__":
    unittest.main()
