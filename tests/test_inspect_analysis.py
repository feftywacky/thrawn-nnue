from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thrawn_nnue.native import (
    discover_binpack_files,
    inspect_binpack,
    inspect_binpack_collection,
    write_fixture_binpack,
)


class InspectAnalysisTests(unittest.TestCase):
    def test_fixture_inspect_includes_raw_format_and_wdl_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixture.binpack"
            write_fixture_binpack(path)
            stats = inspect_binpack(path)
            self.assertIn("format", stats)
            self.assertEqual(stats["format"]["container"]["chunk_magic"], "BINP")
            self.assertIn("file", stats)
            self.assertIn("filters", stats)
            self.assertEqual(stats["entries_seen"], 3)
            self.assertEqual(stats["entries_skipped_by_filters"], 0)
            self.assertEqual(stats["root_entries"], 3)
            self.assertEqual(stats["continuation_entries"], 0)
            self.assertIn("mean_score", stats)
            self.assertIn("mean_raw_score", stats)
            self.assertIn("score_percentiles", stats)
            self.assertIn("raw_score_percentiles", stats)
            self.assertIn("abs_score_percentiles", stats)
            self.assertIn("ply_percentiles", stats)
            self.assertIn("result_percentages", stats)
            self.assertIn("score_buckets", stats)
            self.assertIn("phase_buckets", stats)
            self.assertIn("move_types", stats)
            self.assertIn("position_flags", stats)
            self.assertIn("material", stats)
            self.assertIn("wdl", stats)
            self.assertNotIn("recommendation", stats)
            self.assertFalse(any(key.startswith("recommended_") for key in stats))

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

            combined = inspect_binpack_collection([first, second], jobs=2)

            self.assertEqual(combined["file_count"], 2)
            self.assertEqual(
                combined["filters"],
                {
                    "skip_capture_positions": False,
                    "skip_wdl_score_mismatch": False,
                    "max_abs_score": 0.0,
                    "sample_entries": 0,
                },
            )
            self.assertEqual(len(combined["files"]), 2)
            self.assertEqual(
                [item["path"] for item in combined["files"]],
                [str(first.resolve()), str(second.resolve())],
            )
            self.assertEqual(combined["aggregate"]["entries_read"], 6)
            expected_wins = sum(int(item["stats"]["wins"]) for item in combined["files"])
            expected_draws = sum(int(item["stats"]["draws"]) for item in combined["files"])
            expected_losses = sum(int(item["stats"]["losses"]) for item in combined["files"])
            self.assertEqual(combined["aggregate"]["wins"], expected_wins)
            self.assertEqual(combined["aggregate"]["draws"], expected_draws)
            self.assertEqual(combined["aggregate"]["losses"], expected_losses)
            self.assertIn("score_buckets", combined["aggregate"])
            self.assertIn("wdl", combined["aggregate"])
            self.assertIn("move_types", combined["aggregate"])
            self.assertNotIn("recommendation", combined["aggregate"])
            self.assertIn("aggregate_notes", combined["aggregate"])

    def test_collection_inspect_rejects_invalid_job_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixture.binpack"
            write_fixture_binpack(path)

            with self.assertRaisesRegex(ValueError, "jobs must be >= 1"):
                inspect_binpack_collection([path], jobs=0)


if __name__ == "__main__":
    unittest.main()
