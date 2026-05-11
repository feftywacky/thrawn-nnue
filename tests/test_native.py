from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thrawn_nnue.native import BinpackStream, inspect_binpack, write_fixture_binpack


class NativeTests(unittest.TestCase):
    def test_fixture_binpack_can_be_inspected_and_loaded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixture.binpack"
            write_fixture_binpack(path)

            stats = inspect_binpack(path)
            self.assertEqual(stats["entries_read"], 3)
            self.assertGreaterEqual(stats["white_to_move"], 1)
            self.assertGreaterEqual(stats["black_to_move"], 1)
            self.assertAlmostEqual(float(stats["min_score"]), -12.0 * 100.0 / 208.0, places=5)
            self.assertAlmostEqual(float(stats["max_score"]), 31.0 * 100.0 / 208.0, places=5)

            sampled_stats = inspect_binpack(path, sample_entries=2)
            self.assertTrue(sampled_stats["sampled"])
            self.assertEqual(sampled_stats["filters"]["sample_entries"], 2)
            self.assertEqual(sampled_stats["entries_seen"], 2)
            self.assertEqual(sampled_stats["entries_read"], 2)

            filtered_stats = inspect_binpack(path, max_abs_score=10.0)
            self.assertEqual(filtered_stats["entries_read"], 1)
            self.assertAlmostEqual(float(filtered_stats["min_score"]), -12.0 * 100.0 / 208.0, places=5)
            self.assertAlmostEqual(float(filtered_stats["max_score"]), -12.0 * 100.0 / 208.0, places=5)

            with BinpackStream([path], num_threads=1, cyclic=False) as stream:
                batch = stream.next_batch(2)
                self.assertIsNotNone(batch)
                assert batch is not None
                self.assertEqual(batch.white_indices.shape, (2, 30))
                self.assertEqual(batch.black_indices.shape, (2, 30))
                self.assertEqual(batch.stm.shape, (2,))
                self.assertTrue(((batch.white_indices >= -1) & (batch.white_indices < 40960)).all())
                self.assertTrue(((batch.black_indices >= -1) & (batch.black_indices < 40960)).all())
                self.assertEqual(batch.score.shape, (2,))
                self.assertTrue(set(float(score) for score in batch.score.tolist()).issubset({-12.0, 24.0, 31.0}))

            with BinpackStream([path], num_threads=1, cyclic=False, max_abs_score=10.0) as stream:
                batch = stream.next_batch(3)
                self.assertIsNotNone(batch)
                assert batch is not None
                self.assertEqual(batch.score.shape, (1,))
                self.assertAlmostEqual(float(batch.score[0]), -12.0, places=5)

            with BinpackStream([path], num_threads=1, cyclic=False, skip_tactical_positions=True) as stream:
                batch = stream.next_batch(3)
                self.assertIsNotNone(batch)

            with BinpackStream([path], num_threads=1, cyclic=False) as stream:
                batch = stream.next_batch(3)
                self.assertIsNotNone(batch)
                assert batch is not None
                all_scores = sorted(float(score) for score in batch.score.tolist())

            split_scores: list[float] = []
            for split_role in ("train", "validation"):
                with BinpackStream(
                    [path],
                    num_threads=1,
                    cyclic=False,
                    split_role=split_role,
                    validation_split_fraction=0.5,
                ) as stream:
                    batch = stream.next_batch(3)
                    if batch is not None:
                        split_scores.extend(float(score) for score in batch.score.tolist())

            self.assertEqual(sorted(split_scores), all_scores)


if __name__ == "__main__":
    unittest.main()
