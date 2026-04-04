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

            with BinpackStream([path], num_threads=1, cyclic=False) as stream:
                batch = stream.next_batch(2)
                self.assertIsNotNone(batch)
                assert batch is not None
                self.assertEqual(batch.white_indices.shape, (2, 32))
                self.assertEqual(batch.black_indices.shape, (2, 32))
                self.assertEqual(batch.stm.shape, (2,))
                self.assertTrue(((batch.white_indices >= -1) & (batch.white_indices < 768)).all())
                self.assertTrue(((batch.black_indices >= -1) & (batch.black_indices < 768)).all())


if __name__ == "__main__":
    unittest.main()
