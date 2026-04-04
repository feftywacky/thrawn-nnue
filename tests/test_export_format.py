from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thrawn_nnue.export import ExportedNetwork, load_export, _write_export


class ExportFormatTests(unittest.TestCase):
    def test_header_and_tensor_layout_round_trip(self) -> None:
        exported = ExportedNetwork(
            description="fixture",
            num_features=768,
            ft_size=4,
            hidden_size=2,
            ft_scale=127.0,
            dense_scale=64.0,
            wdl_scale=410.0,
            ft_bias=np.arange(4, dtype=np.int16),
            ft_weight=np.arange(768 * 4, dtype=np.int16).reshape(768, 4),
            l1_bias=np.array([7, -3], dtype=np.int32),
            l1_weight=np.arange(8 * 2, dtype=np.int8).reshape(8, 2),
            out_bias=np.array([11], dtype=np.int32),
            out_weight=np.array([[5], [-2]], dtype=np.int8),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixture.nnue"
            with path.open("wb") as handle:
                _write_export(handle, exported)

            loaded = load_export(path)
            self.assertEqual(loaded.description, "fixture")
            self.assertEqual(loaded.num_features, 768)
            self.assertEqual(loaded.ft_size, 4)
            self.assertEqual(loaded.hidden_size, 2)
            self.assertTrue(np.array_equal(loaded.ft_bias, exported.ft_bias))
            self.assertTrue(np.array_equal(loaded.ft_weight, exported.ft_weight))
            self.assertTrue(np.array_equal(loaded.l1_bias, exported.l1_bias))
            self.assertTrue(np.array_equal(loaded.l1_weight, exported.l1_weight))
            self.assertTrue(np.array_equal(loaded.out_bias, exported.out_bias))
            self.assertTrue(np.array_equal(loaded.out_weight, exported.out_weight))


if __name__ == "__main__":
    unittest.main()
