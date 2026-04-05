from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from thrawn_nnue.export import (
    ExportedNetwork,
    _export_quantization_diagnostics,
    _exported_network_from_model,
    _fit_quantization_scale,
    load_export,
    _write_export,
)


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
            out_weight=np.array([[5], [-2]], dtype=np.int16),
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

    def test_fit_quantization_scale_backs_off_to_avoid_clipping(self) -> None:
        scale = _fit_quantization_scale([np.array([5.0], dtype=np.float32)], 64.0, np.int8)
        self.assertLess(scale, 64.0)
        self.assertLessEqual(5.0 * scale, 127.0)

    def test_export_uses_adaptive_dense_scale_when_weights_are_large(self) -> None:
        class FakeTensor:
            def __init__(self, values):
                self._values = np.asarray(values, dtype=np.float32)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._values

        model = SimpleNamespace(
            ft=SimpleNamespace(weight=FakeTensor([[0.25]])),
            ft_bias=FakeTensor([0.0]),
            l1=SimpleNamespace(
                weight=FakeTensor([[3.5, -3.5], [0.25, -0.25]]),
                bias=FakeTensor([0.0, 0.0]),
            ),
            output=SimpleNamespace(
                weight=FakeTensor([[4.0, -4.0]]),
                bias=FakeTensor([0.0]),
            ),
        )
        config = SimpleNamespace(
            export_description="fixture",
            num_features=1,
            ft_size=1,
            hidden_size=2,
            export_ft_scale=127.0,
            export_dense_scale=64.0,
            wdl_scale=410.0,
        )

        exported = _exported_network_from_model(model, config)

        self.assertLess(exported.dense_scale, 64.0)
        diagnostics = _export_quantization_diagnostics(exported)
        self.assertEqual(diagnostics["l1_weight"]["positive_limit_hits"], 0.0)
        self.assertEqual(diagnostics["l1_weight"]["negative_limit_hits"], 0.0)
        self.assertEqual(diagnostics["out_weight"]["positive_limit_hits"], 0.0)
        self.assertEqual(diagnostics["out_weight"]["negative_limit_hits"], 0.0)
        self.assertGreater(diagnostics["out_weight"]["max_abs_quantized"], 127.0)


if __name__ == "__main__":
    unittest.main()
