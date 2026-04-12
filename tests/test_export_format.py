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
    evaluate_export,
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
            output_buckets=3,
            ft_scale=127.0,
            l1_scale=4096.0,
            dense_scale=64.0,
            wdl_scale=410.0,
            ft_bias=np.arange(4, dtype=np.int16),
            ft_weight=np.arange(768 * 4, dtype=np.int16).reshape(768, 4),
            l1_bias=np.array([7, -3], dtype=np.int32),
            l1_weight=np.arange(8 * 2, dtype=np.int16).reshape(8, 2),
            out_bias=np.array([11, 13, 17], dtype=np.int32),
            out_weight=np.array([[5, 6, 7], [-2, -3, -4]], dtype=np.int16),
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
            self.assertEqual(loaded.output_buckets, 3)
            self.assertEqual(loaded.l1_scale, exported.l1_scale)
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

    def test_export_uses_separate_auto_fit_l1_scale_when_weights_are_large(self) -> None:
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
                weight=FakeTensor([[4.0, -4.0], [2.0, -2.0]]),
                bias=FakeTensor([0.0, 0.0]),
            ),
        )
        config = SimpleNamespace(
            export_description="fixture",
            num_features=1,
            ft_size=1,
            hidden_size=2,
            output_buckets=2,
            export_ft_scale=127.0,
            export_dense_scale=64.0,
            wdl_scale=410.0,
        )

        exported = _exported_network_from_model(model, config)

        self.assertEqual(exported.dense_scale, 64.0)
        self.assertGreater(exported.l1_scale, exported.dense_scale)
        diagnostics = _export_quantization_diagnostics(exported)
        self.assertEqual(diagnostics["l1_weight"]["positive_limit_hits"], 0.0)
        self.assertEqual(diagnostics["l1_weight"]["negative_limit_hits"], 0.0)
        self.assertEqual(diagnostics["out_weight"]["positive_limit_hits"], 0.0)
        self.assertEqual(diagnostics["out_weight"]["negative_limit_hits"], 0.0)
        self.assertGreater(diagnostics["l1_weight"]["max_abs_quantized"], 127.0)
        self.assertGreater(diagnostics["out_weight"]["max_abs_quantized"], 127.0)

    def test_evaluate_export_selects_phase_bucket(self) -> None:
        exported = ExportedNetwork(
            description="fixture",
            num_features=768,
            ft_size=1,
            hidden_size=1,
            output_buckets=8,
            ft_scale=1.0,
            l1_scale=1.0,
            dense_scale=1.0,
            wdl_scale=410.0,
            ft_bias=np.zeros(1, dtype=np.int16),
            ft_weight=np.zeros((768, 1), dtype=np.int16),
            l1_bias=np.zeros(1, dtype=np.int32),
            l1_weight=np.zeros((2, 1), dtype=np.int16),
            out_bias=np.arange(8, dtype=np.int32),
            out_weight=np.zeros((1, 8), dtype=np.int16),
        )

        outputs = evaluate_export(
            exported,
            [
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "8/8/8/8/8/8/8/K6k w - - 0 1",
            ],
        )

        self.assertEqual(outputs, [0.0, 7.0])

    def test_evaluate_export_uses_screlu_before_first_dense(self) -> None:
        fen = "8/8/8/8/8/8/P7/K6k w - - 0 1"

        exported = ExportedNetwork(
            description="fixture",
            num_features=768,
            ft_size=1,
            hidden_size=1,
            output_buckets=1,
            ft_scale=100.0,
            l1_scale=100.0,
            dense_scale=100.0,
            wdl_scale=410.0,
            ft_bias=np.array([50], dtype=np.int16),
            ft_weight=np.zeros((768, 1), dtype=np.int16),
            l1_bias=np.zeros(1, dtype=np.int32),
            l1_weight=np.array([[100], [0]], dtype=np.int16),
            out_bias=np.zeros(1, dtype=np.int32),
            out_weight=np.array([[100]], dtype=np.int16),
        )

        outputs = evaluate_export(
            exported,
            [fen],
        )

        self.assertEqual(len(outputs), 1)
        self.assertAlmostEqual(outputs[0], 0.25, places=8)

    def test_evaluate_export_uses_l1_scale_separately_from_dense_scale(self) -> None:
        fen = "8/8/8/8/8/8/P7/K6k w - - 0 1"

        exported = ExportedNetwork(
            description="fixture",
            num_features=768,
            ft_size=1,
            hidden_size=1,
            output_buckets=1,
            ft_scale=100.0,
            l1_scale=1000.0,
            dense_scale=10.0,
            wdl_scale=410.0,
            ft_bias=np.array([100], dtype=np.int16),
            ft_weight=np.zeros((768, 1), dtype=np.int16),
            l1_bias=np.zeros(1, dtype=np.int32),
            l1_weight=np.array([[500], [0]], dtype=np.int16),
            out_bias=np.zeros(1, dtype=np.int32),
            out_weight=np.array([[10]], dtype=np.int16),
        )

        outputs = evaluate_export(exported, [fen])

        self.assertEqual(outputs, [0.5])

    def test_version3_scalar_round_trip_supports_legacy_256x32_8bucket_layout(self) -> None:
        ft_weight = np.zeros((768, 256), dtype=np.int16)
        ft_weight[12, 0] = 7
        ft_weight[396, 1] = -5
        l1_weight = np.zeros((512, 32), dtype=np.int8)
        l1_weight[0, 0] = 3
        l1_weight[1, 1] = -2
        out_weight = np.zeros((32, 8), dtype=np.int16)
        out_weight[0, 0] = 9
        out_weight[1, 7] = -4

        exported = ExportedNetwork(
            description="fixture-v10",
            num_features=768,
            ft_size=256,
            hidden_size=32,
            output_buckets=8,
            ft_scale=127.0,
            l1_scale=96.0,
            dense_scale=96.0,
            wdl_scale=410.0,
            ft_bias=np.zeros(256, dtype=np.int16),
            ft_weight=ft_weight,
            l1_bias=np.zeros(32, dtype=np.int32),
            l1_weight=l1_weight,
            out_bias=np.arange(8, dtype=np.int32),
            out_weight=out_weight,
            version=3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixture-v10.nnue"
            with path.open("wb") as handle:
                _write_export(handle, exported)
            loaded = load_export(path)

        self.assertEqual(loaded.version, 3)
        self.assertEqual(loaded.l1_scale, loaded.dense_scale)
        self.assertEqual(loaded.ft_weight.shape, (768, 256))
        self.assertEqual(loaded.l1_weight.shape, (512, 32))
        self.assertEqual(loaded.out_weight.shape, (32, 8))
        self.assertEqual(int(loaded.ft_weight[12, 0]), 7)
        self.assertEqual(int(loaded.ft_weight[396, 1]), -5)
        self.assertEqual(int(loaded.out_bias[7]), 7)
        self.assertEqual(int(loaded.out_weight[0, 0]), 9)
        self.assertEqual(int(loaded.out_weight[1, 7]), -4)


if __name__ == "__main__":
    unittest.main()
