from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ModuleNotFoundError:
    torch = None

from thrawn_nnue.checkpoint import save_checkpoint
from thrawn_nnue.export import (
    EXPECTED_NUM_FEATURES,
    ExportedNetwork,
    HEADER_PREFIX_STRUCT,
    HEADER_REST_STRUCT,
    MAGIC,
    OUTPUT_PERSPECTIVE_STM,
    VERSION,
    _export_quantization_diagnostics,
    _exported_network_from_model,
    _fit_quantization_scale,
    _write_export,
    evaluate_export,
    export_checkpoint,
    load_export,
    verify_export,
)


PRODUCTION_FT_SIZE = 1024
PRODUCTION_L1_SIZE = 256
PRODUCTION_L2_SIZE = 64


class ExportFormatTests(unittest.TestCase):
    def _round_trip_exported_network(self, *, ft_size: int, l1_size: int, l2_size: int) -> None:
        ft_bias = np.arange(ft_size, dtype=np.int16)
        ft_weight = np.zeros((EXPECTED_NUM_FEATURES, ft_size), dtype=np.int16)
        ft_weight[0, 0] = 123
        ft_weight[-1, -1] = -456
        l1_bias = np.arange(l1_size, dtype=np.int32)
        l1_weight = np.zeros((ft_size * 2, l1_size), dtype=np.int8)
        l1_weight[0, 0] = 7
        l1_weight[-1, -1] = -8
        l2_bias = np.arange(l2_size, dtype=np.int32)
        l2_weight = np.zeros((l1_size, l2_size), dtype=np.int8)
        l2_weight[0, 0] = 9
        l2_weight[-1, -1] = -10
        out_weight = (np.arange(l2_size) % 127).astype(np.int8)
        exported = ExportedNetwork(
            description="fixture",
            num_features=EXPECTED_NUM_FEATURES,
            ft_size=ft_size,
            l1_size=l1_size,
            l2_size=l2_size,
            ft_scale=127.0,
            l1_scale=64.0,
            l2_scale=64.0,
            out_scale=64.0,
            ft_bias=ft_bias,
            ft_weight=ft_weight,
            l1_bias=l1_bias,
            l1_weight=l1_weight,
            l2_bias=l2_bias,
            l2_weight=l2_weight,
            out_bias=np.array([11], dtype=np.int32),
            out_weight=out_weight,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixture.nnue"
            with path.open("wb") as handle:
                _write_export(handle, exported)

            loaded = load_export(path)
            self.assertEqual(loaded.description, "fixture")
            self.assertEqual(loaded.num_features, EXPECTED_NUM_FEATURES)
            self.assertEqual(loaded.ft_size, ft_size)
            self.assertEqual(loaded.l1_size, l1_size)
            self.assertEqual(loaded.l2_size, l2_size)
            self.assertEqual(loaded.l1_scale, exported.l1_scale)
            self.assertEqual(loaded.l2_scale, exported.l2_scale)
            self.assertEqual(loaded.out_scale, exported.out_scale)
            self.assertTrue(np.array_equal(loaded.ft_bias, exported.ft_bias))
            self.assertTrue(np.array_equal(loaded.ft_weight, exported.ft_weight))
            self.assertTrue(np.array_equal(loaded.l1_bias, exported.l1_bias))
            self.assertTrue(np.array_equal(loaded.l1_weight, exported.l1_weight))
            self.assertTrue(np.array_equal(loaded.l2_bias, exported.l2_bias))
            self.assertTrue(np.array_equal(loaded.l2_weight, exported.l2_weight))
            self.assertTrue(np.array_equal(loaded.out_bias, exported.out_bias))
            self.assertTrue(np.array_equal(loaded.out_weight, exported.out_weight))

    def test_header_and_tensor_layout_round_trip(self) -> None:
        self._round_trip_exported_network(
            ft_size=PRODUCTION_FT_SIZE,
            l1_size=PRODUCTION_L1_SIZE,
            l2_size=PRODUCTION_L2_SIZE,
        )

    def test_fit_quantization_scale_backs_off_to_avoid_clipping(self) -> None:
        scale = _fit_quantization_scale([np.array([5.0], dtype=np.float32)], 64.0, np.int8)
        self.assertLess(scale, 64.0)
        self.assertLessEqual(5.0 * scale, 127.0)

    def test_export_coalesces_factor_weights(self) -> None:
        class FakeTensor:
            def __init__(self, values):
                self._values = np.asarray(values, dtype=np.float32)

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return self._values

        ft_weight = np.zeros((1280, 1), dtype=np.float32)
        ft_weight[0, 0] = 0.25
        ft_weight[640, 0] = -0.50
        factor_weight = np.zeros((640, 1), dtype=np.float32)
        factor_weight[0, 0] = 0.75

        model = SimpleNamespace(
            ft=SimpleNamespace(weight=FakeTensor(ft_weight)),
            ft_factor=SimpleNamespace(weight=FakeTensor(factor_weight)),
            ft_bias=FakeTensor([0.0]),
            l1=SimpleNamespace(
                weight=FakeTensor(np.zeros((2, 1), dtype=np.float32)),
                bias=FakeTensor([0.0]),
            ),
            l2=SimpleNamespace(
                weight=FakeTensor(np.zeros((1, 1), dtype=np.float32)),
                bias=FakeTensor([0.0]),
            ),
            output=SimpleNamespace(
                weight=FakeTensor(np.zeros((1, 1), dtype=np.float32)),
                bias=FakeTensor([0.0]),
            ),
        )
        config = SimpleNamespace(
            export_description="fixture",
            num_features=1280,
            ft_size=1,
            l1_size=1,
            l2_size=1,
            export_ft_scale=100.0,
            export_dense_scale=64.0,
        )

        exported = _exported_network_from_model(model, config)
        dequantized = exported.ft_weight.astype(np.float32) / exported.ft_scale
        self.assertAlmostEqual(float(dequantized[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(dequantized[640, 0]), 0.25, places=5)
        diagnostics = _export_quantization_diagnostics(exported)
        self.assertEqual(diagnostics["ft_weight"]["positive_limit_hits"], 0.0)

    def test_export_uses_direct_scalar_output_layer(self) -> None:
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
            ft=SimpleNamespace(weight=FakeTensor(np.zeros((640, 1), dtype=np.float32))),
            ft_bias=FakeTensor([0.0]),
            l1=SimpleNamespace(
                weight=FakeTensor(np.zeros((2, 1), dtype=np.float32)),
                bias=FakeTensor([0.0]),
            ),
            l2=SimpleNamespace(
                weight=FakeTensor(np.zeros((1, 1), dtype=np.float32)),
                bias=FakeTensor([0.0]),
            ),
            output=SimpleNamespace(
                weight=FakeTensor(np.array([[0.25]], dtype=np.float32)),
                bias=FakeTensor([0.5]),
            ),
        )
        config = SimpleNamespace(
            export_description="fixture",
            num_features=640,
            ft_size=1,
            l1_size=1,
            l2_size=1,
            export_ft_scale=100.0,
            export_dense_scale=64.0,
        )

        exported = _exported_network_from_model(model, config)
        dequantized_weight = exported.out_weight.astype(np.float32) / exported.out_scale
        dequantized_bias = exported.out_bias.astype(np.float32) / exported.out_scale
        self.assertAlmostEqual(float(dequantized_weight[0]), 0.25, delta=0.02)
        self.assertAlmostEqual(float(dequantized_bias[0]), 0.5, places=5)

    def test_evaluate_export_uses_direct_scalar_output(self) -> None:
        exported = ExportedNetwork(
            description="fixture",
            num_features=40960,
            ft_size=1,
            l1_size=1,
            l2_size=1,
            ft_scale=100.0,
            l1_scale=100.0,
            l2_scale=100.0,
            out_scale=100.0,
            ft_bias=np.array([100], dtype=np.int16),
            ft_weight=np.zeros((40960, 1), dtype=np.int16),
            l1_bias=np.zeros(1, dtype=np.int32),
            l1_weight=np.array([[50], [0]], dtype=np.int8),
            l2_bias=np.zeros(1, dtype=np.int32),
            l2_weight=np.array([[100]], dtype=np.int8),
            out_bias=np.zeros(1, dtype=np.int32),
            out_weight=np.array([100], dtype=np.int8),
        )

        outputs = evaluate_export(
            exported,
            ["8/8/8/8/8/8/P7/K6k w - - 0 1"],
        )

        self.assertEqual(outputs, [0.5])

    def test_load_export_rejects_legacy_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.nnue"
            with path.open("wb") as handle:
                handle.write(b"THNNUE\x00\x01")
                handle.write((4).to_bytes(4, "little"))
            with self.assertRaisesRegex(ValueError, "Unsupported \\.nnue version"):
                load_export(path)

    def test_load_export_rejects_corrupt_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "truncated.nnue"
            description = b"fixture"
            header = HEADER_PREFIX_STRUCT.pack(MAGIC, VERSION) + HEADER_REST_STRUCT.pack(
                b"halfkp_v1".ljust(16, b"\x00"),
                EXPECTED_NUM_FEATURES,
                PRODUCTION_FT_SIZE,
                PRODUCTION_L1_SIZE,
                PRODUCTION_L2_SIZE,
                OUTPUT_PERSPECTIVE_STM,
                127.0,
                64.0,
                64.0,
                64.0,
                len(description),
            )
            path.write_bytes(header + description)

            with self.assertRaisesRegex(ValueError, "ft_bias"):
                load_export(path)

    def test_load_export_rejects_trailing_data(self) -> None:
        exported = ExportedNetwork(
            description="fixture",
            num_features=EXPECTED_NUM_FEATURES,
            ft_size=PRODUCTION_FT_SIZE,
            l1_size=PRODUCTION_L1_SIZE,
            l2_size=PRODUCTION_L2_SIZE,
            ft_scale=127.0,
            l1_scale=64.0,
            l2_scale=64.0,
            out_scale=64.0,
            ft_bias=np.zeros(PRODUCTION_FT_SIZE, dtype=np.int16),
            ft_weight=np.zeros((EXPECTED_NUM_FEATURES, PRODUCTION_FT_SIZE), dtype=np.int16),
            l1_bias=np.zeros(PRODUCTION_L1_SIZE, dtype=np.int32),
            l1_weight=np.zeros((PRODUCTION_FT_SIZE * 2, PRODUCTION_L1_SIZE), dtype=np.int8),
            l2_bias=np.zeros(PRODUCTION_L2_SIZE, dtype=np.int32),
            l2_weight=np.zeros((PRODUCTION_L1_SIZE, PRODUCTION_L2_SIZE), dtype=np.int8),
            out_bias=np.zeros(1, dtype=np.int32),
            out_weight=np.zeros(PRODUCTION_L2_SIZE, dtype=np.int8),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trailing.nnue"
            with path.open("wb") as handle:
                _write_export(handle, exported)
                handle.write(b"x")

            with self.assertRaisesRegex(ValueError, "trailing data"):
                load_export(path)


@unittest.skipUnless(torch is not None, "PyTorch is required for verify-export tests")
class VerifyExportTests(unittest.TestCase):
    def test_verify_export_reports_material_sanity_suite(self) -> None:
        from thrawn_nnue.model import HalfKPNNUE

        model = HalfKPNNUE(ft_size=4, l1_size=2, l2_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.992)
        scaler = torch.cuda.amp.GradScaler(enabled=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_path = root / "checkpoint.pt"
            nnue_path = root / "model.nnue"
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                config={
                    "run_name": "test",
                    "train_datasets": ["/tmp/train.binpack"],
                    "total_train_positions": 1000,
                    "epoch_positions": 100,
                    "ft_size": 4,
                    "l1_size": 2,
                    "l2_size": 2,
                },
                global_step=1,
                positions_seen=128,
                epoch_index=1,
            )
            export_checkpoint(checkpoint_path, nnue_path)
            report = verify_export(checkpoint_path, nnue_path)

        self.assertIn("sanity_positions", report)
        self.assertEqual(len(report["sanity_positions"]), 5)
        self.assertIn("material_ordering_ok", report)
        self.assertIn("starting_position_near_zero", report)


if __name__ == "__main__":
    unittest.main()
