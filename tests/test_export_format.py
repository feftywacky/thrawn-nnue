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
    ExportedNetwork,
    _export_quantization_diagnostics,
    _exported_network_from_model,
    _fit_quantization_scale,
    _write_export,
    evaluate_export,
    export_checkpoint,
    load_export,
    verify_export,
)


class ExportFormatTests(unittest.TestCase):
    def test_header_and_tensor_layout_round_trip(self) -> None:
        exported = ExportedNetwork(
            description="fixture",
            num_features=40960,
            ft_size=4,
            l1_size=2,
            l2_size=2,
            ft_scale=127.0,
            l1_scale=64.0,
            l2_scale=64.0,
            out_scale=64.0,
            wdl_scale=410.0,
            ft_bias=np.arange(4, dtype=np.int16),
            ft_weight=np.arange(40960 * 4, dtype=np.int16).reshape(40960, 4),
            l1_bias=np.array([7, -3], dtype=np.int32),
            l1_weight=np.arange(8 * 2, dtype=np.int8).reshape(8, 2),
            l2_bias=np.array([5, 9], dtype=np.int32),
            l2_weight=np.arange(2 * 2, dtype=np.int8).reshape(2, 2),
            out_bias=np.array([11], dtype=np.int32),
            out_weight=np.array([5, -2], dtype=np.int8),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixture.nnue"
            with path.open("wb") as handle:
                _write_export(handle, exported)

            loaded = load_export(path)
            self.assertEqual(loaded.description, "fixture")
            self.assertEqual(loaded.num_features, 40960)
            self.assertEqual(loaded.ft_size, 4)
            self.assertEqual(loaded.l1_size, 2)
            self.assertEqual(loaded.l2_size, 2)
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
            wdl_scale=410.0,
        )

        exported = _exported_network_from_model(model, config)
        dequantized = exported.ft_weight.astype(np.float32) / exported.ft_scale
        self.assertAlmostEqual(float(dequantized[0, 0]), 1.0, places=5)
        self.assertAlmostEqual(float(dequantized[640, 0]), 0.25, places=5)
        diagnostics = _export_quantization_diagnostics(exported)
        self.assertEqual(diagnostics["ft_weight"]["positive_limit_hits"], 0.0)

    def test_export_folds_final_eval_scale_into_output_layer(self) -> None:
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
            final_eval_scale=16.0,
        )
        config = SimpleNamespace(
            export_description="fixture",
            num_features=640,
            ft_size=1,
            l1_size=1,
            l2_size=1,
            export_ft_scale=100.0,
            export_dense_scale=64.0,
            wdl_scale=410.0,
        )

        exported = _exported_network_from_model(model, config)
        dequantized_weight = exported.out_weight.astype(np.float32) / exported.out_scale
        dequantized_bias = exported.out_bias.astype(np.float32) / exported.out_scale
        self.assertAlmostEqual(float(dequantized_weight[0]), 4.0, delta=0.02)
        self.assertAlmostEqual(float(dequantized_bias[0]), 8.0, places=5)

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
            wdl_scale=410.0,
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


@unittest.skipUnless(torch is not None, "PyTorch is required for verify-export tests")
class VerifyExportTests(unittest.TestCase):
    def test_verify_export_reports_material_sanity_suite(self) -> None:
        from thrawn_nnue.model import HalfKPNNUE

        model = HalfKPNNUE()
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
