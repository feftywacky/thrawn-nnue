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
    EXPECTED_FC1_OUTPUT_SIZE,
    EXPECTED_FT_SIZE,
    EXPECTED_HIDDEN_SIZE,
    EXPECTED_NUM_FEATURES,
    FEATURE_SET_ID,
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


class ExportFormatTests(unittest.TestCase):
    def test_header_and_tensor_layout_round_trip(self) -> None:
        hidden_size = EXPECTED_HIDDEN_SIZE
        forward_size = 1
        fc0_output_size = hidden_size + forward_size
        fc1_input_size = hidden_size * 2
        fc1_output_size = EXPECTED_FC1_OUTPUT_SIZE

        ft_bias = np.arange(EXPECTED_FT_SIZE, dtype=np.int16)
        ft_weight = np.zeros((EXPECTED_NUM_FEATURES, EXPECTED_FT_SIZE), dtype=np.int16)
        ft_weight[0, 0] = 123
        ft_weight[-1, -1] = -456
        fc0_bias = np.arange(fc0_output_size, dtype=np.int32)
        fc0_weight = np.zeros((EXPECTED_FT_SIZE, fc0_output_size), dtype=np.int8)
        fc0_weight[0, 0] = 7
        fc0_weight[-1, -1] = -8
        fc1_bias = np.arange(fc1_output_size, dtype=np.int32)
        fc1_weight = np.zeros((fc1_input_size, fc1_output_size), dtype=np.int8)
        fc1_weight[0, 0] = 9
        fc1_weight[-1, -1] = -10
        fc2_weight = (np.arange(fc1_output_size) % 127).astype(np.int8)
        exported = ExportedNetwork(
            description="fixture",
            num_features=EXPECTED_NUM_FEATURES,
            ft_size=EXPECTED_FT_SIZE,
            hidden_size=hidden_size,
            forward_size=forward_size,
            fc1_output_size=fc1_output_size,
            ft_scale=255.0,
            fc0_scale=64.0,
            fc1_scale=64.0,
            fc2_scale=64.0,
            score_scale=1.0,
            ft_bias=ft_bias,
            ft_weight=ft_weight,
            fc0_bias=fc0_bias,
            fc0_weight=fc0_weight,
            fc1_bias=fc1_bias,
            fc1_weight=fc1_weight,
            fc2_bias=np.array([11], dtype=np.int32),
            fc2_weight=fc2_weight,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "fixture.nnue"
            with path.open("wb") as handle:
                _write_export(handle, exported)

            loaded = load_export(path)
            self.assertEqual(loaded.description, "fixture")
            self.assertEqual(loaded.num_features, EXPECTED_NUM_FEATURES)
            self.assertEqual(loaded.ft_size, EXPECTED_FT_SIZE)
            self.assertEqual(loaded.hidden_size, hidden_size)
            self.assertEqual(loaded.forward_size, forward_size)
            self.assertEqual(loaded.fc0_output_size, fc0_output_size)
            self.assertEqual(loaded.fc0_input_size, EXPECTED_FT_SIZE)
            self.assertEqual(loaded.fc1_input_size, fc1_input_size)
            self.assertEqual(loaded.fc1_output_size, fc1_output_size)
            self.assertEqual(loaded.fc0_scale, exported.fc0_scale)
            self.assertEqual(loaded.fc1_scale, exported.fc1_scale)
            self.assertEqual(loaded.fc2_scale, exported.fc2_scale)
            self.assertEqual(loaded.score_scale, exported.score_scale)
            self.assertTrue(np.array_equal(loaded.ft_bias, exported.ft_bias))
            self.assertTrue(np.array_equal(loaded.ft_weight, exported.ft_weight))
            self.assertTrue(np.array_equal(loaded.fc0_bias, exported.fc0_bias))
            self.assertTrue(np.array_equal(loaded.fc0_weight, exported.fc0_weight))
            self.assertTrue(np.array_equal(loaded.fc1_bias, exported.fc1_bias))
            self.assertTrue(np.array_equal(loaded.fc1_weight, exported.fc1_weight))
            self.assertTrue(np.array_equal(loaded.fc2_bias, exported.fc2_bias))
            self.assertTrue(np.array_equal(loaded.fc2_weight, exported.fc2_weight))

    def test_fit_quantization_scale_backs_off_to_avoid_clipping(self) -> None:
        scale = _fit_quantization_scale([np.array([5.0], dtype=np.float32)], 64.0, np.int8)
        self.assertLess(scale, 64.0)
        self.assertLessEqual(5.0 * scale, 127.0)

    def test_export_keeps_nnue2score_as_score_scale(self) -> None:
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
            ft=SimpleNamespace(weight=FakeTensor(np.zeros((2, 1), dtype=np.float32))),
            ft_bias=FakeTensor([0.0]),
            fc0=SimpleNamespace(
                weight=FakeTensor(np.zeros((2, 2), dtype=np.float32)),
                bias=FakeTensor([0.0, 0.0]),
            ),
            fc1=SimpleNamespace(
                weight=FakeTensor(np.zeros((1, 1), dtype=np.float32)),
                bias=FakeTensor([0.0]),
            ),
            fc2=SimpleNamespace(
                weight=FakeTensor(np.array([[0.25]], dtype=np.float32)),
                bias=FakeTensor([0.5]),
            ),
        )
        config = SimpleNamespace(
            export_description="fixture",
            num_features=2,
            ft_size=1,
            hidden_size=1,
            forward_size=1,
            fc1_output_size=1,
            export_ft_scale=100.0,
            export_dense_scale=64.0,
            nnue2score=4.0,
        )

        exported = _exported_network_from_model(model, config)
        dequantized_weight = exported.fc2_weight.astype(np.float32) / exported.fc2_scale
        dequantized_bias = exported.fc2_bias.astype(np.float32) / exported.fc2_scale
        self.assertAlmostEqual(float(dequantized_weight[0]), 0.25, delta=0.02)
        self.assertAlmostEqual(float(dequantized_bias[0]), 0.5, places=5)
        self.assertAlmostEqual(exported.score_scale, 4.0)

    def test_evaluate_export_uses_pairwise_screlu_crelu_and_forward_lane(self) -> None:
        hidden_size = EXPECTED_HIDDEN_SIZE
        fc0_output_size = hidden_size + 1
        fc1_input_size = hidden_size * 2
        fc1_output_size = EXPECTED_FC1_OUTPUT_SIZE
        half = EXPECTED_FT_SIZE // 2
        # Pairwise SqrCReLU multiplies each perspective's two halves, so both the
        # low (index 0) and high (index half) lanes must be driven to 1.0 for the
        # first fc0 input lane to be non-zero: crelu(1) * crelu(1) = 1.
        ft_bias = np.zeros(EXPECTED_FT_SIZE, dtype=np.int16)
        ft_bias[0] = 100
        ft_bias[half] = 100
        fc0_weight = np.zeros((EXPECTED_FT_SIZE, fc0_output_size), dtype=np.int8)
        fc0_weight[0, 0] = 100
        fc0_bias = np.zeros(fc0_output_size, dtype=np.int32)
        fc1_weight = np.zeros((fc1_input_size, fc1_output_size), dtype=np.int8)
        fc1_weight[0, 0] = 100
        fc1_weight[hidden_size, 0] = 100
        fc2_weight = np.zeros(fc1_output_size, dtype=np.int8)
        fc2_weight[0] = 100
        exported = ExportedNetwork(
            description="fixture",
            num_features=EXPECTED_NUM_FEATURES,
            ft_size=EXPECTED_FT_SIZE,
            hidden_size=hidden_size,
            forward_size=1,
            fc1_output_size=fc1_output_size,
            ft_scale=100.0,
            fc0_scale=100.0,
            fc1_scale=100.0,
            fc2_scale=100.0,
            score_scale=1.0,
            ft_bias=ft_bias,
            ft_weight=np.zeros((EXPECTED_NUM_FEATURES, EXPECTED_FT_SIZE), dtype=np.int16),
            fc0_bias=fc0_bias,
            fc0_weight=fc0_weight,
            fc1_bias=np.zeros(fc1_output_size, dtype=np.int32),
            fc1_weight=fc1_weight,
            fc2_bias=np.zeros(1, dtype=np.int32),
            fc2_weight=fc2_weight,
        )

        outputs = evaluate_export(
            exported,
            ["8/8/8/8/8/8/P7/K6k w - - 0 1"],
        )

        self.assertEqual(outputs, [1.0])

    def test_load_export_rejects_legacy_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.nnue"
            with path.open("wb") as handle:
                handle.write(b"THNNUE\x00\x01")
                handle.write((7).to_bytes(4, "little"))
            with self.assertRaisesRegex(ValueError, "Unsupported \\.nnue version"):
                load_export(path)

    def test_load_export_rejects_corrupt_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "truncated.nnue"
            description = b"fixture"
            header = HEADER_PREFIX_STRUCT.pack(MAGIC, VERSION) + HEADER_REST_STRUCT.pack(
                FEATURE_SET_ID.encode("ascii").ljust(16, b"\x00"),
                EXPECTED_NUM_FEATURES,
                EXPECTED_FT_SIZE,
                EXPECTED_HIDDEN_SIZE,
                1,
                EXPECTED_HIDDEN_SIZE + 1,
                EXPECTED_FT_SIZE,
                EXPECTED_HIDDEN_SIZE * 2,
                EXPECTED_FC1_OUTPUT_SIZE,
                OUTPUT_PERSPECTIVE_STM,
                255.0,
                64.0,
                64.0,
                64.0,
                1.0,
                len(description),
            )
            path.write_bytes(header + description)

            with self.assertRaisesRegex(ValueError, "ft_bias"):
                load_export(path)

    def test_load_export_rejects_trailing_data(self) -> None:
        exported = ExportedNetwork(
            description="fixture",
            num_features=EXPECTED_NUM_FEATURES,
            ft_size=EXPECTED_FT_SIZE,
            hidden_size=EXPECTED_HIDDEN_SIZE,
            forward_size=1,
            fc1_output_size=EXPECTED_FC1_OUTPUT_SIZE,
            ft_scale=255.0,
            fc0_scale=64.0,
            fc1_scale=64.0,
            fc2_scale=64.0,
            score_scale=1.0,
            ft_bias=np.zeros(EXPECTED_FT_SIZE, dtype=np.int16),
            ft_weight=np.zeros((EXPECTED_NUM_FEATURES, EXPECTED_FT_SIZE), dtype=np.int16),
            fc0_bias=np.zeros(EXPECTED_HIDDEN_SIZE + 1, dtype=np.int32),
            fc0_weight=np.zeros((EXPECTED_FT_SIZE, EXPECTED_HIDDEN_SIZE + 1), dtype=np.int8),
            fc1_bias=np.zeros(EXPECTED_FC1_OUTPUT_SIZE, dtype=np.int32),
            fc1_weight=np.zeros((EXPECTED_HIDDEN_SIZE * 2, EXPECTED_FC1_OUTPUT_SIZE), dtype=np.int8),
            fc2_bias=np.zeros(1, dtype=np.int32),
            fc2_weight=np.zeros(EXPECTED_FC1_OUTPUT_SIZE, dtype=np.int8),
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
    def test_verify_export_reports_parity_and_quantization_diagnostics(self) -> None:
        from thrawn_nnue.model import HalfKAv2HmNNUE

        model = HalfKAv2HmNNUE()
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
                    "datasets": ["/tmp/train.binpack"],
                    "max_epochs": 10,
                    "epoch_size": 100,
                },
                global_step=1,
                positions_seen=128,
                epoch_index=1,
            )
            export_checkpoint(checkpoint_path, nnue_path)
            report = verify_export(checkpoint_path, nnue_path)
            diagnostics = _export_quantization_diagnostics(load_export(nnue_path))

        self.assertNotIn("sanity_positions", report)
        self.assertIn("export_fc0_scale", report)
        self.assertIn("fc0_weight", diagnostics)


if __name__ == "__main__":
    unittest.main()
