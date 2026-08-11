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
    DEFAULT_VERIFICATION_FENS,
    EXPECTED_FT_SIZE,
    EXPECTED_L2_SIZE,
    EXPECTED_L3_SIZE,
    EXPECTED_NUM_BUCKETS,
    EXPECTED_NUM_FEATURES,
    FEATURE_SET_ID,
    ExportedNetwork,
    HEADER_PREFIX_STRUCT,
    HEADER_REST_STRUCT,
    MAGIC,
    OUTPUT_PERSPECTIVE_STM,
    VERSION,
    QuantizationScaleMismatchError,
    _batch_arrays_from_fens,
    _export_quantization_diagnostics,
    _exported_network_from_model,
    _fit_quantization_scale,
    _write_export,
    bucket_for_piece_count,
    evaluate_export,
    export_checkpoint,
    load_export,
    verify_export,
)


class ExportFormatTests(unittest.TestCase):
    def test_header_and_tensor_layout_round_trip(self) -> None:
        num_buckets = EXPECTED_NUM_BUCKETS
        l2_size = EXPECTED_L2_SIZE
        l3_size = EXPECTED_L3_SIZE
        fc0_input_size = EXPECTED_FT_SIZE
        fc1_input_size = l2_size * 2
        fc2_input_size = l2_size * 2 + l3_size * 2

        ft_bias = np.arange(EXPECTED_FT_SIZE, dtype=np.int16)
        ft_weight = np.zeros((EXPECTED_NUM_FEATURES, EXPECTED_FT_SIZE), dtype=np.int16)
        ft_weight[0, 0] = 123
        ft_weight[-1, -1] = -456

        fc0_bias = np.stack(
            [np.arange(l2_size, dtype=np.int32) + bucket for bucket in range(num_buckets)]
        )
        fc0_weight = np.zeros((num_buckets, fc0_input_size, l2_size), dtype=np.int8)
        fc0_weight[:, 0, 0] = 7
        fc0_weight[:, -1, -1] = -8
        fc1_bias = np.stack(
            [np.arange(l3_size, dtype=np.int32) + bucket for bucket in range(num_buckets)]
        )
        fc1_weight = np.zeros((num_buckets, fc1_input_size, l3_size), dtype=np.int8)
        fc1_weight[:, 0, 0] = 9
        fc1_weight[:, -1, -1] = -10
        fc2_bias = np.arange(num_buckets, dtype=np.int32)
        fc2_weight = np.stack(
            [(np.arange(fc2_input_size) % 127).astype(np.int8) for _ in range(num_buckets)]
        )
        exported = ExportedNetwork(
            description="fixture",
            num_features=EXPECTED_NUM_FEATURES,
            ft_size=EXPECTED_FT_SIZE,
            num_buckets=num_buckets,
            l2_size=l2_size,
            l3_size=l3_size,
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
            fc2_bias=fc2_bias,
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
            self.assertEqual(loaded.num_buckets, num_buckets)
            self.assertEqual(loaded.l2_size, l2_size)
            self.assertEqual(loaded.l3_size, l3_size)
            self.assertEqual(loaded.fc0_input_size, fc0_input_size)
            self.assertEqual(loaded.fc1_input_size, fc1_input_size)
            self.assertEqual(loaded.fc2_input_size, fc2_input_size)
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

    def test_fit_quantization_scale_raises_loudly_instead_of_silently_rescaling(self) -> None:
        # A weight of 5.0 at scale 64.0 would need int8 value 320, way past
        # the int8 range -- this must be a loud failure (QAT trained the
        # forward pass at the *requested* scale; quietly rescaling here would
        # export a net using a different scale than the one training
        # simulated, invalidating QAT with no visible symptom).
        with self.assertRaisesRegex(QuantizationScaleMismatchError, "scale mismatch"):
            _fit_quantization_scale([np.array([5.0], dtype=np.float32)], 64.0, np.int8)

    def test_fit_quantization_scale_error_names_the_offending_tensor_and_the_fix(self) -> None:
        # The message must be actionable: which tensor overflowed, and what
        # to do about it (lower the requested scale, or the weights
        # genuinely diverged -- for ft specifically, nothing clamps it).
        with self.assertRaisesRegex(QuantizationScaleMismatchError, r"ft_weight") as ctx:
            _fit_quantization_scale(
                [np.array([0.1], dtype=np.float32), np.array([5.0], dtype=np.float32)],
                64.0,
                np.int8,
                names=["ft_bias", "ft_weight"],
            )
        message = str(ctx.exception)
        self.assertIn("export_ft_scale", message)
        self.assertIn("export_dense_scale", message)
        self.assertIn("_clip_model_weights", message)

    def test_fit_quantization_scale_returns_requested_scale_when_it_fits(self) -> None:
        scale = _fit_quantization_scale([np.array([0.5], dtype=np.float32)], 64.0, np.int8)
        self.assertEqual(scale, 64.0)

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

        # StackedLinear layout: fc0/fc1/fc2 are plain nn.Linear-shaped
        # (weight: (out_per_bucket * num_buckets, in_features), bias:
        # (out_per_bucket * num_buckets,)); bucket b is rows/entries
        # [b*out_per_bucket : (b+1)*out_per_bucket]. Use num_buckets=2 so the
        # reshape-and-slice extraction is actually exercised, not a no-op.
        num_buckets = 2
        l2_size = 1
        l3_size = 1
        fc0_input_size = 1  # == ft_size
        fc1_input_size = l2_size * 2
        fc2_input_size = l2_size * 2 + l3_size * 2

        model = SimpleNamespace(
            ft=SimpleNamespace(weight=FakeTensor(np.zeros((2, 1), dtype=np.float32))),
            ft_bias=FakeTensor([0.0]),
            fc0=SimpleNamespace(
                weight=FakeTensor(np.zeros((l2_size * num_buckets, fc0_input_size), dtype=np.float32)),
                bias=FakeTensor(np.zeros(l2_size * num_buckets, dtype=np.float32)),
            ),
            fc1=SimpleNamespace(
                weight=FakeTensor(np.zeros((l3_size * num_buckets, fc1_input_size), dtype=np.float32)),
                bias=FakeTensor(np.zeros(l3_size * num_buckets, dtype=np.float32)),
            ),
            fc2=SimpleNamespace(
                weight=FakeTensor(
                    np.array([[0.25, 0.25, 0.25, 0.25], [0.6, 0.6, 0.6, 0.6]], dtype=np.float32)
                ),
                bias=FakeTensor(np.array([0.5, -0.5], dtype=np.float32)),
            ),
        )
        config = SimpleNamespace(
            export_description="fixture",
            num_features=2,
            ft_size=1,
            l2_size=l2_size,
            l3_size=l3_size,
            num_buckets=num_buckets,
            export_ft_scale=100.0,
            export_dense_scale=64.0,
            nnue2score=4.0,
        )

        exported = _exported_network_from_model(model, config)
        dequantized_weight = exported.fc2_weight.astype(np.float32) / exported.fc2_scale
        dequantized_bias = exported.fc2_bias.astype(np.float32) / exported.fc2_scale
        self.assertEqual(dequantized_weight.shape, (num_buckets, fc2_input_size))
        self.assertEqual(dequantized_bias.shape, (num_buckets,))
        self.assertAlmostEqual(float(dequantized_weight[0, 0]), 0.25, delta=0.02)
        self.assertAlmostEqual(float(dequantized_weight[1, 0]), 0.6, delta=0.02)
        self.assertAlmostEqual(float(dequantized_bias[0]), 0.5, places=4)
        self.assertAlmostEqual(float(dequantized_bias[1]), -0.5, places=4)
        self.assertAlmostEqual(exported.score_scale, 4.0)

    def test_bucket_for_piece_count_matches_stockfish_layerstacks_rule(self) -> None:
        # Stockfish's LayerStacks rule: bucket = clamp((piece_count - 1) // 4, 0, num_buckets - 1).
        # 2..32 pieces (both kings always present) must map into 0..7.
        expectations = {
            2: 0,
            4: 0,
            5: 1,
            8: 1,
            9: 2,
            12: 2,
            13: 3,
            28: 6,
            29: 7,
            30: 7,
            31: 7,
            32: 7,
        }
        for piece_count, expected_bucket in expectations.items():
            with self.subTest(piece_count=piece_count):
                self.assertEqual(bucket_for_piece_count(piece_count, num_buckets=8), expected_bucket)

        # Edges called out explicitly by the spec.
        self.assertEqual(bucket_for_piece_count(2, num_buckets=8), 0)
        self.assertEqual(bucket_for_piece_count(32, num_buckets=8), 7)

    def test_evaluate_export_matches_hand_computed_forward_pass(self) -> None:
        # A fully hand-traceable single-bucket network that exercises every
        # detail called out by the spec:
        #  - fc0 is RAW (pre-activation) when read for the skip connection
        #  - the skip lanes are l2_size-2 / l2_size-1, and those lanes are
        #    ALSO fed through the activations like every other lane
        #  - the squared activation is square-THEN-clamp (so a negative raw
        #    value still produces a positive squared term), not
        #    clamp(x, 0, 1) ** 2
        #  - the output head sees concat(a0, a1), both dense layers' activations
        ft_size = 4
        half = ft_size // 2
        num_buckets = 1
        l2_size = 2
        l3_size = 2
        scale = 100.0

        ft_bias = np.zeros(ft_size, dtype=np.int16)
        ft_bias[0] = 100
        ft_bias[half] = 100
        ft_weight = np.zeros((EXPECTED_NUM_FEATURES, ft_size), dtype=np.int16)

        fc0_weight = np.zeros((num_buckets, ft_size, l2_size), dtype=np.int8)
        fc0_weight[0, 0, 0] = 100
        fc0_bias = np.array([[-130, 150]], dtype=np.int32)

        fc1_weight = np.zeros((num_buckets, l2_size * 2, l3_size), dtype=np.int8)
        fc1_weight[0, 0, 0] = 100
        fc1_weight[0, 2, 1] = 100
        fc1_bias = np.array([[-100, 120]], dtype=np.int32)

        fc2_input_size = l2_size * 2 + l3_size * 2
        fc2_weight = np.full((num_buckets, fc2_input_size), 100, dtype=np.int8)
        fc2_bias = np.array([500], dtype=np.int32)

        exported = ExportedNetwork(
            description="fixture",
            num_features=EXPECTED_NUM_FEATURES,
            ft_size=ft_size,
            num_buckets=num_buckets,
            l2_size=l2_size,
            l3_size=l3_size,
            ft_scale=scale,
            fc0_scale=scale,
            fc1_scale=scale,
            fc2_scale=scale,
            score_scale=1.0,
            ft_bias=ft_bias,
            ft_weight=ft_weight,
            fc0_bias=fc0_bias,
            fc0_weight=fc0_weight,
            fc1_bias=fc1_bias,
            fc1_weight=fc1_weight,
            fc2_bias=fc2_bias,
            fc2_weight=fc2_weight,
        )

        outputs = evaluate_export(exported, ["8/8/8/8/8/8/P7/K6k w - - 0 1"])

        # Hand-derived expectation. evaluate_export floors every activation onto
        # the 128-wide uint8 grid (floor(x*128+1e-5)/128, ACT_QUANT_SCALE),
        # matching the engine's integer bitshift; the FT's own per-perspective
        # CReLU halves clamp to FT_CRELU_CEIL = 255/256 (not 1.0) before the
        # pairwise multiply, and the fc0/fc1 dense activations clamp to
        # HIDDEN_ACT_CEIL = 127/128 (not 1.0). Each activation below is computed
        # continuously first, then snapped to its grid, exactly like
        # model._fake_quantize_acts's hard value:
        #   us_acc == them_acc == ft_bias == [1, 0, 1, 0] (ft_weight is all zero)
        #   ft crelu halves = clip([1,0], 0, 255/256) = [0.99609375, 0.0]
        #   ft_out, pre-floor = [0.99609375**2, 0, 0.99609375**2, 0]
        #                     = [0.9921875152587890625, 0.0, 0.9921875152587890625, 0.0]
        #   ft_out, floored   = [floor(0.99218752*128)/128, 0.0, ...] = [127/128, 0, 127/128, 0]
        #            (0.99218752*128 = 127.00047..., floors to 127 -- lands exactly
        #             on HIDDEN_ACT_CEIL's own integer maximum, no extra clip needed)
        #   fc0_out = ft_out @ fc0_weight + fc0_bias = [127/128 - 1.3, 1.5]
        #           = [-0.3078125, 1.5]                                  (RAW, not quantized)
        #   skip = fc0_out[0] - fc0_out[1] = -1.8078125                  (RAW, never quantized)
        #   a0, pre-floor = [clip((-0.3078125)**2,0,127/128), clip(1.5**2,0,127/128),
        #                    clip(-0.3078125,0,127/128), clip(1.5,0,127/128)]
        #                 = [0.09474863..., 127/128, 0.0, 127/128]
        #   a0, floored   = [floor(0.09474863*128)/128, 127/128, 0.0, 127/128]
        #                 = [12/128, 127/128, 0.0, 127/128] = [0.09375, 0.9921875, 0.0, 0.9921875]
        #   fc1_out = a0 @ fc1_weight + fc1_bias = [0.09375 - 1.0, 0.0 + 1.2]
        #           = [-0.90625, 1.2]                                    (RAW)
        #   a1, pre-floor = [clip((-0.90625)**2,0,127/128), clip(1.2**2,0,127/128),
        #                    clip(-0.90625,0,127/128), clip(1.2,0,127/128)]
        #                 = [0.8212890625, 127/128, 0.0, 127/128]
        #   a1, floored   = [floor(0.8212890625*128)/128, 127/128, 0.0, 127/128]
        #                 = [105/128, 127/128, 0.0, 127/128] = [0.8203125, 0.9921875, 0.0, 0.9921875]
        #   out = sum(concat(a0, a1)) * 1.0 + 5.0 + skip
        #       = (0.09375 + 0.9921875 + 0.0 + 0.9921875 + 0.8203125 + 0.9921875 + 0.0 + 0.9921875)
        #         + 5.0 - 1.8078125
        #       = 4.8828125 + 5.0 - 1.8078125 = 8.075
        self.assertAlmostEqual(outputs[0], 8.075, delta=1e-4)

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
                EXPECTED_NUM_BUCKETS,
                EXPECTED_L2_SIZE,
                EXPECTED_L3_SIZE,
                EXPECTED_FT_SIZE,
                EXPECTED_L2_SIZE * 2,
                EXPECTED_L2_SIZE * 2 + EXPECTED_L3_SIZE * 2,
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
        num_buckets = EXPECTED_NUM_BUCKETS
        l2_size = EXPECTED_L2_SIZE
        l3_size = EXPECTED_L3_SIZE
        fc1_input_size = l2_size * 2
        fc2_input_size = l2_size * 2 + l3_size * 2
        exported = ExportedNetwork(
            description="fixture",
            num_features=EXPECTED_NUM_FEATURES,
            ft_size=EXPECTED_FT_SIZE,
            num_buckets=num_buckets,
            l2_size=l2_size,
            l3_size=l3_size,
            ft_scale=255.0,
            fc0_scale=64.0,
            fc1_scale=64.0,
            fc2_scale=64.0,
            score_scale=1.0,
            ft_bias=np.zeros(EXPECTED_FT_SIZE, dtype=np.int16),
            ft_weight=np.zeros((EXPECTED_NUM_FEATURES, EXPECTED_FT_SIZE), dtype=np.int16),
            fc0_bias=np.zeros((num_buckets, l2_size), dtype=np.int32),
            fc0_weight=np.zeros((num_buckets, EXPECTED_FT_SIZE, l2_size), dtype=np.int8),
            fc1_bias=np.zeros((num_buckets, l3_size), dtype=np.int32),
            fc1_weight=np.zeros((num_buckets, fc1_input_size, l3_size), dtype=np.int8),
            fc2_bias=np.zeros(num_buckets, dtype=np.int32),
            fc2_weight=np.zeros((num_buckets, fc2_input_size), dtype=np.int8),
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

    def test_evaluate_export_matches_model_forward_for_bucketed_head(self) -> None:
        # This is the parity target for the engine: evaluate_export (the numpy
        # reference the engine's integer kernel is checked against) must match
        # model.forward (the float training-time reference) to quantization
        # precision, for the new 8-bucket / widened-head architecture.
        from thrawn_nnue.model import HalfKAv2HmNNUE

        torch.manual_seed(1234)
        # Full QAT (the defaults): evaluate_export unconditionally floors
        # activations onto the uint8 grid (it's the engine's parity target,
        # and the engine always does this -- see evaluate_export's
        # docstring), so it only matches model.forward() to near machine
        # precision when the model's own forward pass floors them the same
        # way. This is deliberately the realistic configuration, not QAT
        # disabled: QAT-off's real, nonzero discrepancy against this same
        # evaluate_export is covered by QuantizationAwareTrainingExportParityTests
        # below, and is the intended, useful signal (see
        # docs/nnue_spec.md's Quantization-aware training section).
        model = HalfKAv2HmNNUE()
        # All buckets start from an identical repeated template (see
        # HalfKAv2HmNNUE._init_stacked_layer_uniformly); randomize the biases
        # per bucket so this test actually exercises bucket routing instead of
        # every bucket producing the same output. Biases are stored
        # StackedLinear-style (flat, width out_per_bucket * num_buckets), so
        # reshape to (num_buckets, out_per_bucket) to address them per-bucket.
        with torch.no_grad():
            fc0_bias = model.fc0.bias.view(model.num_buckets, model.l2_size)
            fc1_bias = model.fc1.bias.view(model.num_buckets, model.l3_size)
            fc2_bias = model.fc2.bias.view(model.num_buckets, 1)
            for bucket in range(model.num_buckets):
                fc0_bias[bucket].uniform_(-0.2, 0.2)
                fc1_bias[bucket].uniform_(-0.2, 0.2)
                fc2_bias[bucket].uniform_(-0.2, 0.2)
        model.eval()

        fens = [
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b - - 0 1",  # 32 pieces -> bucket 7
            "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/2P5/PP3PPP/RNBQKBNR b - - 0 3",  # fewer pieces
            "8/2k5/8/8/8/8/5K2/8 w - - 0 1",  # 2 pieces -> bucket 0
        ]
        from thrawn_nnue.export import _batch_arrays_from_fens

        white_indices, black_indices, stm = _batch_arrays_from_fens(fens, max_active_features=32)
        with torch.no_grad():
            checkpoint_predictions = model(
                torch.from_numpy(white_indices).long(),
                torch.from_numpy(black_indices).long(),
                torch.from_numpy(stm).float().unsqueeze(1),
            ).reshape(-1).tolist()

        from thrawn_nnue.config import TrainConfig

        config = TrainConfig(
            datasets=["/tmp/train.binpack"],
            max_epochs=1,
            epoch_size=1,
            nnue2score=600.0,
        )
        config.validate()
        exported = _exported_network_from_model(model, config)
        exported_predictions = evaluate_export(exported, fens)

        checkpoint_scores = [value * config.nnue2score for value in checkpoint_predictions]
        abs_errors = [abs(a - b) for a, b in zip(checkpoint_scores, exported_predictions, strict=True)]
        max_abs_error = max(abs_errors)
        # Near machine precision: full QAT means model.forward and
        # evaluate_export compute the same function (weights rounded the
        # same way, activations floored the same way), so any structural
        # mismatch in bucket selection / skip / activation concat order would
        # show up as a large error here, not just float32 noise.
        self.assertLess(max_abs_error, 1e-3, msg=f"abs errors: {abs_errors}")


@unittest.skipUnless(torch is not None, "PyTorch is required for QAT export-parity tests")
class QuantizationAwareTrainingExportParityTests(unittest.TestCase):
    """QAT's whole point: the checkpoint's forward pass and evaluate_export's
    dequantized-weight float math should compute very nearly the same
    function once fake-quantization is on, instead of the export step
    introducing uncompensated rounding error (see docs/nnue_spec.md).

    evaluate_export unconditionally floors activations onto the uint8 grid
    (it's the engine's parity target, and the engine always does this), so
    the collapse-to-near-zero property only holds for the *full* QAT
    configuration (weight + activation), not weight-QAT alone: with
    activation QAT off, model.forward() never floors its activations while
    evaluate_export still does, so a real, non-vacuous discrepancy is the
    correct outcome -- the verifier telling the user their float model won't
    match the engine.

    A default-init model's weights are tiny and near the quantization grid
    already, which lets floor-quantization coincidentally absorb small
    weight-rounding differences and mask the properties under test. These
    tests instead run a short, fast synthetic training loop first to reach
    weight magnitudes representative of a real checkpoint.
    """

    def _default_config(self):
        from thrawn_nnue.config import TrainConfig

        config = TrainConfig(datasets=["/tmp/train.binpack"], max_epochs=1, epoch_size=1, nnue2score=600.0)
        config.validate()
        return config

    def _briefly_trained_model(self, config):
        from thrawn_nnue.model import HalfKAv2HmNNUE

        torch.manual_seed(0)
        model = HalfKAv2HmNNUE()  # full QAT defaults while "training"
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        dense_limit = (127.0 - 0.5) / max(config.export_dense_scale, 1.0)

        batch_size = 256
        max_active = 32
        num_features = config.num_features
        torch.manual_seed(1)
        for _ in range(60):
            white = torch.full((batch_size, max_active), -1, dtype=torch.long)
            black = torch.full((batch_size, max_active), -1, dtype=torch.long)
            counts = torch.randint(2, max_active + 1, (batch_size,))
            for row in range(batch_size):
                n = int(counts[row])
                white[row, :n] = torch.randint(0, num_features, (n,))
                black[row, :n] = torch.randint(0, num_features, (n,))
            stm = torch.randint(0, 2, (batch_size, 1)).float()
            target = torch.randn(batch_size, 1) * 3.0

            optimizer.zero_grad(set_to_none=True)
            pred = model(white, black, stm)
            loss = (pred - target).square().mean()
            loss.backward()
            optimizer.step()
            # Mirrors training._clip_model_weights.
            with torch.no_grad():
                model.fc0.weight.clamp_(-dense_limit, dense_limit)
                model.fc1.weight.clamp_(-dense_limit, dense_limit)
                model.fc2.weight.clamp_(-dense_limit, dense_limit)
        return model

    def _max_abs_error(self, model, config, **flags) -> float:
        for name, value in flags.items():
            setattr(model, name, value)
        model.eval()
        fens = DEFAULT_VERIFICATION_FENS
        white_indices, black_indices, stm = _batch_arrays_from_fens(fens, max_active_features=32)
        with torch.no_grad():
            predictions = model(
                torch.from_numpy(white_indices).long(),
                torch.from_numpy(black_indices).long(),
                torch.from_numpy(stm).float().unsqueeze(1),
            ).reshape(-1).tolist()
        scores = [value * config.nnue2score for value in predictions]
        exported = _exported_network_from_model(model, config)
        exported_scores = evaluate_export(exported, fens)
        abs_errors = [abs(a - b) for a, b in zip(scores, exported_scores, strict=True)]
        return max(abs_errors)

    def test_full_qat_collapses_export_parity_to_near_machine_precision(self) -> None:
        config = self._default_config()
        model = self._briefly_trained_model(config)

        no_qat_error = self._max_abs_error(
            model,
            config,
            use_fake_weight_quantization=False,
            use_fake_act_quantization=False,
            use_fake_ft_weight_quantization=False,
        )
        full_qat_error = self._max_abs_error(
            model,
            config,
            use_fake_weight_quantization=True,
            use_fake_act_quantization=True,
            use_fake_ft_weight_quantization=True,
        )

        # Baseline (no QAT) is real, substantial export rounding error on a
        # briefly-trained model -- this isn't a vacuous comparison against an
        # already-zero baseline.
        self.assertGreater(no_qat_error, 1.0, msg=f"unexpectedly tiny no-QAT baseline: {no_qat_error}")
        # Full QAT: model.forward() and evaluate_export now round weights and
        # floor activations identically, so they compute the same function up
        # to float32 rounding -- the export-time error should collapse close
        # to zero, not just shrink.
        self.assertLess(
            full_qat_error,
            1e-3,
            msg=f"full QAT did not collapse export parity to near machine precision: {full_qat_error}",
        )
        self.assertLess(full_qat_error, no_qat_error / 1000.0)

    def test_activation_qat_off_leaves_real_discrepancy_against_evaluate_export(self) -> None:
        # This is the deliberate flip side of the test above: evaluate_export
        # always floors activations (it must, to match the engine), so a
        # checkpoint trained with use_fake_act_quantization=False will not
        # match it -- and should not appear to. That mismatch is useful
        # signal (the float model genuinely won't match the engine), not a
        # verifier bug.
        config = self._default_config()
        model = self._briefly_trained_model(config)

        weight_qat_only_error = self._max_abs_error(
            model,
            config,
            use_fake_weight_quantization=True,
            use_fake_act_quantization=False,
            use_fake_ft_weight_quantization=True,
        )

        self.assertGreater(weight_qat_only_error, 1.0)
        # And it must be bounded, not blown up by some interaction bug (an
        # order-of-magnitude sanity check, not a tight bound).
        self.assertLess(weight_qat_only_error, 1000.0)

    def test_export_raises_loudly_instead_of_silently_rescaling_on_scale_mismatch(self) -> None:
        # Integration-level companion to the unit test on
        # _fit_quantization_scale directly: a weight that _clip_model_weights
        # would normally prevent (but this model was never trained, so
        # nothing clipped it) must make the export step fail loudly rather
        # than quietly using a smaller scale than QAT simulated.
        from thrawn_nnue.model import HalfKAv2HmNNUE

        config = self._default_config()
        model = HalfKAv2HmNNUE()
        with torch.no_grad():
            model.fc0.weight[0, 0] = 5.0  # 5.0 * export_dense_scale(64) = 320, overflows int8
        model.eval()

        with self.assertRaises(QuantizationScaleMismatchError):
            _exported_network_from_model(model, config)


if __name__ == "__main__":
    unittest.main()
