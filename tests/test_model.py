from __future__ import annotations

import math
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipUnless(torch is not None, "PyTorch is required for model tests")
class BucketIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        from thrawn_nnue.model import HalfKAv2HmNNUE

        self.model = HalfKAv2HmNNUE(num_buckets=8)

    def _white_indices_for_piece_count(self, piece_count: int) -> "torch.Tensor":
        max_active_features = 32
        row = [0] * piece_count + [-1] * (max_active_features - piece_count)
        return torch.tensor([row], dtype=torch.long)

    def test_bucket_index_derivation_matches_stockfish_layerstacks_rule(self) -> None:
        # bucket = clamp((piece_count - 1) // 4, 0, num_buckets - 1); both kings
        # are always present so piece_count ranges 2..32.
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
                white_indices = self._white_indices_for_piece_count(piece_count)
                bucket = self.model._bucket_index(white_indices)
                self.assertEqual(int(bucket.item()), expected_bucket)

    def test_bucket_index_two_and_thirty_two_piece_edges(self) -> None:
        # The two edges the spec calls out explicitly: the minimum legal piece
        # count (both kings, nothing else) and the maximum (a full board).
        min_bucket = self.model._bucket_index(self._white_indices_for_piece_count(2))
        max_bucket = self.model._bucket_index(self._white_indices_for_piece_count(32))
        self.assertEqual(int(min_bucket.item()), 0)
        self.assertEqual(int(max_bucket.item()), self.model.num_buckets - 1)

    def test_different_piece_counts_route_to_different_buckets_and_outputs(self) -> None:
        # Two positions with different piece counts must (a) select different
        # buckets and (b) actually produce different scores once the buckets'
        # weights differ -- i.e. the bucket selection is really wired into the
        # forward pass, not just computed and discarded.
        max_active_features = 32
        few_pieces = 3  # -> bucket 0
        many_pieces = 32  # -> bucket 7
        white_few = self._white_indices_for_piece_count(few_pieces)
        white_many = self._white_indices_for_piece_count(many_pieces)
        black = torch.tensor([[0] + [-1] * (max_active_features - 1)], dtype=torch.long)
        stm = torch.tensor([[1.0]], dtype=torch.float32)

        bucket_few = int(self.model._bucket_index(white_few).item())
        bucket_many = int(self.model._bucket_index(white_many).item())
        self.assertNotEqual(bucket_few, bucket_many)

        with torch.no_grad():
            # Force every bucket's dense stack to be distinguishable. Biases
            # are stored StackedLinear-style: one flat vector of width
            # out_features_per_bucket * num_buckets, bucket b's slice is
            # rows [b*out : (b+1)*out] -- reshape to (num_buckets, out) to
            # address it per-bucket without hardcoding the stride.
            fc0_bias = self.model.fc0.bias.view(self.model.num_buckets, self.model.l2_size)
            fc1_bias = self.model.fc1.bias.view(self.model.num_buckets, self.model.l3_size)
            fc2_bias = self.model.fc2.bias.view(self.model.num_buckets, 1)
            for bucket in range(self.model.num_buckets):
                fc0_bias[bucket].fill_(float(bucket) * 0.1)
                fc1_bias[bucket].fill_(float(bucket) * 0.1)
                fc2_bias[bucket].fill_(float(bucket) * 0.1)

            score_few = self.model(white_few, black, stm)
            score_many = self.model(white_many, black, stm)

        self.assertNotAlmostEqual(float(score_few.item()), float(score_many.item()), places=4)


@unittest.skipUnless(torch is not None, "PyTorch is required for model tests")
class FakeQuantizeSTETests(unittest.TestCase):
    """The two STE primitives QAT is built on:

    - ``_fake_quantize_weights`` ROUNDs (matches export._quantize's np.rint,
      since weights are rounded at serialization time).
    - ``_fake_quantize_acts`` FLOORs (matches the engine's integer bitshift
      activation, ``(a*b) >> SHIFT``, which truncates rather than rounds).

    Both must (a) return the hard-quantized value on the forward pass and
    (b) pass gradients straight through unchanged (the STE property).
    """

    def setUp(self) -> None:
        from thrawn_nnue.model import _fake_quantize_acts, _fake_quantize_weights

        self.fake_quantize_weights = _fake_quantize_weights
        self.fake_quantize_acts = _fake_quantize_acts

    def test_fake_quantize_weights_rounds_the_forward_value(self) -> None:
        scale = 64.0
        x = torch.tensor([0.1, -0.1, 0.2734, -0.2734, 1.0], dtype=torch.float32)
        out = self.fake_quantize_weights(x, scale)
        expected = torch.round(x * scale) / scale
        self.assertTrue(torch.allclose(out, expected, atol=1e-7))

    def test_fake_quantize_weights_gradient_passes_through_unchanged(self) -> None:
        scale = 64.0
        x = torch.tensor([0.1, -0.1, 0.2734, -0.2734, 1.0], dtype=torch.float32, requires_grad=True)
        out = self.fake_quantize_weights(x, scale)
        out.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))

    def test_fake_quantize_acts_floors_the_forward_value(self) -> None:
        scale = 127.0
        x = torch.tensor([0.0, 0.3, 0.999, 1.0, 0.5000001], dtype=torch.float32)
        out = self.fake_quantize_acts(x, scale)
        expected = torch.floor(x * scale + 1e-5) / scale
        self.assertTrue(torch.allclose(out, expected, atol=1e-9))

    def test_fake_quantize_acts_floors_where_weights_would_round(self) -> None:
        # 0.997 * 127 = 126.619: round() -> 127, floor() -> 126. Confirms the
        # deliberate round/floor asymmetry (weights round at serialization;
        # activations floor via the engine's bitshift) actually takes effect
        # rather than both primitives quietly doing the same thing.
        scale = 127.0
        x = torch.tensor([0.997], dtype=torch.float32)
        rounded = self.fake_quantize_weights(x, scale)
        floored = self.fake_quantize_acts(x, scale)
        self.assertAlmostEqual(floored.item(), 126.0 / scale, places=6)
        self.assertAlmostEqual(rounded.item(), 127.0 / scale, places=6)
        self.assertGreater(rounded.item(), floored.item())

    def test_fake_quantize_acts_gradient_passes_through_unchanged(self) -> None:
        scale = 127.0
        x = torch.tensor([0.0, 0.3, 0.999, 0.5000001], dtype=torch.float32, requires_grad=True)
        out = self.fake_quantize_acts(x, scale)
        out.sum().backward()
        self.assertTrue(torch.allclose(x.grad, torch.ones_like(x)))

    def test_fake_quantize_weights_forces_float32_precision_under_fp16_autocast(self) -> None:
        # x * scale = 9.0001 * 255 = 2295.0255. fp16 is only integer-exact up
        # to 2048, so naive fp16 arithmetic rounds this to the wrong integer;
        # the fake-quantizer must force float32 math regardless of an active
        # autocast context, or the "quantized" value comes out silently wrong
        # (not just imprecise -- a different integer than the export step
        # would produce).
        scale = 255.0
        x = torch.tensor([9.0001], dtype=torch.float32)
        expected = round(9.0001 * 255.0) / 255.0

        naive_fp16 = (x.half() * scale).round() / scale
        self.assertNotAlmostEqual(naive_fp16.item(), expected, places=4)

        with torch.autocast(device_type="cpu", dtype=torch.float16, enabled=True):
            out = self.fake_quantize_weights(x, scale)
        self.assertEqual(out.item(), expected)

    def test_fake_quantize_acts_forces_float32_precision_under_fp16_autocast(self) -> None:
        scale = 127.0
        x = torch.tensor([17.0001], dtype=torch.float32)
        expected = math.floor(17.0001 * scale + 1e-5) / scale

        with torch.autocast(device_type="cpu", dtype=torch.float16, enabled=True):
            out = self.fake_quantize_acts(x, scale)
        self.assertEqual(out.item(), expected)


@unittest.skipUnless(torch is not None, "PyTorch is required for model tests")
class QuantizationFlagTests(unittest.TestCase):
    """The QAT config flags (use_fake_weight_quantization,
    use_fake_act_quantization, use_fake_ft_weight_quantization) must actually
    gate whether fake-quantization runs, independently of each other, and
    identically in train() and eval() mode."""

    def _white_indices_for_piece_count(self, piece_count: int, max_active_features: int = 32) -> "torch.Tensor":
        row = [0] * piece_count + [-1] * (max_active_features - piece_count)
        return torch.tensor([row], dtype=torch.long)

    def test_use_fake_weight_quantization_flag_toggles_dense_param_quantization(self) -> None:
        from thrawn_nnue.model import HalfKAv2HmNNUE

        model = HalfKAv2HmNNUE(use_fake_weight_quantization=False)
        with torch.no_grad():
            model.fc0.weight.fill_(0.01)  # 0.01 * 64 = 0.64, off the quantization grid
            model.fc0.bias.fill_(0.01)

        weight_off, bias_off = model._quantized_dense_params(model.fc0)
        self.assertIs(weight_off, model.fc0.weight)
        self.assertIs(bias_off, model.fc0.bias)

        model.use_fake_weight_quantization = True
        weight_on, bias_on = model._quantized_dense_params(model.fc0)
        expected = round(0.01 * model.export_dense_scale) / model.export_dense_scale
        self.assertTrue(torch.allclose(weight_on, torch.full_like(weight_on, expected)))
        self.assertTrue(torch.allclose(bias_on, torch.full_like(bias_on, expected)))
        self.assertFalse(torch.allclose(weight_on, model.fc0.weight))

    def test_use_fake_ft_weight_quantization_is_independent_of_use_fake_weight_quantization(self) -> None:
        from thrawn_nnue.model import HalfKAv2HmNNUE

        # General weight-quant flag ON, but the FT-weight-specific flag OFF:
        # ft.weight must stay untouched while ft_bias (governed by the
        # general flag) still gets quantized.
        model = HalfKAv2HmNNUE(use_fake_weight_quantization=True, use_fake_ft_weight_quantization=False)
        with torch.no_grad():
            model.ft.weight.fill_(0.01)  # 0.01 * 256 = 2.56, off the quantization grid
            model.ft_bias.fill_(0.01)

        ft_weight = model._quantized_ft_weight()
        ft_bias = model._quantized_ft_bias()
        self.assertIs(ft_weight, model.ft.weight)
        expected_bias = round(0.01 * model.export_ft_scale) / model.export_ft_scale
        self.assertTrue(torch.allclose(ft_bias, torch.full_like(ft_bias, expected_bias)))
        self.assertFalse(torch.allclose(ft_bias, model.ft_bias))

    def test_use_fake_act_quantization_flag_toggles_activation_quantization(self) -> None:
        from thrawn_nnue.model import HalfKAv2HmNNUE

        model = HalfKAv2HmNNUE(use_fake_act_quantization=False)
        x = torch.tensor([0.3, 0.6, 0.997])
        self.assertTrue(torch.equal(model._quantize_act(x), x))

        model.use_fake_act_quantization = True
        quantized = model._quantize_act(x)
        expected = torch.floor(x * 128.0 + 1e-5) / 128.0
        self.assertTrue(torch.allclose(quantized, expected))
        self.assertFalse(torch.allclose(quantized, x))

    def test_quantization_flags_apply_identically_in_train_and_eval_mode(self) -> None:
        # QAT is not a training-only trick: the model IS the quantized
        # network in both modes.
        from thrawn_nnue.model import HalfKAv2HmNNUE

        model = HalfKAv2HmNNUE(
            use_fake_weight_quantization=True,
            use_fake_ft_weight_quantization=False,
            use_fake_act_quantization=False,
        )
        with torch.no_grad():
            model.fc0.weight.fill_(0.01)
            model.fc0.bias.fill_(0.01)

        model.train()
        weight_train, bias_train = model._quantized_dense_params(model.fc0)
        model.eval()
        weight_eval, bias_eval = model._quantized_dense_params(model.fc0)

        self.assertTrue(torch.allclose(weight_train, weight_eval))
        self.assertTrue(torch.allclose(bias_train, bias_eval))
        expected = round(0.01 * model.export_dense_scale) / model.export_dense_scale
        self.assertTrue(torch.allclose(weight_eval, torch.full_like(weight_eval, expected)))

    def test_flags_change_end_to_end_forward_output(self) -> None:
        # Unit-level checks above confirm the gating logic in isolation; this
        # confirms the flags are actually wired into forward(), not just
        # available as dead attributes.
        from thrawn_nnue.model import HalfKAv2HmNNUE

        white = self._white_indices_for_piece_count(6)
        black = self._white_indices_for_piece_count(6)
        stm = torch.tensor([[1.0]])

        model = HalfKAv2HmNNUE()
        with torch.no_grad():
            # Push every fake-quantized tensor off its grid point so QAT is
            # guaranteed to change the forward value if (and only if) its
            # flag is enabled.
            model.ft.weight.fill_(0.01)
            model.ft_bias.fill_(0.01)
            for layer in (model.fc0, model.fc1, model.fc2):
                layer.weight.fill_(0.01)
                layer.bias.fill_(0.01)
        model.eval()

        def forward_with(**flags) -> "torch.Tensor":
            model.use_fake_weight_quantization = flags.get("use_fake_weight_quantization", False)
            model.use_fake_ft_weight_quantization = flags.get("use_fake_ft_weight_quantization", False)
            model.use_fake_act_quantization = flags.get("use_fake_act_quantization", False)
            with torch.no_grad():
                return model(white, black, stm).clone()

        baseline = forward_with()
        weight_on = forward_with(use_fake_weight_quantization=True, use_fake_ft_weight_quantization=True)
        act_on = forward_with(use_fake_act_quantization=True)

        self.assertFalse(torch.allclose(baseline, weight_on))
        self.assertFalse(torch.allclose(baseline, act_on))


if __name__ == "__main__":
    unittest.main()
