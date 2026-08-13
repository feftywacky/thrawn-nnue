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
class DedicatedSkipLaneTests(unittest.TestCase):
    """fc0's last lane is a DEDICATED skip lane: it bypasses the activations
    and fc1/fc2 entirely and is added straight onto the head output.

    All weight values below are chosen on the dense quantization grid
    (multiples of 1/64) so QAT's weight rounding, which is on by default, is a
    no-op and the arithmetic stays exactly hand-checkable.
    """

    def setUp(self) -> None:
        from thrawn_nnue.model import HalfKAv2HmNNUE

        self.model = HalfKAv2HmNNUE()
        self.white = self._white_indices_for_piece_count(6)
        self.black = self._white_indices_for_piece_count(6)
        self.stm = torch.tensor([[1.0]], dtype=torch.float32)

    def _white_indices_for_piece_count(self, piece_count: int) -> "torch.Tensor":
        max_active_features = 32
        row = [0] * piece_count + [-1] * (max_active_features - piece_count)
        return torch.tensor([row], dtype=torch.long)

    def test_fc0_reserves_exactly_one_lane_beyond_the_hidden_lanes(self) -> None:
        model = self.model
        self.assertEqual(model.fc0_output_size, model.hidden_size + model.forward_size)
        self.assertEqual(model.fc0.out_features, model.fc0_output_size)
        # Only the hidden lanes are activated, so fc1's input is 2 *
        # hidden_size -- not 2 * fc0_output_size.
        self.assertEqual(model.fc1.in_features, model.hidden_size * 2)

    def test_skip_lane_reaches_the_output_unactivated(self) -> None:
        # A NEGATIVE skip value is the point: if the lane were fed through
        # crelu/sqr_crelu like the hidden lanes, it could not arrive at the
        # output as a negative number.
        model = self.model
        with torch.no_grad():
            model.fc0.weight.zero_()
            model.fc0.bias.zero_()
            model.fc0.bias[model.hidden_size] = -0.5
            model.fc2.weight.zero_()
            model.fc2.bias.zero_()
        model.eval()

        with torch.no_grad():
            out = model(self.white, self.black, self.stm)

        self.assertAlmostEqual(float(out.item()), -0.5, places=6)

    def test_skip_lane_is_excluded_from_the_activation_path(self) -> None:
        # With the head live (non-zero fc2), perturbing ONLY the skip lane must
        # move the output by exactly that perturbation. If the lane also fed
        # a0 -- i.e. it were a shared lane rather than a dedicated one -- the
        # activation path would contribute an extra, different delta.
        model = self.model
        with torch.no_grad():
            model.fc0.weight.zero_()
            model.fc0.bias.fill_(0.5)
            model.fc0.bias[model.hidden_size] = 0.0
            model.fc2.weight.fill_(0.25)
            model.fc2.bias.zero_()
        model.eval()

        with torch.no_grad():
            baseline = float(model(self.white, self.black, self.stm).item())
            model.fc0.bias[model.hidden_size] = 0.25
            perturbed = float(model(self.white, self.black, self.stm).item())

        self.assertAlmostEqual(perturbed - baseline, 0.25, places=6)

    def test_output_head_sees_fc0s_activations_directly_not_just_fc1s(self) -> None:
        # The widened head takes concat(a0, a1). Zeroing the a1 half of fc2's
        # weights must still leave a live path from fc0's activations to the
        # output -- that path is what "wide head" means.
        model = self.model
        a0_width = model.hidden_size * 2
        self.assertEqual(model.fc2.in_features, a0_width + model.fc1_output_size * 2)

        with torch.no_grad():
            model.fc0.weight.zero_()
            model.fc0.bias.fill_(0.5)
            model.fc0.bias[model.hidden_size] = 0.0  # skip lane contributes nothing
            model.fc1.weight.zero_()
            model.fc1.bias.zero_()
            model.fc2.bias.zero_()
            model.fc2.weight.zero_()
        model.eval()

        with torch.no_grad():
            head_silent = float(model(self.white, self.black, self.stm).item())
            model.fc2.weight[:, :a0_width] = 0.25  # a0 half only; a1 half stays zero
            a0_only = float(model(self.white, self.black, self.stm).item())

        self.assertAlmostEqual(head_silent, 0.0, places=6)
        self.assertNotAlmostEqual(a0_only, 0.0, places=4)


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
