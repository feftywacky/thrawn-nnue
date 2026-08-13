from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None


def _require_torch():
    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for model and training commands")


if torch is not None:

    # The engine's fc0/fc1 activations (and the FT pairwise-product output
    # feeding fc0) are a uint8 whose "one" is 128 -- Stockfish's HiddenOneVal
    # (see docs/nnue_spec.md Fast Inference Notes). With ft_one = 256 (see
    # export_ft_scale below, and FT_CRELU_CEIL), the FT pairwise product
    # ``(a*b) >> 9`` with a,b in [0,255] renormalizes EXACTLY onto this grid:
    # ``255*255 >> 9 == 127`` and ``(256x * 256y) >> 9 == 128*x*y`` for any
    # float x,y in [0,1]. This replaces the old ft_one=255 scheme, whose
    # analogous renormalization only reached ~127.002 -- an unavoidable ~1 LSB
    # systematic bias once compounded through the SqrClippedReLU's ``>> 19``.
    # Every renormalization in the new scheme is an exact power-of-two shift.
    FT_ACT_QUANT_SCALE = 128.0

    # The FT accumulator's own CReLU (used on each perspective's raw
    # accumulator halves inside ft_pairwise_screlu, BEFORE the pairwise
    # multiply) clamps to the integer range [0, ft_one - 1] = [0, 255], i.e.
    # [0, 255/256] as a float -- the FT's u8 grid never reaches a true 1.0.
    # This is a DIFFERENT ceiling than FT_ACT_QUANT_SCALE above, which is the
    # "one" of the *output* grid the pairwise product (and the fc0/fc1
    # dense_activation components) quantize onto.
    FT_CRELU_CEIL = 255.0 / 256.0

    # Hidden dense-layer activation ceiling: the widest value the
    # FT_ACT_QUANT_SCALE=128-wide uint8 grid can represent is (128-1)/128 =
    # 127/128. Pleasant coincidence: FT_CRELU_CEIL**2 == 0.9921875152..., which
    # floor-quantizes (see _quantize_act below) onto the exact same integer
    # maximum (127) as this ceiling -- so the FT pairwise product never incurs
    # extra clipping loss beyond what its own halves' CReLU already applied.
    HIDDEN_ACT_CEIL = 127.0 / 128.0

    def _fake_quantize_weights(x: torch.Tensor, scale: float) -> torch.Tensor:
        """Straight-through estimator for weight quantization.

        Weights are ROUNDed at serialization time (``export._quantize`` uses
        ``np.rint``), so the forward value here is ``round(x * scale) /
        scale`` while the gradient passes straight through unchanged (the
        classic STE trick: ``hard.detach() + (x - x.detach())`` has forward
        value ``hard`` but local gradient 1 w.r.t. ``x``).

        The round is forced into float32 regardless of the caller's autocast
        state: at fp16, ``(x * scale).round()`` silently loses precision once
        ``x * scale`` exceeds ~2048 (fp16's integer-exact range), which would
        make the "quantized" value wrong instead of merely imprecise.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            hard = (x32 * scale).round() / scale
            fake = hard.detach() + (x32 - x32.detach())
        return fake.to(x.dtype)

    def _fake_quantize_acts(x: torch.Tensor, scale: float) -> torch.Tensor:
        """Straight-through estimator for activation quantization.

        Activations are produced by a bitshift in the engine (an integer
        right-shift is a FLOOR, not a round), so the forward value here is
        ``floor(x * scale + eps) / scale``. Same float32-forced STE
        construction as ``_fake_quantize_weights``; see its docstring.
        """
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            hard = (x32 * scale + 1e-5).floor() / scale
            fake = hard.detach() + (x32 - x32.detach())
        return fake.to(x.dtype)

    def crelu(x: torch.Tensor, ceil: float = HIDDEN_ACT_CEIL) -> torch.Tensor:
        """Clipped ReLU: ``clamp(x, 0, ceil)``.

        ``ceil`` defaults to ``HIDDEN_ACT_CEIL`` (127/128) for the hidden
        dense-layer CReLU component (``crelu(hidden)`` / ``crelu(fc1_out)``
        in ``forward`` below). ``ft_pairwise_screlu`` calls this with
        ``ceil=FT_CRELU_CEIL`` (255/256) instead, for the feature
        transformer's own per-perspective CReLU halves -- the two ceilings
        differ because they quantize onto different-width integer grids
        (``ft_one=256`` vs ``hidden_one=128``), even though both represent
        the same float value 1.0 as their unreachable upper bound.
        """
        return torch.clamp(x, 0.0, ceil)


    def sqr_crelu(x: torch.Tensor) -> torch.Tensor:
        """Square-then-clamp activation (Stockfish's SqrClippedReLU).

        This squares the *signed* pre-activation and then clamps to
        ``[0, HIDDEN_ACT_CEIL]`` (127/128), which is not the same as
        ``crelu(x) ** 2``: negative inputs produce a positive
        ``min(x**2, 127/128)`` here, whereas clamping first would zero them
        out. Stockfish's integer kernel does this with a saturating
        ``mulhi_epi16`` on the signed value, so the float reference must square
        before clamping to stay bit-parity with the engine.
        """
        return torch.clamp(x * x, 0.0, HIDDEN_ACT_CEIL)


    def ft_pairwise_screlu(us: torch.Tensor, them: torch.Tensor) -> torch.Tensor:
        """Stockfish-style feature-transformer output activation.

        Clamps the raw accumulator to the FT's own activation range
        ``[0, FT_CRELU_CEIL]`` (255/256, see ``FT_CRELU_CEIL`` above) and
        multiplies the two halves of each perspective together (a pairwise
        squared-ClippedReLU). This bounds the FT output so the engine can run the
        first dense layer as ``u8 x i8`` (``vpdpbusd`` / ``vdotq``) instead of the
        ``i16 x i16`` path forced by feeding a raw un-clamped accumulator into fc0,
        and it halves fc0's input width from ``ft_size * 2`` to ``ft_size``.

        ``us`` / ``them`` each have shape ``[batch, ft_size]``. For a perspective
        ``p`` split into low/high halves ``p_lo``/``p_hi`` of ``ft_size // 2``, the
        activation emits ``crelu(p_lo) * crelu(p_hi)``. The two perspective results
        are concatenated, so the output width is ``ft_size``. The product is not
        separately clamped: its ceiling ``FT_CRELU_CEIL ** 2`` floor-quantizes onto
        the same integer maximum as ``HIDDEN_ACT_CEIL`` (see ``_quantize_act`` /
        ``FT_ACT_QUANT_SCALE`` above), so no clipping loss is introduced here.
        """
        half = us.shape[-1] // 2
        us_lo, us_hi = torch.split(crelu(us, ceil=FT_CRELU_CEIL), half, dim=-1)
        them_lo, them_hi = torch.split(crelu(them, ceil=FT_CRELU_CEIL), half, dim=-1)
        return torch.cat([us_lo * us_hi, them_lo * them_hi], dim=-1)


    class HalfKAv2HmNNUE(nn.Module):
        """HalfKAv2_hm NNUE: feature transformer + a three-layer dense tail.

        The dense tail has one dedicated skip lane and a widened output head:

        - ``fc0`` produces ``hidden_size + forward_size`` (31 + 1 = 32) lanes.
          The first ``hidden_size`` lanes are the hidden activations; the last
          ``forward_size`` lane is a DEDICATED skip lane, excluded from the
          activations entirely and added straight onto the final output. This
          mirrors upstream nnue-pytorch's ``nn.Linear(..., L2 + 1)`` layout
          (and Stockfish's own ``L2 = 15`` (+1) = 16), where the total fc0
          output width is the SIMD-friendly power of two and one of those
          lanes is spent on the skip.
        - The output head (``fc2``) sees BOTH dense layers' squared+linear
          activations (``2 * hidden_size + 2 * fc1_output_size`` wide), not
          just fc1's.

        Quantization-aware training (QAT): when enabled, the forward pass
        fake-quantizes weights (round) and activations (floor) with
        straight-through estimators, so the model *is* the quantized network
        (this applies in both train and eval mode -- see
        ``_fake_quantize_weights`` / ``_fake_quantize_acts`` above). This
        mirrors upstream nnue-pytorch's ``model/quantize.py`` and means the
        float checkpoint's forward pass and the exported int net's
        dequantized-weight forward pass compute very nearly the same
        function, instead of the export step introducing uncompensated
        rounding error. ``use_fake_ft_weight_quantization`` gates only
        ``ft.weight`` (22528x1024 ~= 23M floats) separately from the other,
        cheap weight tensors because fake-quantizing it every forward pass is
        measurably expensive -- see docs/nnue_spec.md for the measured cost.
        """

        def __init__(
            self,
            *,
            num_features: int = 22_528,
            ft_size: int = 1024,
            hidden_size: int = 31,
            forward_size: int = 1,
            fc1_output_size: int = 32,
            export_ft_scale: float = 256.0,
            export_dense_scale: float = 64.0,
            use_fake_weight_quantization: bool = True,
            use_fake_act_quantization: bool = True,
            use_fake_ft_weight_quantization: bool = True,
        ):
            super().__init__()
            if forward_size != 1:
                raise ValueError("HalfKAv2_hm uses exactly one forward lane")
            if ft_size % 2 != 0:
                raise ValueError("ft_size must be even for the pairwise FT activation")

            self.num_features = num_features
            self.ft_size = ft_size
            self.hidden_size = hidden_size
            self.forward_size = forward_size
            self.fc1_output_size = fc1_output_size
            self.fc0_output_size = hidden_size + forward_size

            # Pairwise SqrCReLU on the FT output halves the fc0 input from
            # ft_size * 2 (raw us||them concat) to ft_size.
            self.fc0_input_size = ft_size
            self.fc1_input_size = hidden_size * 2
            # Widened output head: both dense layers' squared+linear activations.
            self.fc2_input_size = hidden_size * 2 + fc1_output_size * 2

            # QAT: same scales the exporter quantizes at (export.py reads
            # these from TrainConfig), plus flags for which tensors get
            # fake-quantized in the forward pass. use_fake_weight_quantization
            # covers ft_bias and the fc0/fc1/fc2 weight+bias pairs;
            # use_fake_ft_weight_quantization is a separate, independently
            # toggleable flag just for the large ft.weight embedding table.
            self.export_ft_scale = float(export_ft_scale)
            self.export_dense_scale = float(export_dense_scale)
            self.use_fake_weight_quantization = bool(use_fake_weight_quantization)
            self.use_fake_act_quantization = bool(use_fake_act_quantization)
            self.use_fake_ft_weight_quantization = bool(use_fake_ft_weight_quantization)

            self.ft = nn.Embedding(num_features, ft_size)
            self.ft_bias = nn.Parameter(torch.zeros(ft_size, dtype=torch.float32))
            self.fc0 = nn.Linear(self.fc0_input_size, self.fc0_output_size)
            self.fc1 = nn.Linear(self.fc1_input_size, fc1_output_size)
            self.fc2 = nn.Linear(self.fc2_input_size, 1)
            self.reset_parameters()

        def reset_parameters(self) -> None:
            nn.init.uniform_(self.ft.weight, -0.01, 0.01)
            nn.init.zeros_(self.ft_bias)
            nn.init.xavier_uniform_(self.fc0.weight)
            nn.init.zeros_(self.fc0.bias)
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def _quantized_ft_weight(self) -> torch.Tensor:
            """``ft.weight``, fake-quantized at ``export_ft_scale`` if enabled.

            Gated by its own flag (not ``use_fake_weight_quantization``)
            because this tensor is ~23M floats (92MB fp32): fake-quantizing it
            every forward pass is measurably expensive (see
            docs/nnue_spec.md), so it can be turned off independently of the
            other, cheap weight tensors.
            """
            if self.use_fake_ft_weight_quantization:
                return _fake_quantize_weights(self.ft.weight, self.export_ft_scale)
            return self.ft.weight

        def _quantized_ft_bias(self) -> torch.Tensor:
            if self.use_fake_weight_quantization:
                return _fake_quantize_weights(self.ft_bias, self.export_ft_scale)
            return self.ft_bias

        def _quantized_dense_params(self, layer: "nn.Linear") -> tuple[torch.Tensor, torch.Tensor]:
            """``layer``'s weight/bias, fake-quantized at ``export_dense_scale``.

            The exporter quantizes fc0/fc1/fc2 bias at the same scale as their
            layer's weight (see ``_exported_network_from_model`` in
            export.py), so the bias is fake-quantized here too, not left
            float.
            """
            if not self.use_fake_weight_quantization:
                return layer.weight, layer.bias
            weight = _fake_quantize_weights(layer.weight, self.export_dense_scale)
            bias = _fake_quantize_weights(layer.bias, self.export_dense_scale)
            return weight, bias

        def _quantize_act(self, x: torch.Tensor) -> torch.Tensor:
            if self.use_fake_act_quantization:
                return _fake_quantize_acts(x, FT_ACT_QUANT_SCALE)
            return x

        def _accumulate(
            self,
            indices: torch.Tensor,
            ft_weight: torch.Tensor,
            ft_bias: torch.Tensor,
        ) -> torch.Tensor:
            mask = indices.ge(0)
            clamped = indices.clamp_min(0)
            counts = mask.sum(dim=1, dtype=torch.long)
            offsets = torch.empty(counts.numel() + 1, dtype=torch.long, device=indices.device)
            offsets[0] = 0
            offsets[1:] = torch.cumsum(counts, dim=0)
            flat_indices = clamped[mask]
            if flat_indices.numel() == 0:
                return ft_bias.unsqueeze(0).expand(indices.shape[0], -1)

            acc = torch.nn.functional.embedding_bag(
                flat_indices,
                ft_weight,
                offsets,
                mode="sum",
                include_last_offset=True,
            )
            return acc + ft_bias

        def forward(
            self,
            white_indices: torch.Tensor,
            black_indices: torch.Tensor,
            stm: torch.Tensor,
        ) -> torch.Tensor:
            ft_weight = self._quantized_ft_weight()
            ft_bias = self._quantized_ft_bias()
            white_acc = self._accumulate(white_indices, ft_weight, ft_bias)
            black_acc = self._accumulate(black_indices, ft_weight, ft_bias)
            stm_bool = stm.ge(0.5)
            us = torch.where(stm_bool, white_acc, black_acc)
            them = torch.where(stm_bool, black_acc, white_acc)
            x = ft_pairwise_screlu(us, them)
            # The FT pairwise product output feeding fc0 -- quantize it,
            # matching the engine's uint8 activation (see FT_ACT_QUANT_SCALE).
            x = self._quantize_act(x)

            fc0_weight, fc0_bias = self._quantized_dense_params(self.fc0)
            fc0_out = torch.nn.functional.linear(x, fc0_weight, fc0_bias)  # [batch, fc0_output_size]

            # Dedicated-lane skip: fc0's LAST lane is reserved for the skip
            # connection and never activated -- it bypasses fc1/fc2 entirely
            # and is added straight onto the final output. The skip is
            # deliberately NOT quantized (upstream's fake_quantize_skip_act is
            # an identity), and it stays [batch, 1] so it broadcasts onto the
            # scalar head output without a reshape.
            hidden = fc0_out[:, : self.hidden_size]
            skip = fc0_out[:, self.hidden_size : self.fc0_output_size]

            a0 = torch.cat(
                [self._quantize_act(sqr_crelu(hidden)), self._quantize_act(crelu(hidden))], dim=1
            )  # width 2 * hidden_size

            fc1_weight, fc1_bias = self._quantized_dense_params(self.fc1)
            fc1_out = torch.nn.functional.linear(a0, fc1_weight, fc1_bias)  # [batch, fc1_output_size]

            a1 = torch.cat(
                [self._quantize_act(sqr_crelu(fc1_out)), self._quantize_act(crelu(fc1_out))], dim=1
            )  # width 2 * fc1_output_size

            head_in = torch.cat([a0, a1], dim=1)  # width fc2_input_size
            fc2_weight, fc2_bias = self._quantized_dense_params(self.fc2)
            out = torch.nn.functional.linear(head_in, fc2_weight, fc2_bias)  # [batch, 1]

            return out + skip

else:

    class HalfKAv2HmNNUE:  # pragma: no cover - exercised only when torch is missing
        def __init__(self, *args, **kwargs):
            _require_torch()
