from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any

import numpy as np

from .board import BoardState
from .features import FEATURES, HALFKA_V2_HM_NUM_FEATURES, active_feature_indices


# v10 -> v11: switched the quantization scheme to Stockfish's shift-exact
# constants (ft_scale 255 -> 256, activation "one" 127 -> 128; see
# docs/nnue_spec.md's Quantization section). The struct layout is byte-for-
# byte unchanged, but the activation-clamp semantics a loader must apply
# aren't encoded in the header, so a v10 file must never be silently read as
# v11 (or vice versa) -- the version bump is the only guard against that.
MAGIC = b"THNNUE\x00\x01"
VERSION = 11
FEATURE_SET_ID = "HalfKAv2_hm"
OUTPUT_PERSPECTIVE_STM = 1
EXPECTED_NUM_FEATURES = HALFKA_V2_HM_NUM_FEATURES
EXPECTED_FT_SIZE = 1024
EXPECTED_NUM_BUCKETS = 8
EXPECTED_L2_SIZE = 32
EXPECTED_L3_SIZE = 32
# Pairwise SqrCReLU on the FT output feeds fc0 an activation of width ft_size
# (not ft_size * 2): each perspective's accumulator halves are multiplied together.
EXPECTED_FC0_INPUT_SIZE = EXPECTED_FT_SIZE
EXPECTED_FC1_INPUT_SIZE = EXPECTED_L2_SIZE * 2
# The output head sees both dense layers' squared+linear activations.
EXPECTED_FC2_INPUT_SIZE = EXPECTED_L2_SIZE * 2 + EXPECTED_L3_SIZE * 2
MAX_DESCRIPTION_BYTES = 1_000_000
STOCKFISH_INTERNAL_TO_CP = 100.0 / 208.0
HEADER_PREFIX_STRUCT = struct.Struct("<8sI")
HEADER_REST_STRUCT = struct.Struct("<16sIIIIIIIIIfffffI")
DEFAULT_VERIFICATION_FENS = [
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b - - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/2P5/PP3PPP/RNBQKBNR b - - 0 3",
    "8/2k5/8/8/8/8/5K2/8 w - - 0 1",
]


@dataclass(slots=True)
class ExportedNetwork:
    description: str
    num_features: int
    ft_size: int
    num_buckets: int
    l2_size: int
    l3_size: int
    ft_scale: float
    fc0_scale: float
    fc1_scale: float
    fc2_scale: float
    score_scale: float
    ft_bias: np.ndarray
    ft_weight: np.ndarray
    fc0_bias: np.ndarray  # (num_buckets, l2_size)
    fc0_weight: np.ndarray  # (num_buckets, fc0_input_size, l2_size)
    fc1_bias: np.ndarray  # (num_buckets, l3_size)
    fc1_weight: np.ndarray  # (num_buckets, fc1_input_size, l3_size)
    fc2_bias: np.ndarray  # (num_buckets,)
    fc2_weight: np.ndarray  # (num_buckets, fc2_input_size)
    version: int = VERSION

    @property
    def fc0_input_size(self) -> int:
        # Pairwise SqrCReLU halves the raw us||them concat (ft_size * 2) to ft_size.
        return self.ft_size

    @property
    def fc1_input_size(self) -> int:
        return self.l2_size * 2

    @property
    def fc2_input_size(self) -> int:
        # The output head sees both dense layers' squared+linear activations.
        return self.l2_size * 2 + self.l3_size * 2


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for export commands") from exc
    return torch


def export_checkpoint(checkpoint_path: str | Path, output_path: str | Path) -> Path:
    from .checkpoint import load_checkpoint
    from .config import TrainConfig
    from .model import HalfKAv2HmNNUE

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = TrainConfig.from_dict(dict(checkpoint["config"]))
    model = HalfKAv2HmNNUE(
        num_features=config.num_features,
        ft_size=config.ft_size,
        l2_size=config.l2_size,
        l3_size=config.l3_size,
        num_buckets=config.num_buckets,
        export_ft_scale=config.export_ft_scale,
        export_dense_scale=config.export_dense_scale,
        use_fake_weight_quantization=config.use_fake_weight_quantization,
        use_fake_act_quantization=config.use_fake_act_quantization,
        use_fake_ft_weight_quantization=config.use_fake_ft_weight_quantization,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    exported = _exported_network_from_model(model, config)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as handle:
        _write_export(handle, exported)
    return output


def load_export(path: str | Path) -> ExportedNetwork:
    with Path(path).open("rb") as handle:
        raw_prefix = _read_exact(handle, HEADER_PREFIX_STRUCT.size, "header prefix")
        magic, version = HEADER_PREFIX_STRUCT.unpack(raw_prefix)

        if magic != MAGIC:
            raise ValueError("Unexpected .nnue magic")
        if version != VERSION:
            raise ValueError(f"Unsupported .nnue version: {version}")

        raw_rest = _read_exact(handle, HEADER_REST_STRUCT.size, "header body")
        (
            feature_set,
            num_features,
            ft_size,
            num_buckets,
            l2_size,
            l3_size,
            fc0_input_size,
            fc1_input_size,
            fc2_input_size,
            output_perspective,
            ft_scale,
            fc0_scale,
            fc1_scale,
            fc2_scale,
            score_scale,
            description_length,
        ) = HEADER_REST_STRUCT.unpack(raw_rest)

        if feature_set.rstrip(b"\x00").decode("ascii") != FEATURE_SET_ID:
            raise ValueError("Unexpected feature-set identifier")
        _validate_export_header(
            num_features=num_features,
            ft_size=ft_size,
            num_buckets=num_buckets,
            l2_size=l2_size,
            l3_size=l3_size,
            fc0_input_size=fc0_input_size,
            fc1_input_size=fc1_input_size,
            fc2_input_size=fc2_input_size,
            output_perspective=output_perspective,
            ft_scale=ft_scale,
            fc0_scale=fc0_scale,
            fc1_scale=fc1_scale,
            fc2_scale=fc2_scale,
            score_scale=score_scale,
            description_length=description_length,
        )

        description = _read_exact(handle, description_length, "description").decode("utf-8")
        ft_bias = np.frombuffer(_read_exact(handle, ft_size * 2, "ft_bias"), dtype="<i2").copy()
        ft_weight = np.frombuffer(
            _read_exact(handle, num_features * ft_size * 2, "ft_weight"),
            dtype="<i2",
        ).copy().reshape(num_features, ft_size)

        fc0_bias_buckets = []
        fc0_weight_buckets = []
        fc1_bias_buckets = []
        fc1_weight_buckets = []
        fc2_bias_buckets = []
        fc2_weight_buckets = []
        for bucket in range(num_buckets):
            fc0_bias_buckets.append(
                np.frombuffer(_read_exact(handle, l2_size * 4, f"fc0_bias[{bucket}]"), dtype="<i4").copy()
            )
            fc0_weight_buckets.append(
                np.frombuffer(
                    _read_exact(handle, fc0_input_size * l2_size, f"fc0_weight[{bucket}]"),
                    dtype=np.int8,
                ).copy().reshape(fc0_input_size, l2_size)
            )
            fc1_bias_buckets.append(
                np.frombuffer(_read_exact(handle, l3_size * 4, f"fc1_bias[{bucket}]"), dtype="<i4").copy()
            )
            fc1_weight_buckets.append(
                np.frombuffer(
                    _read_exact(handle, fc1_input_size * l3_size, f"fc1_weight[{bucket}]"),
                    dtype=np.int8,
                ).copy().reshape(fc1_input_size, l3_size)
            )
            fc2_bias_buckets.append(
                np.frombuffer(_read_exact(handle, 4, f"fc2_bias[{bucket}]"), dtype="<i4").copy()
            )
            fc2_weight_buckets.append(
                np.frombuffer(
                    _read_exact(handle, fc2_input_size, f"fc2_weight[{bucket}]"),
                    dtype=np.int8,
                ).copy()
            )

        if handle.read(1):
            raise ValueError("Unexpected trailing data in .nnue export")
        return ExportedNetwork(
            description=description,
            version=version,
            num_features=num_features,
            ft_size=ft_size,
            num_buckets=num_buckets,
            l2_size=l2_size,
            l3_size=l3_size,
            ft_scale=ft_scale,
            fc0_scale=fc0_scale,
            fc1_scale=fc1_scale,
            fc2_scale=fc2_scale,
            score_scale=score_scale,
            ft_bias=ft_bias,
            ft_weight=ft_weight,
            fc0_bias=np.stack(fc0_bias_buckets, axis=0),
            fc0_weight=np.stack(fc0_weight_buckets, axis=0),
            fc1_bias=np.stack(fc1_bias_buckets, axis=0),
            fc1_weight=np.stack(fc1_weight_buckets, axis=0),
            fc2_bias=np.concatenate(fc2_bias_buckets, axis=0),
            fc2_weight=np.stack(fc2_weight_buckets, axis=0),
        )


def _read_exact(handle, size: int, label: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"File ended while reading {label}: expected {size} bytes, got {len(data)}")
    return data


def _validate_export_header(
    *,
    num_features: int,
    ft_size: int,
    num_buckets: int,
    l2_size: int,
    l3_size: int,
    fc0_input_size: int,
    fc1_input_size: int,
    fc2_input_size: int,
    output_perspective: int,
    ft_scale: float,
    fc0_scale: float,
    fc1_scale: float,
    fc2_scale: float,
    score_scale: float,
    description_length: int,
) -> None:
    if num_features != EXPECTED_NUM_FEATURES:
        raise ValueError(f"Unexpected num_features: {num_features}")
    expected_sizes = {
        "ft_size": EXPECTED_FT_SIZE,
        "num_buckets": EXPECTED_NUM_BUCKETS,
        "l2_size": EXPECTED_L2_SIZE,
        "l3_size": EXPECTED_L3_SIZE,
    }
    actual_sizes = {
        "ft_size": ft_size,
        "num_buckets": num_buckets,
        "l2_size": l2_size,
        "l3_size": l3_size,
    }
    for name, expected in expected_sizes.items():
        if actual_sizes[name] != expected:
            raise ValueError(f"{name} must be {expected}")
    if fc0_input_size != ft_size:
        raise ValueError("fc0_input_size must equal ft_size (pairwise SqrCReLU FT activation)")
    if fc1_input_size != l2_size * 2:
        raise ValueError("fc1_input_size must equal l2_size * 2")
    if fc2_input_size != l2_size * 2 + l3_size * 2:
        raise ValueError("fc2_input_size must equal l2_size * 2 + l3_size * 2 (widened output head)")
    if output_perspective != OUTPUT_PERSPECTIVE_STM:
        raise ValueError("Only side-to-move exports are supported")
    for name, scale in (
        ("ft_scale", ft_scale),
        ("fc0_scale", fc0_scale),
        ("fc1_scale", fc1_scale),
        ("fc2_scale", fc2_scale),
        ("score_scale", score_scale),
    ):
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if description_length > MAX_DESCRIPTION_BYTES:
        raise ValueError(f"description_length is too large: {description_length}")


def verify_export(
    checkpoint_path: str | Path,
    nnue_path: str | Path,
    fens: list[str] | None = None,
) -> dict[str, Any]:
    torch = _require_torch()
    from .checkpoint import load_checkpoint
    from .config import TrainConfig
    from .model import HalfKAv2HmNNUE

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = TrainConfig.from_dict(dict(checkpoint["config"]))
    model = HalfKAv2HmNNUE(
        num_features=config.num_features,
        ft_size=config.ft_size,
        l2_size=config.l2_size,
        l3_size=config.l3_size,
        num_buckets=config.num_buckets,
        export_ft_scale=config.export_ft_scale,
        export_dense_scale=config.export_dense_scale,
        use_fake_weight_quantization=config.use_fake_weight_quantization,
        use_fake_act_quantization=config.use_fake_act_quantization,
        use_fake_ft_weight_quantization=config.use_fake_ft_weight_quantization,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    fens = fens or DEFAULT_VERIFICATION_FENS
    with torch.no_grad():
        white_indices, black_indices, stm = _batch_arrays_from_fens(
            fens,
            max_active_features=config.max_active_features,
        )
        predictions = model(
            torch.from_numpy(white_indices).long(),
            torch.from_numpy(black_indices).long(),
            torch.from_numpy(stm).float().unsqueeze(1),
        )
        predictions = predictions * float(config.nnue2score)
        predictions = [float(value) for value in predictions.reshape(-1).cpu().tolist()]

    exported = load_export(nnue_path)
    if (
        exported.num_features != config.num_features
        or exported.ft_size != config.ft_size
        or exported.num_buckets != config.num_buckets
        or exported.l2_size != config.l2_size
        or exported.l3_size != config.l3_size
    ):
        raise ValueError("Export dimensions do not match checkpoint config")
    exported_predictions = evaluate_export(exported, fens)
    abs_errors = [abs(a - b) for a, b in zip(predictions, exported_predictions, strict=True)]

    report: dict[str, Any] = {
        "positions": len(fens),
        "max_abs_error": max(abs_errors) if abs_errors else 0.0,
        "mean_abs_error": (sum(abs_errors) / len(abs_errors)) if abs_errors else 0.0,
        "checkpoint_predictions": predictions,
        "exported_predictions": exported_predictions,
        "checkpoint_predictions_cp": [_score_to_cp(value) for value in predictions],
        "exported_predictions_cp": [_score_to_cp(value) for value in exported_predictions],
        "abs_errors": abs_errors,
        "abs_errors_cp": [_score_to_cp(value) for value in abs_errors],
        "score_units": "Stockfish internal score",
        "score_cp_formula": "score_cp = score * 100 / 208",
        "export_ft_scale": float(exported.ft_scale),
        "export_fc0_scale": float(exported.fc0_scale),
        "export_fc1_scale": float(exported.fc1_scale),
        "export_fc2_scale": float(exported.fc2_scale),
        "export_score_scale": float(exported.score_scale),
        "nnue2score": float(config.nnue2score),
        "quantization": _export_quantization_diagnostics(exported),
    }
    return report


def render_verify_report(report: dict[str, Any]) -> str:
    max_abs_error = float(report.get("max_abs_error", 0.0))
    mean_abs_error = float(report.get("mean_abs_error", 0.0))
    total_limit_hits, limit_hit_detail = _quantization_limit_hits(report.get("quantization", {}))
    lines = [
        "verify-export:",
        f"positions: {int(report.get('positions', 0))}",
        (
            "error: "
            f"max={max_abs_error:.6f} ({_score_to_cp(max_abs_error):.3f} cp) "
            f"mean={mean_abs_error:.6f} ({_score_to_cp(mean_abs_error):.3f} cp)"
        ),
        (
            "scales: "
            f"ft={float(report.get('export_ft_scale', 0.0)):.6g} "
            f"fc0={float(report.get('export_fc0_scale', 0.0)):.6g} "
            f"fc1={float(report.get('export_fc1_scale', 0.0)):.6g} "
            f"fc2={float(report.get('export_fc2_scale', 0.0)):.6g} "
            f"score={float(report.get('export_score_scale', 0.0)):.6g}"
        ),
        _format_limit_hits(total_limit_hits, limit_hit_detail),
    ]
    return "\n".join(lines)


def bucket_for_piece_count(piece_count: int, num_buckets: int) -> int:
    """Stockfish's piece-count LayerStacks bucket rule.

    ``piece_count`` is the number of active features for the white
    perspective (always includes both kings, so it ranges 2..32).
    """
    return min(max((piece_count - 1) // 4, 0), num_buckets - 1)


# Matches model.FT_ACT_QUANT_SCALE / FT_CRELU_CEIL / HIDDEN_ACT_CEIL.
# Duplicated rather than imported so evaluate_export -- pure numpy -- doesn't
# gain a hard torch dependency; the two must be kept in sync (see model.py's
# comment: with ft_one=256, this renormalization is an EXACT power-of-two
# shift, not an approximation the way the old ft_one=255 scheme was).
ACT_QUANT_SCALE = 128.0
# FT accumulator CReLU ceiling (each perspective's raw accumulator halves,
# BEFORE the pairwise multiply): [0, 255] int == [0, 255/256] float.
FT_CRELU_CEIL = 255.0 / 256.0
# Hidden dense-layer activation ceiling (fc0_out/fc1_out's sqr_crelu/crelu
# components): [0, 127] int == [0, 127/128] float.
HIDDEN_ACT_CEIL = 127.0 / 128.0


def _quantize_act(x: np.ndarray) -> np.ndarray:
    """Floor an activation onto the uint8 grid the engine's integer bitshift
    actually produces (mirrors model._fake_quantize_acts's hard value, minus
    the STE -- this is a pure numpy forward reference, no gradients).

    Applied unconditionally, never gated on the checkpoint's QAT flags: the
    real engine's u8 x i8 dense layers always floor their activations this
    way, so evaluate_export -- the parity target described in
    docs/nnue_spec.md's Reference Forward Pass -- always models that. If a
    checkpoint was trained with use_fake_act_quantization=False, this is
    supposed to surface a real discrepancy against evaluate_export: that's
    the verifier correctly reporting the float model won't match the engine,
    not a bug in the verifier.
    """
    return np.floor(x * ACT_QUANT_SCALE + 1e-5) / ACT_QUANT_SCALE


def evaluate_export(exported: ExportedNetwork, fens: list[str]) -> list[float]:
    ft_bias = exported.ft_bias.astype(np.float32) / exported.ft_scale
    ft_weight = exported.ft_weight.astype(np.float32) / exported.ft_scale
    fc0_bias = exported.fc0_bias.astype(np.float32) / exported.fc0_scale
    fc0_weight = exported.fc0_weight.astype(np.float32) / exported.fc0_scale
    fc1_bias = exported.fc1_bias.astype(np.float32) / exported.fc1_scale
    fc1_weight = exported.fc1_weight.astype(np.float32) / exported.fc1_scale
    fc2_bias = exported.fc2_bias.astype(np.float32) / exported.fc2_scale
    fc2_weight = exported.fc2_weight.astype(np.float32) / exported.fc2_scale

    results: list[float] = []
    for fen in fens:
        board = BoardState.from_fen(fen)
        white_features = active_feature_indices(board, "white", features=FEATURES)
        black_features = active_feature_indices(board, "black", features=FEATURES)
        white_acc = ft_bias + ft_weight[white_features].sum(axis=0)
        black_acc = ft_bias + ft_weight[black_features].sum(axis=0)
        if board.side_to_move == "w":
            us_acc, them_acc = white_acc, black_acc
        else:
            us_acc, them_acc = black_acc, white_acc
        # Pairwise SqrCReLU: clamp each perspective to the FT's own activation range
        # (FT_CRELU_CEIL = 255/256, not 1.0 -- the FT's u8 grid never reaches a true
        # 1.0) and multiply its two halves together. Mirrors model.ft_pairwise_screlu
        # so the exported int net matches the float checkpoint (fc0 input width =
        # ft_size). The product itself is not separately clamped -- its ceiling
        # (FT_CRELU_CEIL ** 2) floor-quantizes onto the same integer maximum as
        # HIDDEN_ACT_CEIL below, so no extra clipping loss is introduced.
        us_lo, us_hi = np.split(np.clip(us_acc, 0.0, FT_CRELU_CEIL), 2)
        them_lo, them_hi = np.split(np.clip(them_acc, 0.0, FT_CRELU_CEIL), 2)
        # Floor onto the uint8 grid -- the engine's (a*b) >> SHIFT step -- so
        # this reference models the actual integer pipeline fc0 consumes, not
        # a continuous approximation of it.
        ft_out = _quantize_act(np.concatenate([us_lo * us_hi, them_lo * them_hi], axis=0))

        # LayerStacks bucket selection: only one bucket's dense stack is
        # evaluated per position (see docs/nnue_spec.md Fast Inference Notes).
        bucket = bucket_for_piece_count(len(white_features), exported.num_buckets)

        # fc0's u8 x i8 dot product accumulates into int32 exactly (integer
        # multiply-accumulate introduces no rounding); the float
        # dequantized-weight matmul below reproduces that value exactly, so
        # fc0_out needs no quantization of its own.
        fc0_out = ft_out @ fc0_weight[bucket] + fc0_bias[bucket]
        # The skip connection reads the RAW pre-activation fc0_out's last two
        # lanes; those lanes are also fed through the activations below like
        # every other lane (nothing is reserved/excluded). Not quantized --
        # matches model.forward's skip and upstream's fake_quantize_skip_act
        # (deliberately an identity).
        skip = fc0_out[exported.l2_size - 2] - fc0_out[exported.l2_size - 1]
        # Square-then-clamp, NOT clamp-then-square: negatives survive as a
        # positive min(x**2, HIDDEN_ACT_CEIL), matching Stockfish's
        # SqrClippedReLU. Each component is floored onto the uint8 grid
        # separately, before concatenation -- mirrors model.forward's
        # per-component _quantize_act calls on sqr_crelu(fc0_out) /
        # crelu(fc0_out).
        a0 = np.concatenate(
            [
                _quantize_act(np.clip(fc0_out * fc0_out, 0.0, HIDDEN_ACT_CEIL)),
                _quantize_act(np.clip(fc0_out, 0.0, HIDDEN_ACT_CEIL)),
            ],
            axis=0,
        )

        fc1_out = a0 @ fc1_weight[bucket] + fc1_bias[bucket]
        a1 = np.concatenate(
            [
                _quantize_act(np.clip(fc1_out * fc1_out, 0.0, HIDDEN_ACT_CEIL)),
                _quantize_act(np.clip(fc1_out, 0.0, HIDDEN_ACT_CEIL)),
            ],
            axis=0,
        )

        head_in = np.concatenate([a0, a1], axis=0)
        out = head_in @ fc2_weight[bucket] + fc2_bias[bucket] + skip
        output = out * exported.score_scale
        results.append(float(output))
    return results


def _score_to_cp(value: float) -> float:
    return float(value) * STOCKFISH_INTERNAL_TO_CP


def _exported_network_from_model(model, config) -> ExportedNetwork:
    ft_weight = model.ft.weight.detach().cpu().numpy()
    ft_bias = model.ft_bias.detach().cpu().numpy()

    # The model stores each dense layer as ONE wide nn.Linear producing every
    # bucket's output (StackedLinear layout): weight is (out_per_bucket *
    # num_buckets, in_features), out-major like nn.Linear always is. Bucket
    # b's slice is rows [b*out_per_bucket : (b+1)*out_per_bucket]. Reshape
    # splits that leading dimension into (num_buckets, out_per_bucket, in)
    # without reordering data, then transpose to the export's input-major
    # layout (num_buckets, in, out_per_bucket).
    num_buckets = config.num_buckets
    fc0_weight = (
        model.fc0.weight.detach().cpu().numpy().reshape(num_buckets, config.l2_size, -1).transpose(0, 2, 1)
    )
    fc0_bias = model.fc0.bias.detach().cpu().numpy().reshape(num_buckets, config.l2_size)
    fc1_weight = (
        model.fc1.weight.detach().cpu().numpy().reshape(num_buckets, config.l3_size, -1).transpose(0, 2, 1)
    )
    fc1_bias = model.fc1.bias.detach().cpu().numpy().reshape(num_buckets, config.l3_size)
    # fc2 has exactly 1 output per bucket, so its stacked weight (num_buckets,
    # fc2_input_size) and bias (num_buckets,) are already the export shape.
    fc2_weight = model.fc2.weight.detach().cpu().numpy().reshape(num_buckets, -1)
    fc2_bias = model.fc2.bias.detach().cpu().numpy().reshape(num_buckets)
    score_scale = float(config.nnue2score)

    # One shared scale per layer type, fit over all buckets' weights together
    # (not per-bucket scales).
    ft_scale = _fit_quantization_scale(
        [ft_bias, ft_weight], config.export_ft_scale, np.int16, names=["ft_bias", "ft_weight"]
    )
    fc0_scale = _fit_quantization_scale([fc0_weight], config.export_dense_scale, np.int8, names=["fc0_weight"])
    fc1_scale = _fit_quantization_scale([fc1_weight], config.export_dense_scale, np.int8, names=["fc1_weight"])
    fc2_scale = _fit_quantization_scale([fc2_weight], config.export_dense_scale, np.int8, names=["fc2_weight"])

    return ExportedNetwork(
        description=config.export_description,
        version=VERSION,
        num_features=config.num_features,
        ft_size=config.ft_size,
        num_buckets=config.num_buckets,
        l2_size=config.l2_size,
        l3_size=config.l3_size,
        ft_scale=ft_scale,
        fc0_scale=fc0_scale,
        fc1_scale=fc1_scale,
        fc2_scale=fc2_scale,
        score_scale=score_scale,
        ft_bias=_quantize(ft_bias, ft_scale, np.int16),
        ft_weight=_quantize(ft_weight, ft_scale, np.int16),
        fc0_bias=_quantize(fc0_bias, fc0_scale, np.int32),
        fc0_weight=_quantize(fc0_weight, fc0_scale, np.int8),
        fc1_bias=_quantize(fc1_bias, fc1_scale, np.int32),
        fc1_weight=_quantize(fc1_weight, fc1_scale, np.int8),
        fc2_bias=_quantize(fc2_bias, fc2_scale, np.int32),
        fc2_weight=_quantize(fc2_weight, fc2_scale, np.int8),
    )


def _quantize(values: np.ndarray, scale: float, dtype) -> np.ndarray:
    info = np.iinfo(dtype)
    quantized = np.rint(values * scale)
    quantized = np.clip(quantized, info.min, info.max)
    return quantized.astype(dtype)


class QuantizationScaleMismatchError(ValueError):
    """Requested export scale would overflow the target dtype.

    QAT fake-quantizes the training-time forward pass at exactly the
    requested scale (``TrainConfig.export_ft_scale`` / ``export_dense_scale``,
    the same values threaded into the model in ``training.py`` /
    ``export.py``). If the export step silently reduced the scale to avoid
    int overflow, the exported net would use a different scale than the one
    training simulated -- silently invalidating QAT with no symptom short of
    a strength regression. Fix by retraining with tighter weight clipping
    (see ``_clip_model_weights`` in training.py) or a lower requested export
    scale, not by rescaling at export time.
    """


def _fit_quantization_scale(
    values: list[np.ndarray],
    requested_scale: float,
    dtype,
    *,
    names: list[str] | None = None,
) -> float:
    info = np.iinfo(dtype)
    per_tensor_max = [(float(np.max(np.abs(value))) if value.size > 0 else 0.0) for value in values]
    max_abs = max(per_tensor_max, default=0.0)
    if max_abs <= 0.0:
        return float(requested_scale)
    max_scale_without_limit_hits = max(float(info.max) - 0.5, 1.0) / max_abs
    headroom_scale = float(np.nextafter(max_scale_without_limit_hits, 0.0))
    if headroom_scale < requested_scale:
        offender_index = per_tensor_max.index(max_abs)
        offender = (
            names[offender_index] if names is not None and offender_index < len(names) else f"tensor[{offender_index}]"
        )
        # ft_weight/ft_bias are the one case this can realistically fire for:
        # unlike fc0/fc1/fc2 weight, they are never touched by
        # _clip_model_weights, so unbounded growth during training -- not
        # just an over-eager scale choice -- is a real possible cause.
        cause_hint = (
            " This tensor is never clamped by _clip_model_weights (only fc0/fc1/fc2 weight are), "
            "so check whether it has genuinely diverged during training, not just whether the "
            "requested scale is too high."
            if offender.startswith("ft")
            else " _clip_model_weights should keep this tensor in range every step; if this fires, "
            "check that clamp is still running before assuming the weights diverged."
        )
        raise QuantizationScaleMismatchError(
            f"{offender}: quantization scale mismatch -- requested scale {requested_scale!r} would "
            f"overflow {np.dtype(dtype).name} (max |{offender}|={max_abs!r}, needs scale <= "
            f"{headroom_scale!r} to fit without clipping). Training's QAT forward pass "
            "fake-quantizes at exactly the requested scale, so silently rescaling here would export "
            "a net using a different scale than the one training simulated. Fix by lowering "
            f"export_ft_scale / export_dense_scale in the config.{cause_hint}"
        )
    return float(requested_scale)


def _export_quantization_diagnostics(exported: ExportedNetwork) -> dict[str, dict[str, float]]:
    return {
        "ft_bias": _quantized_tensor_stats(exported.ft_bias),
        "ft_weight": _quantized_tensor_stats(exported.ft_weight),
        "fc0_bias": _quantized_tensor_stats(exported.fc0_bias),
        "fc0_weight": _quantized_tensor_stats(exported.fc0_weight),
        "fc1_bias": _quantized_tensor_stats(exported.fc1_bias),
        "fc1_weight": _quantized_tensor_stats(exported.fc1_weight),
        "fc2_bias": _quantized_tensor_stats(exported.fc2_bias),
        "fc2_weight": _quantized_tensor_stats(exported.fc2_weight),
    }


def _quantization_limit_hits(quantization: object) -> tuple[int, list[str]]:
    if not isinstance(quantization, dict):
        return 0, []
    total = 0
    detail: list[str] = []
    for name, stats in sorted(quantization.items()):
        if not isinstance(stats, dict):
            continue
        hits = int(float(stats.get("positive_limit_hits", 0.0)) + float(stats.get("negative_limit_hits", 0.0)))
        total += hits
        if hits:
            detail.append(f"{name}={hits}")
    return total, detail


def _format_limit_hits(total_limit_hits: int, detail: list[str]) -> str:
    if total_limit_hits == 0:
        return "quantization: limit_hits=0"
    return f"quantization: limit_hits={total_limit_hits} ({', '.join(detail)})"


def _quantized_tensor_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "count": 0.0,
            "max_abs_quantized": 0.0,
            "positive_limit_hits": 0.0,
            "negative_limit_hits": 0.0,
        }
    info = np.iinfo(values.dtype)
    return {
        "count": float(values.size),
        "max_abs_quantized": float(np.max(np.abs(values))),
        "positive_limit_hits": float(np.count_nonzero(values == info.max)),
        "negative_limit_hits": float(np.count_nonzero(values == info.min)),
    }


def _write_export(handle, exported: ExportedNetwork) -> None:
    feature_set_bytes = FEATURE_SET_ID.encode("ascii").ljust(16, b"\x00")
    description_bytes = exported.description.encode("utf-8")
    header = HEADER_PREFIX_STRUCT.pack(
        MAGIC,
        exported.version,
    ) + HEADER_REST_STRUCT.pack(
        feature_set_bytes,
        exported.num_features,
        exported.ft_size,
        exported.num_buckets,
        exported.l2_size,
        exported.l3_size,
        exported.fc0_input_size,
        exported.fc1_input_size,
        exported.fc2_input_size,
        OUTPUT_PERSPECTIVE_STM,
        float(exported.ft_scale),
        float(exported.fc0_scale),
        float(exported.fc1_scale),
        float(exported.fc2_scale),
        float(exported.score_scale),
        len(description_bytes),
    )
    handle.write(header)
    handle.write(description_bytes)
    handle.write(exported.ft_bias.astype("<i2").tobytes())
    handle.write(exported.ft_weight.astype("<i2").tobytes())
    # Bucket-major payload: all of bucket b's dense layers are contiguous, for
    # engine cache locality (only one bucket is read per position at runtime).
    for bucket in range(exported.num_buckets):
        handle.write(exported.fc0_bias[bucket].astype("<i4").tobytes())
        handle.write(exported.fc0_weight[bucket].astype(np.int8).tobytes())
        handle.write(exported.fc1_bias[bucket].astype("<i4").tobytes())
        handle.write(exported.fc1_weight[bucket].astype(np.int8).tobytes())
        handle.write(np.asarray([exported.fc2_bias[bucket]], dtype="<i4").tobytes())
        handle.write(exported.fc2_weight[bucket].astype(np.int8).tobytes())


def _batch_arrays_from_fens(
    fens: list[str],
    *,
    max_active_features: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    white_indices = []
    black_indices = []
    stm = []
    for fen in fens:
        board = BoardState.from_fen(fen)
        white = active_feature_indices(board, "white", features=FEATURES)
        black = active_feature_indices(board, "black", features=FEATURES)
        white_indices.append(white + [-1] * (max_active_features - len(white)))
        black_indices.append(black + [-1] * (max_active_features - len(black)))
        stm.append(1.0 if board.side_to_move == "w" else 0.0)
    return (
        np.asarray(white_indices, dtype=np.int32),
        np.asarray(black_indices, dtype=np.int32),
        np.asarray(stm, dtype=np.float32),
    )
