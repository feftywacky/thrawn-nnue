from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any

import numpy as np

from .board import BoardState
from .features import FEATURES, HALFKA_V2_HM_NUM_FEATURES, active_feature_indices


MAGIC = b"THNNUE\x00\x01"
VERSION = 8
FEATURE_SET_ID = "HalfKAv2_hm"
OUTPUT_PERSPECTIVE_STM = 1
EXPECTED_NUM_FEATURES = HALFKA_V2_HM_NUM_FEATURES
EXPECTED_FT_SIZE = 1024
EXPECTED_HIDDEN_SIZE = 31
EXPECTED_FORWARD_SIZE = 1
EXPECTED_FC1_OUTPUT_SIZE = 32
MAX_DESCRIPTION_BYTES = 1_000_000
STOCKFISH_INTERNAL_TO_CP = 100.0 / 208.0
MATERIAL_ORDER_MIN_GAP_CP = 20.0
HEADER_PREFIX_STRUCT = struct.Struct("<8sI")
HEADER_REST_STRUCT = struct.Struct("<16sIIIIIIIIfffffI")
DEFAULT_VERIFICATION_FENS = [
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b - - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/2P5/PP3PPP/RNBQKBNR b - - 0 3",
    "8/2k5/8/8/8/8/5K2/8 w - - 0 1",
]
STARTING_POSITION_SANITY = ("starting_position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1")
MATERIAL_LADDER_POSITIONS = [
    ("equal_material", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w - - 0 3"),
    ("white_up_pawn", "r1bqkb1r/ppp2ppp/2p2n2/8/4P3/8/PPPP1PPP/RNBQKB1R w KQkq - 0 5"),
    ("white_up_knight", "r1bqk1nr/pppp1ppp/2n5/4p3/4P3/2N3P1/PPPP1K1P/R1BQ1BNR w kq - 1 5"),
    ("white_up_rook", "r1bk1b1N/p1p1qBpp/1pnp1n2/4p3/4P3/8/PPPP1PPP/RNBQK2R w KQ - 2 8"),
    ("white_up_queen", "2r2b1r/pQ2p1pk/4bn1p/8/3n4/2N5/PPP2PPP/R1B2RK1 w - - 5 13"),
]
SANITY_POSITIONS = [STARTING_POSITION_SANITY, *MATERIAL_LADDER_POSITIONS]


@dataclass(slots=True)
class ExportedNetwork:
    description: str
    num_features: int
    ft_size: int
    hidden_size: int
    forward_size: int
    fc1_output_size: int
    ft_scale: float
    fc0_scale: float
    fc1_scale: float
    fc2_scale: float
    score_scale: float
    ft_bias: np.ndarray
    ft_weight: np.ndarray
    fc0_bias: np.ndarray
    fc0_weight: np.ndarray
    fc1_bias: np.ndarray
    fc1_weight: np.ndarray
    fc2_bias: np.ndarray
    fc2_weight: np.ndarray
    version: int = VERSION

    @property
    def fc0_output_size(self) -> int:
        return self.hidden_size + self.forward_size

    @property
    def fc1_input_size(self) -> int:
        return self.hidden_size * 2


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
        hidden_size=config.hidden_size,
        forward_size=config.forward_size,
        fc1_output_size=config.fc1_output_size,
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
            hidden_size,
            forward_size,
            fc0_output_size,
            fc1_input_size,
            fc1_output_size,
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
            hidden_size=hidden_size,
            forward_size=forward_size,
            fc0_output_size=fc0_output_size,
            fc1_input_size=fc1_input_size,
            fc1_output_size=fc1_output_size,
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
        fc0_bias = np.frombuffer(_read_exact(handle, fc0_output_size * 4, "fc0_bias"), dtype="<i4").copy()
        fc0_weight = np.frombuffer(
            _read_exact(handle, ft_size * 2 * fc0_output_size, "fc0_weight"),
            dtype=np.int8,
        ).copy().reshape(ft_size * 2, fc0_output_size)
        fc1_bias = np.frombuffer(_read_exact(handle, fc1_output_size * 4, "fc1_bias"), dtype="<i4").copy()
        fc1_weight = np.frombuffer(
            _read_exact(handle, fc1_input_size * fc1_output_size, "fc1_weight"),
            dtype=np.int8,
        ).copy().reshape(fc1_input_size, fc1_output_size)
        fc2_bias = np.frombuffer(_read_exact(handle, 4, "fc2_bias"), dtype="<i4").copy()
        fc2_weight = np.frombuffer(_read_exact(handle, fc1_output_size, "fc2_weight"), dtype=np.int8).copy()
        if handle.read(1):
            raise ValueError("Unexpected trailing data in .nnue export")
        return ExportedNetwork(
            description=description,
            version=version,
            num_features=num_features,
            ft_size=ft_size,
            hidden_size=hidden_size,
            forward_size=forward_size,
            fc1_output_size=fc1_output_size,
            ft_scale=ft_scale,
            fc0_scale=fc0_scale,
            fc1_scale=fc1_scale,
            fc2_scale=fc2_scale,
            score_scale=score_scale,
            ft_bias=ft_bias,
            ft_weight=ft_weight,
            fc0_bias=fc0_bias,
            fc0_weight=fc0_weight,
            fc1_bias=fc1_bias,
            fc1_weight=fc1_weight,
            fc2_bias=fc2_bias,
            fc2_weight=fc2_weight,
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
    hidden_size: int,
    forward_size: int,
    fc0_output_size: int,
    fc1_input_size: int,
    fc1_output_size: int,
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
        "hidden_size": EXPECTED_HIDDEN_SIZE,
        "forward_size": EXPECTED_FORWARD_SIZE,
        "fc1_output_size": EXPECTED_FC1_OUTPUT_SIZE,
    }
    actual_sizes = {
        "ft_size": ft_size,
        "hidden_size": hidden_size,
        "forward_size": forward_size,
        "fc1_output_size": fc1_output_size,
    }
    for name, expected in expected_sizes.items():
        if actual_sizes[name] != expected:
            raise ValueError(f"{name} must be {expected}")
    if fc0_output_size != hidden_size + forward_size:
        raise ValueError("fc0_output_size must equal hidden_size + forward_size")
    if fc1_input_size != hidden_size * 2:
        raise ValueError("fc1_input_size must equal hidden_size * 2")
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
    *,
    include_sanity: bool = False,
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
        hidden_size=config.hidden_size,
        forward_size=config.forward_size,
        fc1_output_size=config.fc1_output_size,
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
        or exported.hidden_size != config.hidden_size
        or exported.forward_size != config.forward_size
        or exported.fc1_output_size != config.fc1_output_size
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
    if include_sanity:
        report.update(_verify_material_sanity(model, exported, config, torch))
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
    if "material_ordering_ok" in report:
        lines.append(
            "sanity: "
            f"ordering={_pass_fail(bool(report.get('material_ordering_ok')))} "
            f"margins={_pass_fail(bool(report.get('material_margins_ok')))} "
            f"start_zero={_pass_fail(bool(report.get('starting_position_near_zero')))}"
        )
    return "\n".join(lines)


def _verify_material_sanity(model, exported: ExportedNetwork, config, torch) -> dict[str, Any]:
    sanity_fens = [fen for _, fen in SANITY_POSITIONS]
    with torch.no_grad():
        white_indices, black_indices, stm = _batch_arrays_from_fens(
            sanity_fens,
            max_active_features=config.max_active_features,
        )
        sanity_checkpoint = model(
            torch.from_numpy(white_indices).long(),
            torch.from_numpy(black_indices).long(),
            torch.from_numpy(stm).float().unsqueeze(1),
        )
        sanity_checkpoint = sanity_checkpoint * float(config.nnue2score)
        sanity_checkpoint_values = [float(value) for value in sanity_checkpoint.reshape(-1).cpu().tolist()]
    sanity_exported_values = evaluate_export(exported, sanity_fens)
    sanity_positions = [
        {
            "name": name,
            "fen": fen,
            "checkpoint_score": checkpoint_score,
            "exported_score": exported_score,
            "checkpoint_cp": _score_to_cp(checkpoint_score),
            "exported_cp": _score_to_cp(exported_score),
            "abs_error": abs(checkpoint_score - exported_score),
            "abs_error_cp": _score_to_cp(abs(checkpoint_score - exported_score)),
        }
        for (name, fen), checkpoint_score, exported_score in zip(
            SANITY_POSITIONS,
            sanity_checkpoint_values,
            sanity_exported_values,
            strict=True,
        )
    ]
    sanity_export_lookup = {item["name"]: float(item["exported_score"]) for item in sanity_positions}
    sanity_checkpoint_lookup = {item["name"]: float(item["checkpoint_score"]) for item in sanity_positions}
    material_ordering_ok = (
        sanity_export_lookup["equal_material"]
        < sanity_export_lookup["white_up_pawn"]
        < sanity_export_lookup["white_up_knight"]
        < sanity_export_lookup["white_up_rook"]
        < sanity_export_lookup["white_up_queen"]
    )
    exported_material_gaps_cp = _material_gaps_cp(sanity_export_lookup)
    checkpoint_material_gaps_cp = _material_gaps_cp(sanity_checkpoint_lookup)
    material_margins_ok = bool(
        material_ordering_ok
        and all(gap >= MATERIAL_ORDER_MIN_GAP_CP for gap in exported_material_gaps_cp.values())
    )
    return {
        "sanity_positions": sanity_positions,
        "material_ordering_ok": bool(material_ordering_ok),
        "material_margins_ok": material_margins_ok,
        "material_min_gap_cp": MATERIAL_ORDER_MIN_GAP_CP,
        "checkpoint_material_gaps_cp": checkpoint_material_gaps_cp,
        "exported_material_gaps_cp": exported_material_gaps_cp,
        "starting_position_near_zero": abs(sanity_export_lookup["starting_position"]) <= _cp_to_score(50.0),
    }


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
            combined = np.concatenate([white_acc, black_acc], axis=0)
        else:
            combined = np.concatenate([black_acc, white_acc], axis=0)
        fc0_out = combined @ fc0_weight + fc0_bias
        hidden = fc0_out[: exported.hidden_size]
        forward = fc0_out[exported.hidden_size : exported.fc0_output_size]
        hidden_crelu = np.clip(hidden, 0.0, 1.0)
        fc1_in = np.concatenate([hidden_crelu * hidden_crelu, hidden_crelu], axis=0)
        fc1_out = np.clip(fc1_in @ fc1_weight + fc1_bias, 0.0, 1.0)
        fc2_out = fc1_out @ fc2_weight + fc2_bias[0]
        output = (fc2_out + forward[0]) * exported.score_scale
        results.append(float(output))
    return results


def _score_to_cp(value: float) -> float:
    return float(value) * STOCKFISH_INTERNAL_TO_CP


def _cp_to_score(value: float) -> float:
    return float(value) / STOCKFISH_INTERNAL_TO_CP


def _material_gaps_cp(values: dict[str, float]) -> dict[str, float]:
    return {
        "pawn_over_equal": _score_to_cp(values["white_up_pawn"] - values["equal_material"]),
        "knight_over_pawn": _score_to_cp(values["white_up_knight"] - values["white_up_pawn"]),
        "rook_over_knight": _score_to_cp(values["white_up_rook"] - values["white_up_knight"]),
        "queen_over_rook": _score_to_cp(values["white_up_queen"] - values["white_up_rook"]),
    }


def _exported_network_from_model(model, config) -> ExportedNetwork:
    ft_weight = model.ft.weight.detach().cpu().numpy()
    ft_bias = model.ft_bias.detach().cpu().numpy()
    fc0_weight = model.fc0.weight.detach().cpu().numpy().T
    fc0_bias = model.fc0.bias.detach().cpu().numpy()
    fc1_weight = model.fc1.weight.detach().cpu().numpy().T
    fc1_bias = model.fc1.bias.detach().cpu().numpy()
    fc2_weight = model.fc2.weight.detach().cpu().numpy().reshape(-1)
    fc2_bias = model.fc2.bias.detach().cpu().numpy()
    score_scale = float(config.nnue2score)

    ft_scale = _fit_quantization_scale([ft_bias, ft_weight], config.export_ft_scale, np.int16)
    fc0_scale = _fit_quantization_scale([fc0_weight], config.export_dense_scale, np.int8)
    fc1_scale = _fit_quantization_scale([fc1_weight], config.export_dense_scale, np.int8)
    fc2_scale = _fit_quantization_scale([fc2_weight], config.export_dense_scale, np.int8)

    return ExportedNetwork(
        description=config.export_description,
        version=VERSION,
        num_features=config.num_features,
        ft_size=config.ft_size,
        hidden_size=config.hidden_size,
        forward_size=config.forward_size,
        fc1_output_size=config.fc1_output_size,
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


def _fit_quantization_scale(values: list[np.ndarray], requested_scale: float, dtype) -> float:
    info = np.iinfo(dtype)
    max_abs = max((float(np.max(np.abs(value))) for value in values if value.size > 0), default=0.0)
    if max_abs <= 0.0:
        return float(requested_scale)
    max_scale_without_limit_hits = max(float(info.max) - 0.5, 1.0) / max_abs
    headroom_scale = float(np.nextafter(max_scale_without_limit_hits, 0.0))
    return float(min(requested_scale, headroom_scale))


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


def _pass_fail(value: bool) -> str:
    return "pass" if value else "fail"


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
        exported.hidden_size,
        exported.forward_size,
        exported.fc0_output_size,
        exported.fc1_input_size,
        exported.fc1_output_size,
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
    handle.write(exported.fc0_bias.astype("<i4").tobytes())
    handle.write(exported.fc0_weight.astype(np.int8).tobytes())
    handle.write(exported.fc1_bias.astype("<i4").tobytes())
    handle.write(exported.fc1_weight.astype(np.int8).tobytes())
    handle.write(exported.fc2_bias.astype("<i4").tobytes())
    handle.write(exported.fc2_weight.astype(np.int8).tobytes())


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
