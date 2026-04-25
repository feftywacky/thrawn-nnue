from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any

import numpy as np

from .board import BoardState
from .features import active_feature_indices


MAGIC = b"THNNUE\x00\x01"
VERSION = 6
FEATURE_SET_ID = "halfkp_v1"
OUTPUT_PERSPECTIVE_STM = 1
EXPECTED_NUM_FEATURES = 40960
MAX_DESCRIPTION_BYTES = 1_000_000
HEADER_PREFIX_STRUCT = struct.Struct("<8sI")
HEADER_REST_STRUCT = struct.Struct("<16sIIIIIffffI")
DEFAULT_VERIFICATION_FENS = [
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b - - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/2P5/PP3PPP/RNBQKBNR b - - 0 3",
    "8/2k5/8/8/8/8/5K2/8 w - - 0 1",
]
MATERIAL_SANITY_POSITIONS = [
    ("starting_position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
    ("white_up_pawn", "rnbqkbnr/1ppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
    ("white_up_knight", "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
    ("white_up_rook", "1nbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
    ("white_up_queen", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
]


@dataclass(slots=True)
class ExportedNetwork:
    description: str
    num_features: int
    ft_size: int
    l1_size: int
    l2_size: int
    ft_scale: float
    l1_scale: float
    l2_scale: float
    out_scale: float
    ft_bias: np.ndarray
    ft_weight: np.ndarray
    l1_bias: np.ndarray
    l1_weight: np.ndarray
    l2_bias: np.ndarray
    l2_weight: np.ndarray
    out_bias: np.ndarray
    out_weight: np.ndarray
    version: int = VERSION


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for export commands") from exc
    return torch


def export_checkpoint(checkpoint_path: str | Path, output_path: str | Path) -> Path:
    from .checkpoint import load_checkpoint
    from .config import TrainConfig
    from .model import HalfKPNNUE

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = TrainConfig.from_dict(dict(checkpoint["config"]))
    model = HalfKPNNUE(
        num_features=config.num_features,
        num_factor_features=config.num_factor_features,
        ft_size=config.ft_size,
        l1_size=config.l1_size,
        l2_size=config.l2_size,
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
            l1_size,
            l2_size,
            output_perspective,
            ft_scale,
            l1_scale,
            l2_scale,
            out_scale,
            description_length,
        ) = HEADER_REST_STRUCT.unpack(raw_rest)

        if feature_set.rstrip(b"\x00").decode("ascii") != FEATURE_SET_ID:
            raise ValueError("Unexpected feature-set identifier")
        _validate_export_header(
            num_features=num_features,
            ft_size=ft_size,
            l1_size=l1_size,
            l2_size=l2_size,
            output_perspective=output_perspective,
            ft_scale=ft_scale,
            l1_scale=l1_scale,
            l2_scale=l2_scale,
            out_scale=out_scale,
            description_length=description_length,
        )

        description = _read_exact(handle, description_length, "description").decode("utf-8")
        ft_bias = np.frombuffer(_read_exact(handle, ft_size * 2, "ft_bias"), dtype="<i2").copy()
        ft_weight = np.frombuffer(
            _read_exact(handle, num_features * ft_size * 2, "ft_weight"),
            dtype="<i2",
        ).copy()
        ft_weight = ft_weight.reshape(num_features, ft_size)
        l1_bias = np.frombuffer(_read_exact(handle, l1_size * 4, "l1_bias"), dtype="<i4").copy()
        l1_weight = np.frombuffer(
            _read_exact(handle, ft_size * 2 * l1_size, "l1_weight"),
            dtype=np.int8,
        ).copy().reshape(ft_size * 2, l1_size)
        l2_bias = np.frombuffer(_read_exact(handle, l2_size * 4, "l2_bias"), dtype="<i4").copy()
        l2_weight = np.frombuffer(
            _read_exact(handle, l1_size * l2_size, "l2_weight"),
            dtype=np.int8,
        ).copy().reshape(l1_size, l2_size)
        out_bias = np.frombuffer(_read_exact(handle, 4, "out_bias"), dtype="<i4").copy()
        out_weight = np.frombuffer(_read_exact(handle, l2_size, "out_weight"), dtype=np.int8).copy()
        if handle.read(1):
            raise ValueError("Unexpected trailing data in .nnue export")
        return ExportedNetwork(
            description=description,
            version=version,
            num_features=num_features,
            ft_size=ft_size,
            l1_size=l1_size,
            l2_size=l2_size,
            ft_scale=ft_scale,
            l1_scale=l1_scale,
            l2_scale=l2_scale,
            out_scale=out_scale,
            ft_bias=ft_bias,
            ft_weight=ft_weight,
            l1_bias=l1_bias,
            l1_weight=l1_weight,
            l2_bias=l2_bias,
            l2_weight=l2_weight,
            out_bias=out_bias,
            out_weight=out_weight,
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
    l1_size: int,
    l2_size: int,
    output_perspective: int,
    ft_scale: float,
    l1_scale: float,
    l2_scale: float,
    out_scale: float,
    description_length: int,
) -> None:
    if num_features != EXPECTED_NUM_FEATURES:
        raise ValueError(f"Unexpected num_features: {num_features}")
    for name, size in (("ft_size", ft_size), ("l1_size", l1_size), ("l2_size", l2_size)):
        if size <= 0:
            raise ValueError(f"{name} must be positive")
    if output_perspective != OUTPUT_PERSPECTIVE_STM:
        raise ValueError("Only side-to-move exports are supported")
    for name, scale in (
        ("ft_scale", ft_scale),
        ("l1_scale", l1_scale),
        ("l2_scale", l2_scale),
        ("out_scale", out_scale),
    ):
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if description_length > MAX_DESCRIPTION_BYTES:
        raise ValueError(f"description_length is too large: {description_length}")


def verify_export(checkpoint_path: str | Path, nnue_path: str | Path, fens: list[str] | None = None) -> dict[str, Any]:
    torch = _require_torch()
    from .checkpoint import load_checkpoint
    from .config import TrainConfig
    from .model import HalfKPNNUE

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = TrainConfig.from_dict(dict(checkpoint["config"]))
    model = HalfKPNNUE(
        num_features=config.num_features,
        num_factor_features=config.num_factor_features,
        ft_size=config.ft_size,
        l1_size=config.l1_size,
        l2_size=config.l2_size,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    fens = fens or DEFAULT_VERIFICATION_FENS
    with torch.no_grad():
        white_indices, black_indices, stm = _batch_arrays_from_fens(fens)
        predictions = model(
            torch.from_numpy(white_indices).long(),
            torch.from_numpy(black_indices).long(),
            torch.from_numpy(stm).float().unsqueeze(1),
        )
        predictions = [float(value) for value in predictions.reshape(-1).cpu().tolist()]

    exported = load_export(nnue_path)
    if (
        exported.num_features != config.num_features
        or exported.ft_size != config.ft_size
        or exported.l1_size != config.l1_size
        or exported.l2_size != config.l2_size
    ):
        raise ValueError("Export dimensions do not match checkpoint config")
    exported_predictions = evaluate_export(exported, fens)
    abs_errors = [abs(a - b) for a, b in zip(predictions, exported_predictions, strict=True)]

    sanity_fens = [fen for _, fen in MATERIAL_SANITY_POSITIONS]
    with torch.no_grad():
        white_indices, black_indices, stm = _batch_arrays_from_fens(sanity_fens)
        sanity_checkpoint = model(
            torch.from_numpy(white_indices).long(),
            torch.from_numpy(black_indices).long(),
            torch.from_numpy(stm).float().unsqueeze(1),
        )
        sanity_checkpoint_values = [float(value) for value in sanity_checkpoint.reshape(-1).cpu().tolist()]
    sanity_exported_values = evaluate_export(exported, sanity_fens)
    sanity_positions = [
        {
            "name": name,
            "fen": fen,
            "checkpoint_cp": checkpoint_cp,
            "exported_cp": exported_cp,
            "abs_error": abs(checkpoint_cp - exported_cp),
        }
        for (name, fen), checkpoint_cp, exported_cp in zip(
            MATERIAL_SANITY_POSITIONS,
            sanity_checkpoint_values,
            sanity_exported_values,
            strict=True,
        )
    ]
    sanity_export_lookup = {item["name"]: float(item["exported_cp"]) for item in sanity_positions}
    material_ordering_ok = (
        sanity_export_lookup["starting_position"]
        < sanity_export_lookup["white_up_pawn"]
        < sanity_export_lookup["white_up_knight"]
        < sanity_export_lookup["white_up_rook"]
        < sanity_export_lookup["white_up_queen"]
    )

    return {
        "positions": float(len(fens)),
        "max_abs_error": max(abs_errors) if abs_errors else 0.0,
        "mean_abs_error": (sum(abs_errors) / len(abs_errors)) if abs_errors else 0.0,
        "checkpoint_predictions": predictions,
        "exported_predictions": exported_predictions,
        "abs_errors": abs_errors,
        "export_ft_scale": float(exported.ft_scale),
        "export_l1_scale": float(exported.l1_scale),
        "export_l2_scale": float(exported.l2_scale),
        "export_out_scale": float(exported.out_scale),
        "quantization": _export_quantization_diagnostics(exported),
        "sanity_positions": sanity_positions,
        "material_ordering_ok": bool(material_ordering_ok),
        "starting_position_near_zero": abs(sanity_export_lookup["starting_position"]) <= 50.0,
    }


def evaluate_export(exported: ExportedNetwork, fens: list[str]) -> list[float]:
    ft_bias = exported.ft_bias.astype(np.float32) / exported.ft_scale
    ft_weight = exported.ft_weight.astype(np.float32) / exported.ft_scale
    l1_bias = exported.l1_bias.astype(np.float32) / exported.l1_scale
    l1_weight = exported.l1_weight.astype(np.float32) / exported.l1_scale
    l2_bias = exported.l2_bias.astype(np.float32) / exported.l2_scale
    l2_weight = exported.l2_weight.astype(np.float32) / exported.l2_scale
    out_bias = exported.out_bias.astype(np.float32) / exported.out_scale
    out_weight = exported.out_weight.astype(np.float32) / exported.out_scale

    results: list[float] = []
    for fen in fens:
        board = BoardState.from_fen(fen)
        white_features = active_feature_indices(board, "white")
        black_features = active_feature_indices(board, "black")
        white_acc = ft_bias + ft_weight[white_features].sum(axis=0)
        black_acc = ft_bias + ft_weight[black_features].sum(axis=0)
        if board.side_to_move == "w":
            combined = np.concatenate([white_acc, black_acc], axis=0)
        else:
            combined = np.concatenate([black_acc, white_acc], axis=0)
        hidden1 = np.clip(combined, 0.0, 1.0)
        hidden2 = np.clip(hidden1 @ l1_weight + l1_bias, 0.0, 1.0)
        hidden3 = np.clip(hidden2 @ l2_weight + l2_bias, 0.0, 1.0)
        output = hidden3 @ out_weight + out_bias[0]
        results.append(float(output))
    return results


def _coalesced_ft_weights_from_model(model) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(model, "coalesced_feature_transform"):
        ft_weight, ft_bias = model.coalesced_feature_transform()
        return (
            ft_weight.detach().cpu().numpy(),
            ft_bias.detach().cpu().numpy(),
        )

    ft_weight = model.ft.weight.detach().cpu().numpy()
    ft_bias = model.ft_bias.detach().cpu().numpy()
    if hasattr(model, "ft_factor"):
        factor_weight = model.ft_factor.weight.detach().cpu().numpy()
        repeats = ft_weight.shape[0] // factor_weight.shape[0]
        ft_weight = ft_weight + np.tile(factor_weight, (repeats, 1))
    return ft_weight, ft_bias


def _exported_network_from_model(model, config) -> ExportedNetwork:
    ft_weight, ft_bias = _coalesced_ft_weights_from_model(model)
    l1_weight = model.l1.weight.detach().cpu().numpy().T
    l1_bias = model.l1.bias.detach().cpu().numpy()
    l2_weight = model.l2.weight.detach().cpu().numpy().T
    l2_bias = model.l2.bias.detach().cpu().numpy()
    out_weight = model.output.weight.detach().cpu().numpy().reshape(-1)
    out_bias = model.output.bias.detach().cpu().numpy()

    ft_scale = _fit_quantization_scale([ft_bias, ft_weight], config.export_ft_scale, np.int16)
    l1_scale = _fit_quantization_scale([l1_weight], config.export_dense_scale, np.int8)
    l2_scale = _fit_quantization_scale([l2_weight], config.export_dense_scale, np.int8)
    out_scale = _fit_quantization_scale([out_weight], config.export_dense_scale, np.int8)

    return ExportedNetwork(
        description=config.export_description,
        version=VERSION,
        num_features=config.num_features,
        ft_size=config.ft_size,
        l1_size=config.l1_size,
        l2_size=config.l2_size,
        ft_scale=ft_scale,
        l1_scale=l1_scale,
        l2_scale=l2_scale,
        out_scale=out_scale,
        ft_bias=_quantize(ft_bias, ft_scale, np.int16),
        ft_weight=_quantize(ft_weight, ft_scale, np.int16),
        l1_bias=_quantize(l1_bias, l1_scale, np.int32),
        l1_weight=_quantize(l1_weight, l1_scale, np.int8),
        l2_bias=_quantize(l2_bias, l2_scale, np.int32),
        l2_weight=_quantize(l2_weight, l2_scale, np.int8),
        out_bias=_quantize(out_bias, out_scale, np.int32),
        out_weight=_quantize(out_weight, out_scale, np.int8),
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
        "l1_bias": _quantized_tensor_stats(exported.l1_bias),
        "l1_weight": _quantized_tensor_stats(exported.l1_weight),
        "l2_bias": _quantized_tensor_stats(exported.l2_bias),
        "l2_weight": _quantized_tensor_stats(exported.l2_weight),
        "out_bias": _quantized_tensor_stats(exported.out_bias),
        "out_weight": _quantized_tensor_stats(exported.out_weight),
    }


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
        exported.l1_size,
        exported.l2_size,
        OUTPUT_PERSPECTIVE_STM,
        float(exported.ft_scale),
        float(exported.l1_scale),
        float(exported.l2_scale),
        float(exported.out_scale),
        len(description_bytes),
    )
    handle.write(header)
    handle.write(description_bytes)
    handle.write(exported.ft_bias.astype("<i2").tobytes())
    handle.write(exported.ft_weight.astype("<i2").tobytes())
    handle.write(exported.l1_bias.astype("<i4").tobytes())
    handle.write(exported.l1_weight.astype(np.int8).tobytes())
    handle.write(exported.l2_bias.astype("<i4").tobytes())
    handle.write(exported.l2_weight.astype(np.int8).tobytes())
    handle.write(exported.out_bias.astype("<i4").tobytes())
    handle.write(exported.out_weight.astype(np.int8).tobytes())


def _batch_arrays_from_fens(
    fens: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    white_indices = []
    black_indices = []
    stm = []
    for fen in fens:
        board = BoardState.from_fen(fen)
        white = active_feature_indices(board, "white")
        black = active_feature_indices(board, "black")
        white_indices.append(white + [-1] * (30 - len(white)))
        black_indices.append(black + [-1] * (30 - len(black)))
        stm.append(1.0 if board.side_to_move == "w" else 0.0)
    return (
        np.asarray(white_indices, dtype=np.int32),
        np.asarray(black_indices, dtype=np.int32),
        np.asarray(stm, dtype=np.float32),
    )
