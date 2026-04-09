from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any

import numpy as np

from .board import BoardState
from .features import active_feature_indices, output_bucket_index


MAGIC = b"THNNUE\x00\x01"
VERSION = 4
FEATURE_SET_ID = "a768_dual_v1"
OUTPUT_PERSPECTIVE_STM = 1
HEADER_PREFIX_STRUCT = struct.Struct("<8sI")
LEGACY_HEADER_REST_STRUCT = struct.Struct("<16sIIIIfffI")
HEADER_REST_STRUCT = struct.Struct("<16sIIIIIfffI")
DEFAULT_VERIFICATION_FENS = [
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/2P5/PP3PPP/RNBQKBNR b KQkq - 0 3",
    "8/2k5/8/8/8/8/5K2/8 w - - 0 1",
]


@dataclass(slots=True)
class ExportedNetwork:
    description: str
    num_features: int
    ft_size: int
    hidden_size: int
    output_buckets: int
    ft_scale: float
    dense_scale: float
    wdl_scale: float
    ft_bias: np.ndarray
    ft_weight: np.ndarray
    l1_bias: np.ndarray
    l1_weight: np.ndarray
    out_bias: np.ndarray
    out_weight: np.ndarray
    version: int = 3
    head_type: str = "scalar"
    wdl_out_bias: np.ndarray | None = None
    wdl_out_weight: np.ndarray | None = None


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for export commands") from exc
    return torch


def export_checkpoint(checkpoint_path: str | Path, output_path: str | Path) -> Path:
    torch = _require_torch()
    from .checkpoint import load_checkpoint
    from .config import TrainConfig
    from .model import DualPerspectiveA768NNUE

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = TrainConfig.from_dict(dict(checkpoint["config"]))
    model = DualPerspectiveA768NNUE(
        num_features=config.num_features,
        ft_size=config.ft_size,
        hidden_size=config.hidden_size,
        output_buckets=config.output_buckets,
        head_type=config.head_type,
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
        raw_prefix = handle.read(HEADER_PREFIX_STRUCT.size)
        if len(raw_prefix) != HEADER_PREFIX_STRUCT.size:
            raise ValueError("File too small to contain a Thrawn NNUE header")
        magic, version = HEADER_PREFIX_STRUCT.unpack(raw_prefix)

        if magic != MAGIC:
            raise ValueError("Unexpected .nnue magic")
        if version not in {1, 2, 3, VERSION}:
            raise ValueError(f"Unsupported .nnue version: {version}")

        if version in {1, 2}:
            raw_rest = handle.read(LEGACY_HEADER_REST_STRUCT.size)
            if len(raw_rest) != LEGACY_HEADER_REST_STRUCT.size:
                raise ValueError("File too small to contain a legacy Thrawn NNUE header")
            (
                feature_set,
                num_features,
                ft_size,
                hidden_size,
                output_perspective,
                ft_scale,
                dense_scale,
                wdl_scale,
                description_length,
            ) = LEGACY_HEADER_REST_STRUCT.unpack(raw_rest)
            output_buckets = 1
            head_type = "scalar"
        else:
            raw_rest = handle.read(HEADER_REST_STRUCT.size)
            if len(raw_rest) != HEADER_REST_STRUCT.size:
                raise ValueError("File too small to contain a Thrawn NNUE header")
            (
                feature_set,
                num_features,
                ft_size,
                hidden_size,
                output_buckets,
                output_perspective,
                ft_scale,
                dense_scale,
                wdl_scale,
                description_length,
            ) = HEADER_REST_STRUCT.unpack(raw_rest)
            head_type = "scalar" if version == 3 else "dual_value_wdl"

        if feature_set.rstrip(b"\x00").decode("ascii") != FEATURE_SET_ID:
            raise ValueError("Unexpected feature-set identifier")
        if output_perspective != OUTPUT_PERSPECTIVE_STM:
            raise ValueError("Only side-to-move exports are supported")

        description = handle.read(description_length).decode("utf-8")
        ft_bias = np.frombuffer(handle.read(ft_size * 2), dtype="<i2").copy()
        ft_weight = np.frombuffer(handle.read(num_features * ft_size * 2), dtype="<i2").copy()
        ft_weight = ft_weight.reshape(num_features, ft_size)
        l1_bias = np.frombuffer(handle.read(hidden_size * 4), dtype="<i4").copy()
        l1_weight = np.frombuffer(handle.read(ft_size * 2 * hidden_size), dtype=np.int8).copy()
        l1_weight = l1_weight.reshape(ft_size * 2, hidden_size)
        out_bias = np.frombuffer(handle.read(output_buckets * 4), dtype="<i4").copy()
        if version == 1:
            out_weight = np.frombuffer(handle.read(hidden_size), dtype=np.int8).copy().reshape(hidden_size, 1)
        else:
            out_weight = np.frombuffer(
                handle.read(hidden_size * output_buckets * 2),
                dtype="<i2",
            ).copy().reshape(hidden_size, output_buckets)
        wdl_out_bias = None
        wdl_out_weight = None
        if version >= 4:
            wdl_out_bias = np.frombuffer(handle.read(output_buckets * 3 * 4), dtype="<i4").copy()
            wdl_out_weight = np.frombuffer(
                handle.read(hidden_size * output_buckets * 3 * 2),
                dtype="<i2",
            ).copy().reshape(hidden_size, output_buckets * 3)
        return ExportedNetwork(
            description=description,
            version=version,
            head_type=head_type,
            num_features=num_features,
            ft_size=ft_size,
            hidden_size=hidden_size,
            output_buckets=output_buckets,
            ft_scale=ft_scale,
            dense_scale=dense_scale,
            wdl_scale=wdl_scale,
            ft_bias=ft_bias,
            ft_weight=ft_weight,
            l1_bias=l1_bias,
            l1_weight=l1_weight,
            out_bias=out_bias,
            out_weight=out_weight,
            wdl_out_bias=wdl_out_bias,
            wdl_out_weight=wdl_out_weight,
        )


def verify_export(checkpoint_path: str | Path, nnue_path: str | Path, fens: list[str] | None = None) -> dict[str, Any]:
    torch = _require_torch()
    from .checkpoint import load_checkpoint
    from .config import TrainConfig
    from .model import DualPerspectiveA768NNUE

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = TrainConfig.from_dict(dict(checkpoint["config"]))
    model = DualPerspectiveA768NNUE(
        num_features=config.num_features,
        ft_size=config.ft_size,
        hidden_size=config.hidden_size,
        output_buckets=config.output_buckets,
        head_type=config.head_type,
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    fens = fens or DEFAULT_VERIFICATION_FENS
    with torch.no_grad():
        predictions = []
        wdl_predictions = []
        for fen in fens:
            white_indices, black_indices, stm, bucket_indices = _batch_arrays_from_fens(
                [fen],
                output_buckets=config.output_buckets,
            )
            pred = model(
                torch.from_numpy(white_indices).long(),
                torch.from_numpy(black_indices).long(),
                torch.from_numpy(stm).float().unsqueeze(1),
                torch.from_numpy(bucket_indices).long(),
                return_wdl=config.head_type == "dual_value_wdl",
            )
            if config.head_type == "dual_value_wdl":
                value_pred, wdl_logits = pred
                predictions.append(float(value_pred.squeeze().cpu().item()))
                expected = torch.softmax(wdl_logits, dim=1)[:, 2] + 0.5 * torch.softmax(wdl_logits, dim=1)[:, 1]
                wdl_predictions.append(float(torch.logit(expected.clamp(1e-6, 1.0 - 1e-6)).cpu().item()))
            else:
                predictions.append(float(pred.squeeze().cpu().item()))

    exported = load_export(nnue_path)
    exported_predictions = evaluate_export(exported, fens, head="value")

    abs_errors = [abs(a - b) for a, b in zip(predictions, exported_predictions)]
    result = {
        "positions": float(len(fens)),
        "max_abs_error": max(abs_errors) if abs_errors else 0.0,
        "mean_abs_error": (sum(abs_errors) / len(abs_errors)) if abs_errors else 0.0,
        "checkpoint_predictions": predictions,
        "exported_predictions": exported_predictions,
        "abs_errors": abs_errors,
        "export_ft_scale": float(exported.ft_scale),
        "export_dense_scale": float(exported.dense_scale),
        "quantization": _export_quantization_diagnostics(exported),
    }
    if exported.head_type == "dual_value_wdl":
        exported_wdl_predictions = evaluate_export(exported, fens, head="wdl")
        wdl_errors = [abs(a - b) for a, b in zip(wdl_predictions, exported_wdl_predictions)]
        result["checkpoint_wdl_predictions"] = wdl_predictions
        result["exported_wdl_predictions"] = exported_wdl_predictions
        result["wdl_max_abs_error"] = max(wdl_errors) if wdl_errors else 0.0
        result["wdl_mean_abs_error"] = (sum(wdl_errors) / len(wdl_errors)) if wdl_errors else 0.0
    return result


def evaluate_export(exported: ExportedNetwork, fens: list[str], *, head: str = "value") -> list[float]:
    ft_bias = exported.ft_bias.astype(np.float32) / exported.ft_scale
    ft_weight = exported.ft_weight.astype(np.float32) / exported.ft_scale
    l1_bias = exported.l1_bias.astype(np.float32) / exported.dense_scale
    l1_weight = exported.l1_weight.astype(np.float32) / exported.dense_scale
    out_bias = exported.out_bias.astype(np.float32) / exported.dense_scale
    out_weight = exported.out_weight.astype(np.float32) / exported.dense_scale
    wdl_out_bias = None if exported.wdl_out_bias is None else exported.wdl_out_bias.astype(np.float32) / exported.dense_scale
    wdl_out_weight = None if exported.wdl_out_weight is None else exported.wdl_out_weight.astype(np.float32) / exported.dense_scale

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
        hidden = np.square(np.clip(combined, 0.0, 1.0))
        hidden = np.clip(hidden @ l1_weight + l1_bias, 0.0, 1.0)
        output = hidden @ out_weight + out_bias
        bucket = output_bucket_index(len(board.board), exported.output_buckets)
        if head == "value" or exported.head_type == "scalar":
            results.append(float(output[bucket]))
            continue
        if head != "wdl":
            raise ValueError(f"Unknown export head: {head}")
        if wdl_out_bias is None or wdl_out_weight is None:
            raise ValueError("Requested WDL head from scalar export")
        wdl_output = hidden @ wdl_out_weight + wdl_out_bias
        logits = wdl_output.reshape(exported.output_buckets, 3)[bucket]
        probabilities = _softmax_numpy(logits)
        expected = probabilities[2] + 0.5 * probabilities[1]
        results.append(float(_logit_numpy(expected)))
    return results


def _exported_network_from_model(model, config) -> ExportedNetwork:
    head_type = getattr(config, "head_type", "scalar")
    ft_weight = model.ft.weight.detach().cpu().numpy()
    ft_bias = model.ft_bias.detach().cpu().numpy()
    l1_weight = model.l1.weight.detach().cpu().numpy().T
    l1_bias = model.l1.bias.detach().cpu().numpy()
    out_weight = model.output.weight.detach().cpu().numpy().T
    out_bias = model.output.bias.detach().cpu().numpy()
    ft_scale = _fit_quantization_scale([ft_bias, ft_weight], config.export_ft_scale, np.int16)
    dense_values = [l1_weight, out_weight]
    wdl_out_weight = None
    wdl_out_bias = None
    if getattr(model, "wdl_output", None) is not None:
        wdl_out_weight = model.wdl_output.weight.detach().cpu().numpy().T
        wdl_out_bias = model.wdl_output.bias.detach().cpu().numpy()
        dense_values.append(wdl_out_weight)
    dense_scale = _fit_quantization_scale([l1_weight], config.export_dense_scale, np.int8)

    return ExportedNetwork(
        description=config.export_description,
        version=VERSION if head_type == "dual_value_wdl" else 3,
        head_type=head_type,
        num_features=config.num_features,
        ft_size=config.ft_size,
        hidden_size=config.hidden_size,
        output_buckets=config.output_buckets,
        ft_scale=ft_scale,
        dense_scale=dense_scale,
        wdl_scale=config.wdl_scale,
        ft_bias=_quantize(ft_bias, ft_scale, np.int16),
        ft_weight=_quantize(ft_weight, ft_scale, np.int16),
        l1_bias=_quantize(l1_bias, dense_scale, np.int32),
        l1_weight=_quantize(l1_weight, dense_scale, np.int8),
        out_bias=_quantize(out_bias, dense_scale, np.int32),
        out_weight=_quantize(out_weight, dense_scale, np.int16),
        wdl_out_bias=None if wdl_out_bias is None else _quantize(wdl_out_bias, dense_scale, np.int32),
        wdl_out_weight=None if wdl_out_weight is None else _quantize(wdl_out_weight, dense_scale, np.int16),
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
    diagnostics = {
        "ft_bias": _quantized_tensor_stats(exported.ft_bias),
        "ft_weight": _quantized_tensor_stats(exported.ft_weight),
        "l1_bias": _quantized_tensor_stats(exported.l1_bias),
        "l1_weight": _quantized_tensor_stats(exported.l1_weight),
        "out_bias": _quantized_tensor_stats(exported.out_bias),
        "out_weight": _quantized_tensor_stats(exported.out_weight),
    }
    if exported.wdl_out_bias is not None and exported.wdl_out_weight is not None:
        diagnostics["wdl_out_bias"] = _quantized_tensor_stats(exported.wdl_out_bias)
        diagnostics["wdl_out_weight"] = _quantized_tensor_stats(exported.wdl_out_weight)
    return diagnostics


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
        exported.output_buckets,
        OUTPUT_PERSPECTIVE_STM,
        float(exported.ft_scale),
        float(exported.dense_scale),
        float(exported.wdl_scale),
        len(description_bytes),
    )
    handle.write(header)
    handle.write(description_bytes)
    handle.write(exported.ft_bias.astype("<i2").tobytes())
    handle.write(exported.ft_weight.astype("<i2").tobytes())
    handle.write(exported.l1_bias.astype("<i4").tobytes())
    handle.write(exported.l1_weight.astype(np.int8).tobytes())
    handle.write(exported.out_bias.astype("<i4").tobytes())
    handle.write(exported.out_weight.astype("<i2").tobytes())
    if exported.version >= 4 and exported.wdl_out_bias is not None and exported.wdl_out_weight is not None:
        handle.write(exported.wdl_out_bias.astype("<i4").tobytes())
        handle.write(exported.wdl_out_weight.astype("<i2").tobytes())


def _softmax_numpy(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exps = np.exp(shifted)
    return exps / np.sum(exps)


def _logit_numpy(value: float) -> float:
    epsilon = np.finfo(np.float32).eps
    clamped = float(np.clip(value, epsilon, 1.0 - epsilon))
    return float(np.log(clamped / (1.0 - clamped)))


def _batch_arrays_from_fens(
    fens: list[str],
    *,
    output_buckets: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    white_indices = []
    black_indices = []
    stm = []
    bucket_indices = []
    for fen in fens:
        board = BoardState.from_fen(fen)
        white = active_feature_indices(board, "white")
        black = active_feature_indices(board, "black")
        white_indices.append(white + [-1] * (32 - len(white)))
        black_indices.append(black + [-1] * (32 - len(black)))
        stm.append(1.0 if board.side_to_move == "w" else 0.0)
        bucket_indices.append(output_bucket_index(len(board.board), output_buckets))
    return (
        np.asarray(white_indices, dtype=np.int32),
        np.asarray(black_indices, dtype=np.int32),
        np.asarray(stm, dtype=np.float32),
        np.asarray(bucket_indices, dtype=np.int64),
    )
