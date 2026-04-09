from __future__ import annotations

from pathlib import Path

import numpy as np

from .export import load_export
from .native import BinpackStream, discover_binpack_files


HARD_CODED_POSITIONS: list[tuple[str, str]] = [
    ("starting_position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("white_up_pawn", "rnbqkbnr/ppppppp1/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("white_up_knight", "r1bqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
]


class _ExportEvaluator:
    def __init__(self, nnue_path: str | Path) -> None:
        exported = load_export(nnue_path)
        self._exported = exported
        self.ft_bias = exported.ft_bias.astype(np.float32) / exported.ft_scale
        self.ft_weight = exported.ft_weight.astype(np.float32) / exported.ft_scale
        self.l1_bias = exported.l1_bias.astype(np.float32) / exported.dense_scale
        self.l1_weight = exported.l1_weight.astype(np.float32) / exported.dense_scale
        self.out_bias = exported.out_bias.astype(np.float32) / exported.dense_scale
        self.out_weight = exported.out_weight.astype(np.float32) / exported.dense_scale

    def eval_batch(self, batch) -> np.ndarray:
        white_indices = batch.white_indices
        black_indices = batch.black_indices
        white_mask = white_indices >= 0
        black_mask = black_indices >= 0

        white_indices = np.clip(white_indices, 0, None)
        black_indices = np.clip(black_indices, 0, None)

        white_acc = self.ft_bias[None, :] + (
            self.ft_weight[white_indices] * white_mask[..., None]
        ).sum(axis=1)
        black_acc = self.ft_bias[None, :] + (
            self.ft_weight[black_indices] * black_mask[..., None]
        ).sum(axis=1)

        stm = batch.stm >= 0.5
        combined_when_white = np.concatenate([white_acc, black_acc], axis=1)
        combined_when_black = np.concatenate([black_acc, white_acc], axis=1)
        combined = np.where(stm[:, None], combined_when_white, combined_when_black)

        hidden = np.square(np.clip(combined, 0.0, 1.0))
        hidden = np.clip(hidden @ self.l1_weight + self.l1_bias, 0.0, 1.0)
        outputs = hidden @ self.out_weight + self.out_bias

        if self._exported.output_buckets <= 1:
            return outputs[:, 0].astype(np.float64)

        bucket_indices = _output_bucket_indices(batch.white_counts, self._exported.output_buckets)
        return outputs[np.arange(outputs.shape[0]), bucket_indices].astype(np.float64)

    def eval_fens(self, fens: list[str]) -> list[float]:
        from .export import evaluate_export

        return evaluate_export(self._exported, fens)


def calibrate_scale(
    nnue_path: str | Path,
    validation_path: str | Path,
    *,
    max_positions: int = 300_000,
    batch_size: int = 1024,
    threads: int = 4,
    fit_window_cp: float = 600.0,
    min_fit_positions: int = 1000,
) -> dict[str, object]:
    if max_positions <= 0:
        raise ValueError("max_positions must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if threads <= 0:
        raise ValueError("threads must be positive")
    if fit_window_cp <= 0.0:
        raise ValueError("fit_window_cp must be positive")
    if min_fit_positions <= 0:
        raise ValueError("min_fit_positions must be positive")

    evaluator = _ExportEvaluator(nnue_path)
    dataset_paths = discover_binpack_files(validation_path)

    raw_parts: list[np.ndarray] = []
    teacher_cp_parts: list[np.ndarray] = []
    positions_used = 0

    with BinpackStream(dataset_paths, num_threads=threads, cyclic=False) as stream:
        while positions_used < max_positions:
            requested = min(batch_size, max_positions - positions_used)
            batch = stream.next_batch(requested)
            if batch is None:
                break
            raw_parts.append(evaluator.eval_batch(batch))
            score_cp = batch.score_cp.astype(np.float64)
            teacher_cp_parts.append(score_cp)
            positions_used += int(batch.score_cp.shape[0])

    if not raw_parts:
        raise ValueError("No positions were read from validation data")

    raw = np.concatenate(raw_parts, axis=0)
    teacher_cp = np.concatenate(teacher_cp_parts, axis=0)

    global_cp_per_raw = _fit_scale_through_origin(raw, teacher_cp)
    global_fit_metrics = _fit_metrics(raw, teacher_cp, global_cp_per_raw)

    quiet_mask = np.abs(teacher_cp) <= float(fit_window_cp)
    within_window = int(np.count_nonzero(quiet_mask))
    if within_window < min_fit_positions:
        raise ValueError(
            "Not enough positions in quiet fit window: "
            f"required at least {min_fit_positions}, got {within_window} "
            f"for |teacher_cp| <= {fit_window_cp}"
        )

    quiet_raw = raw[quiet_mask]
    quiet_cp = teacher_cp[quiet_mask]
    cp_per_raw = _fit_scale_through_origin(quiet_raw, quiet_cp)
    if abs(cp_per_raw) <= 1e-12:
        raise ValueError("Cannot fit scale: fitted cp_per_raw is approximately zero")
    fit_metrics = _fit_metrics(quiet_raw, quiet_cp, cp_per_raw)

    hardcoded_fens = [fen for _, fen in HARD_CODED_POSITIONS]
    hardcoded_black_fens = [_flip_side_to_move(fen) for fen in hardcoded_fens]
    hardcoded_raw = evaluator.eval_fens(hardcoded_fens)
    hardcoded_black_raw = evaluator.eval_fens(hardcoded_black_fens)
    hardcoded_positions = [
        {
            "name": name,
            "fen": fen,
            "raw": float(raw_value),
            "scaled_cp": float(raw_value * cp_per_raw),
        }
        for (name, fen), raw_value in zip(HARD_CODED_POSITIONS, hardcoded_raw, strict=True)
    ]
    symmetry_checks = [
        {
            "name": name,
            "fen_white_stm": fen_white,
            "fen_black_stm": fen_black,
            "white_raw": float(raw_white),
            "black_raw": float(raw_black),
            "white_scaled_cp": float(raw_white * cp_per_raw),
            "black_scaled_cp": float(raw_black * cp_per_raw),
            "sum_scaled_cp": float((raw_white + raw_black) * cp_per_raw),
        }
        for (name, fen_white), fen_black, raw_white, raw_black in zip(
            HARD_CODED_POSITIONS,
            hardcoded_black_fens,
            hardcoded_raw,
            hardcoded_black_raw,
            strict=True,
        )
    ]
    starting_position_cp = float(hardcoded_positions[0]["scaled_cp"])
    sanity_flags: list[str] = []
    if abs(starting_position_cp) > 150.0:
        sanity_flags.append("starting_position_scaled_cp_magnitude_gt_150")
    if float(fit_metrics["corr"]) < 0.6:
        sanity_flags.append("quiet_fit_correlation_lt_0.6")
    for check in symmetry_checks:
        if check["name"] == "starting_position":
            continue
        if not (check["white_scaled_cp"] > 0.0 and check["black_scaled_cp"] < 0.0):
            sanity_flags.append(f"symmetry_sign_mismatch_{check['name']}")

    normalization_constant = float(100.0 / cp_per_raw)

    return {
        "positions_used": int(positions_used),
        "teacher_perspective": "stm",
        "fit_scope": "quiet_range",
        "fit_filter": {
            "total_sampled": int(raw.shape[0]),
            "within_window": within_window,
            "used_for_fit": int(quiet_raw.shape[0]),
            "window_cp": float(fit_window_cp),
        },
        "cp_per_raw": float(cp_per_raw),
        "raw_per_cp": float(1.0 / cp_per_raw),
        "fit_metrics": fit_metrics,
        "global_cp_per_raw": float(global_cp_per_raw),
        "global_fit_metrics": global_fit_metrics,
        "normalization_constant": normalization_constant,
        "normalization_constant_rounded": int(round(normalization_constant)),
        "sanity_flags": sanity_flags,
        "hardcoded_positions": hardcoded_positions,
        "symmetry_checks": symmetry_checks,
    }


def _fit_scale_through_origin(raw: np.ndarray, cp: np.ndarray) -> float:
    denominator = float(np.dot(raw, raw))
    if denominator <= 1e-12:
        raise ValueError("Cannot fit scale: NNUE raw outputs are all zero on sampled positions")
    return float(np.dot(raw, cp) / denominator)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(value):
        return 0.0
    return value


def _fit_metrics(raw: np.ndarray, cp: np.ndarray, cp_per_raw: float) -> dict[str, float]:
    predicted_cp = cp_per_raw * raw
    return {
        "mae_cp": float(np.mean(np.abs(predicted_cp - cp))),
        "rmse_cp": float(np.sqrt(np.mean((predicted_cp - cp) ** 2))),
        "corr": _safe_corr(raw, cp),
    }


def _output_bucket_indices(piece_counts: np.ndarray, output_buckets: int) -> np.ndarray:
    clamped = np.clip(piece_counts.astype(np.int64), 2, 32)
    phase_progress = 32 - clamped
    return np.minimum(output_buckets - 1, (phase_progress * output_buckets) // 31)


def _flip_side_to_move(fen: str) -> str:
    parts = fen.split()
    if len(parts) != 6:
        raise ValueError(f"Invalid FEN for side-to-move flip: {fen}")
    if parts[1] == "w":
        parts[1] = "b"
    elif parts[1] == "b":
        parts[1] = "w"
    else:
        raise ValueError(f"Invalid side-to-move token in FEN: {fen}")
    return " ".join(parts)
