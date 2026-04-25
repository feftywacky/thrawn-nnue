from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import ctypes
import math
import os
import subprocess
import sys

import numpy as np


class NativeError(RuntimeError):
    pass


class _BatchView(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int32),
        ("max_active_features", ctypes.c_int32),
        ("white_indices", ctypes.POINTER(ctypes.c_int32)),
        ("black_indices", ctypes.POINTER(ctypes.c_int32)),
        ("stm", ctypes.POINTER(ctypes.c_float)),
        ("score_cp", ctypes.POINTER(ctypes.c_float)),
        ("result_wdl", ctypes.POINTER(ctypes.c_float)),
    ]


class _InspectStats(ctypes.Structure):
    _fields_ = [
        ("entries_read", ctypes.c_uint64),
        ("white_to_move", ctypes.c_uint64),
        ("black_to_move", ctypes.c_uint64),
        ("wins", ctypes.c_uint64),
        ("draws", ctypes.c_uint64),
        ("losses", ctypes.c_uint64),
        ("min_score", ctypes.c_double),
        ("max_score", ctypes.c_double),
        ("min_ply", ctypes.c_uint16),
        ("max_ply", ctypes.c_uint16),
        ("mean_score", ctypes.c_double),
        ("score_std", ctypes.c_double),
        ("mean_abs_score", ctypes.c_double),
        ("abs_score_std", ctypes.c_double),
        ("mean_ply", ctypes.c_double),
        ("ply_std", ctypes.c_double),
        ("mean_piece_count", ctypes.c_double),
        ("piece_count_std", ctypes.c_double),
        ("mean_non_king_piece_count", ctypes.c_double),
        ("result_mean", ctypes.c_double),
        ("score_result_correlation", ctypes.c_double),
        ("score_p01", ctypes.c_double),
        ("score_p05", ctypes.c_double),
        ("score_p10", ctypes.c_double),
        ("score_p25", ctypes.c_double),
        ("score_p50", ctypes.c_double),
        ("score_p75", ctypes.c_double),
        ("score_p90", ctypes.c_double),
        ("score_p95", ctypes.c_double),
        ("score_p99", ctypes.c_double),
        ("score_p999", ctypes.c_double),
        ("abs_score_p50", ctypes.c_double),
        ("abs_score_p75", ctypes.c_double),
        ("abs_score_p90", ctypes.c_double),
        ("abs_score_p95", ctypes.c_double),
        ("abs_score_p99", ctypes.c_double),
        ("abs_score_p999", ctypes.c_double),
        ("ply_p05", ctypes.c_double),
        ("ply_p25", ctypes.c_double),
        ("ply_p50", ctypes.c_double),
        ("ply_p75", ctypes.c_double),
        ("ply_p95", ctypes.c_double),
        ("ply_p99", ctypes.c_double),
        ("piece_count_p05", ctypes.c_double),
        ("piece_count_p25", ctypes.c_double),
        ("piece_count_p50", ctypes.c_double),
        ("piece_count_p75", ctypes.c_double),
        ("piece_count_p95", ctypes.c_double),
        ("abs_score_ge_1000", ctypes.c_uint64),
        ("abs_score_ge_2000", ctypes.c_uint64),
        ("abs_score_ge_4000", ctypes.c_uint64),
        ("abs_score_ge_8000", ctypes.c_uint64),
        ("abs_score_ge_16000", ctypes.c_uint64),
        ("score_bucket_counts", ctypes.c_uint64 * 21),
        ("abs_score_bucket_counts", ctypes.c_uint64 * 11),
        ("ply_bucket_counts", ctypes.c_uint64 * 8),
        ("piece_count_bucket_counts", ctypes.c_uint64 * 4),
        ("phase_counts", ctypes.c_uint64 * 4),
        ("phase_mean_score", ctypes.c_double * 4),
        ("phase_mean_abs_score", ctypes.c_double * 4),
        ("phase_result_mean", ctypes.c_double * 4),
        ("result_score_agree", ctypes.c_uint64),
        ("result_score_disagree", ctypes.c_uint64),
        ("decisive_result_near_zero_score", ctypes.c_uint64),
        ("draw_high_abs_score", ctypes.c_uint64),
        ("mean_score_win", ctypes.c_double),
        ("mean_score_draw", ctypes.c_double),
        ("mean_score_loss", ctypes.c_double),
        ("mean_abs_score_win", ctypes.c_double),
        ("mean_abs_score_draw", ctypes.c_double),
        ("mean_abs_score_loss", ctypes.c_double),
        ("piece_type_counts", ctypes.c_uint64 * 6),
        ("white_piece_counts", ctypes.c_uint64 * 6),
        ("black_piece_counts", ctypes.c_uint64 * 6),
        ("wdl_scale_signed_target_mean", ctypes.c_double * 7),
        ("wdl_scale_signed_target_std", ctypes.c_double * 7),
        ("wdl_scale_abs_target_mean", ctypes.c_double * 7),
        ("wdl_scale_saturated_99", ctypes.c_uint64 * 7),
        ("wdl_scale_saturated_999", ctypes.c_uint64 * 7),
    ]


_SCORE_BUCKET_LABELS = [
    "lt_-8000",
    "-8000_-4000",
    "-4000_-2000",
    "-2000_-1000",
    "-1000_-600",
    "-600_-400",
    "-400_-200",
    "-200_-100",
    "-100_-50",
    "-50_0",
    "eq_0",
    "0_50",
    "50_100",
    "100_200",
    "200_400",
    "400_600",
    "600_1000",
    "1000_2000",
    "2000_4000",
    "4000_8000",
    "ge_8000",
]
_ABS_SCORE_BUCKET_LABELS = [
    "0_50",
    "50_100",
    "100_200",
    "200_400",
    "400_600",
    "600_1000",
    "1000_2000",
    "2000_4000",
    "4000_8000",
    "8000_16000",
    "ge_16000",
]
_PLY_BUCKET_LABELS = ["0_19", "20_39", "40_59", "60_79", "80_99", "100_149", "150_199", "ge_200"]
_PIECE_COUNT_BUCKET_LABELS = ["2_7", "8_15", "16_23", "24_32"]
_PHASE_LABELS = ["tablebase_like", "endgame", "late_middlegame", "opening_middlegame"]
_PIECE_TYPE_LABELS = ["pawns", "knights", "bishops", "rooks", "queens", "kings"]
_WDL_SCALE_LABELS = ["197", "410", "600", "1000", "2000", "4000", "8000"]


@dataclass(slots=True)
class NativeBatch:
    white_indices: np.ndarray
    black_indices: np.ndarray
    stm: np.ndarray
    score_cp: np.ndarray
    result_wdl: np.ndarray


def build_native_extension(force: bool = False) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "build" / "native_binpack"
    build_dir.mkdir(parents=True, exist_ok=True)

    existing = _find_library(build_dir)
    if existing is not None and not force and not _native_sources_newer(repo_root / "native_binpack", existing):
        return existing

    subprocess.run(
        ["cmake", "-S", str(repo_root / "native_binpack"), "-B", str(build_dir)],
        check=True,
        cwd=repo_root,
    )
    subprocess.run(
        ["cmake", "--build", str(build_dir), "--config", "Release"],
        check=True,
        cwd=repo_root,
    )

    lib_path = _find_library(build_dir)
    if lib_path is None:
        raise NativeError(f"Could not find compiled native library under {build_dir}")
    return lib_path


def inspect_binpack(path: str | Path) -> dict[str, int | float]:
    lib = _load_library()
    return _inspect_binpack_with_library(path, lib)


def _inspect_binpack_with_library(path: str | Path, lib: ctypes.CDLL) -> dict[str, int | float]:
    stats = _InspectStats()
    resolved = Path(path).resolve()
    ok = lib.thrawn_inspect_binpack(os.fsencode(resolved), ctypes.byref(stats))
    if ok != 1:
        raise NativeError(_last_error(lib))
    entries_read = int(stats.entries_read)
    result = {
        "path": str(resolved),
        "file_bytes": resolved.stat().st_size,
        "entries_read": int(stats.entries_read),
        "white_to_move": int(stats.white_to_move),
        "black_to_move": int(stats.black_to_move),
        "wins": int(stats.wins),
        "draws": int(stats.draws),
        "losses": int(stats.losses),
        "min_score": float(stats.min_score),
        "max_score": float(stats.max_score),
        "min_ply": int(stats.min_ply),
        "max_ply": int(stats.max_ply),
        "mean_score": float(stats.mean_score),
        "score_std": float(stats.score_std),
        "mean_abs_score": float(stats.mean_abs_score),
        "abs_score_std": float(stats.abs_score_std),
        "mean_ply": float(stats.mean_ply),
        "ply_std": float(stats.ply_std),
        "mean_piece_count": float(stats.mean_piece_count),
        "piece_count_std": float(stats.piece_count_std),
        "mean_non_king_piece_count": float(stats.mean_non_king_piece_count),
        "result_mean": float(stats.result_mean),
        "score_result_correlation": float(stats.score_result_correlation),
        "score_percentiles": {
            "p01": float(stats.score_p01),
            "p05": float(stats.score_p05),
            "p10": float(stats.score_p10),
            "p25": float(stats.score_p25),
            "p50": float(stats.score_p50),
            "p75": float(stats.score_p75),
            "p90": float(stats.score_p90),
            "p95": float(stats.score_p95),
            "p99": float(stats.score_p99),
            "p999": float(stats.score_p999),
        },
        "abs_score_percentiles": {
            "p50": float(stats.abs_score_p50),
            "p75": float(stats.abs_score_p75),
            "p90": float(stats.abs_score_p90),
            "p95": float(stats.abs_score_p95),
            "p99": float(stats.abs_score_p99),
            "p999": float(stats.abs_score_p999),
        },
        "ply_percentiles": {
            "p05": float(stats.ply_p05),
            "p25": float(stats.ply_p25),
            "p50": float(stats.ply_p50),
            "p75": float(stats.ply_p75),
            "p95": float(stats.ply_p95),
            "p99": float(stats.ply_p99),
        },
        "piece_count_percentiles": {
            "p05": float(stats.piece_count_p05),
            "p25": float(stats.piece_count_p25),
            "p50": float(stats.piece_count_p50),
            "p75": float(stats.piece_count_p75),
            "p95": float(stats.piece_count_p95),
        },
        "abs_score_threshold_counts": {
            "ge_1000": int(stats.abs_score_ge_1000),
            "ge_2000": int(stats.abs_score_ge_2000),
            "ge_4000": int(stats.abs_score_ge_4000),
            "ge_8000": int(stats.abs_score_ge_8000),
            "ge_16000": int(stats.abs_score_ge_16000),
        },
        "score_buckets": _bucket_summary(stats.score_bucket_counts, _SCORE_BUCKET_LABELS, entries_read),
        "abs_score_buckets": _bucket_summary(stats.abs_score_bucket_counts, _ABS_SCORE_BUCKET_LABELS, entries_read),
        "ply_buckets": _bucket_summary(stats.ply_bucket_counts, _PLY_BUCKET_LABELS, entries_read),
        "piece_count_buckets": _bucket_summary(
            stats.piece_count_bucket_counts,
            _PIECE_COUNT_BUCKET_LABELS,
            entries_read,
        ),
        "phase_buckets": _phase_summary(stats, entries_read),
        "result_score_alignment": {
            "decisive_result_score_agree": int(stats.result_score_agree),
            "decisive_result_score_disagree": int(stats.result_score_disagree),
            "decisive_result_near_zero_score": int(stats.decisive_result_near_zero_score),
            "draw_high_abs_score": int(stats.draw_high_abs_score),
        },
        "score_by_result": {
            "wins": {
                "mean_score": float(stats.mean_score_win),
                "mean_abs_score": float(stats.mean_abs_score_win),
            },
            "draws": {
                "mean_score": float(stats.mean_score_draw),
                "mean_abs_score": float(stats.mean_abs_score_draw),
            },
            "losses": {
                "mean_score": float(stats.mean_score_loss),
                "mean_abs_score": float(stats.mean_abs_score_loss),
            },
        },
        "material": _material_summary(stats, entries_read),
        "wdl_scale_saturation": _native_wdl_scale_saturation(stats, entries_read),
    }
    result["result_percentages"] = {
        "wins": _safe_fraction(int(stats.wins), entries_read),
        "draws": _safe_fraction(int(stats.draws), entries_read),
        "losses": _safe_fraction(int(stats.losses), entries_read),
    }
    result["abs_score_threshold_fractions"] = {
        key: _safe_fraction(value, entries_read)
        for key, value in result["abs_score_threshold_counts"].items()
    }
    result["result_score_alignment"]["fractions"] = {
        key: _safe_fraction(value, entries_read)
        for key, value in result["result_score_alignment"].items()
    }
    result["wdl_scale_diagnostics"] = _wdl_scale_diagnostics(result)
    result["recommendation"] = _inspect_recommendation(result)
    return result


def inspect_binpack_collection(paths: list[str | Path], *, jobs: int | None = None) -> dict[str, object]:
    resolved_paths = [Path(path).resolve() for path in paths]
    if not resolved_paths:
        raise ValueError("At least one .binpack path is required")

    sorted_paths = sorted(resolved_paths)
    worker_count = _inspection_worker_count(len(sorted_paths), jobs)
    lib = _load_library()

    def inspect_path(path: Path) -> dict[str, int | float]:
        return _inspect_binpack_with_library(path, lib)

    if worker_count == 1:
        stats_list = [inspect_path(path) for path in sorted_paths]
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            stats_list = list(executor.map(inspect_path, sorted_paths))

    per_file = [
        {"path": str(path), "stats": stats}
        for path, stats in zip(sorted_paths, stats_list)
    ]

    aggregate = _aggregate_inspection_stats([item["stats"] for item in per_file])
    return {
        "file_count": len(per_file),
        "aggregate": aggregate,
        "files": per_file,
    }


def discover_binpack_files(path: str | Path) -> list[Path]:
    root = Path(path).resolve()
    if root.is_file():
        if root.suffix != ".binpack":
            raise ValueError(f"Expected a .binpack file or directory, got: {root}")
        return [root]
    if not root.exists():
        raise ValueError(f"Path does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Expected a directory or .binpack file, got: {root}")

    files = sorted(candidate for candidate in root.rglob("*.binpack") if candidate.is_file())
    if not files:
        raise ValueError(f"No .binpack files found under: {root}")
    return files


def write_fixture_binpack(path: str | Path) -> None:
    lib = _load_library()
    ok = lib.thrawn_write_fixture_binpack(os.fsencode(Path(path).resolve()))
    if ok != 1:
        raise NativeError(_last_error(lib))


class BinpackStream:
    def __init__(
        self,
        paths: list[str | Path],
        *,
        num_threads: int = 1,
        cyclic: bool = False,
        skip_capture_positions: bool = False,
        skip_decisive_score_mismatch: bool = False,
        decisive_score_mismatch_margin: float = 0.0,
        skip_draw_score_mismatch: bool = False,
        draw_score_mismatch_margin: float = 0.0,
        max_abs_score: float = 0.0,
    ):
        if not paths:
            raise ValueError("At least one dataset path is required")
        if decisive_score_mismatch_margin < 0.0:
            raise ValueError("decisive_score_mismatch_margin must be >= 0")
        if draw_score_mismatch_margin < 0.0:
            raise ValueError("draw_score_mismatch_margin must be >= 0")
        if max_abs_score < 0.0:
            raise ValueError("max_abs_score must be >= 0")

        self._handle = None
        self._lib = _load_library()
        encoded_paths = [os.fsencode(Path(path).resolve()) for path in paths]
        self._path_array = (ctypes.c_char_p * len(encoded_paths))(*encoded_paths)
        self._handle = self._lib.thrawn_binpack_open_many(
            self._path_array,
            len(encoded_paths),
            num_threads,
            1 if cyclic else 0,
            1 if skip_capture_positions else 0,
            1 if skip_decisive_score_mismatch else 0,
            float(decisive_score_mismatch_margin),
            1 if skip_draw_score_mismatch else 0,
            float(draw_score_mismatch_margin),
            float(max_abs_score),
        )
        if not self._handle:
            raise NativeError(_last_error(self._lib))

    def close(self) -> None:
        if self._handle is not None:
            self._lib.thrawn_binpack_close(self._handle)
            self._handle = None

    def next_batch(self, batch_size: int) -> NativeBatch | None:
        batch_ptr = self._lib.thrawn_binpack_next_batch(self._handle, batch_size)
        if not batch_ptr:
            error = _last_error(self._lib)
            if error:
                raise NativeError(error)
            return None

        try:
            view = batch_ptr.contents
            size = int(view.size)
            max_active = int(view.max_active_features)
            white_indices = np.ctypeslib.as_array(view.white_indices, shape=(size, max_active)).copy()
            black_indices = np.ctypeslib.as_array(view.black_indices, shape=(size, max_active)).copy()
            stm = np.ctypeslib.as_array(view.stm, shape=(size,)).copy()
            score_cp = np.ctypeslib.as_array(view.score_cp, shape=(size,)).copy()
            result_wdl = np.ctypeslib.as_array(view.result_wdl, shape=(size,)).copy()
            return NativeBatch(
                white_indices=white_indices,
                black_indices=black_indices,
                stm=stm,
                score_cp=score_cp,
                result_wdl=result_wdl,
            )
        finally:
            self._lib.thrawn_batch_free(batch_ptr)

    def __enter__(self) -> "BinpackStream":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


def _find_library(build_dir: Path) -> Path | None:
    patterns = ["*.so", "*.dylib", "*.dll"]
    for pattern in patterns:
        matches = list(build_dir.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _load_library() -> ctypes.CDLL:
    lib_path = build_native_extension()
    lib = ctypes.CDLL(str(lib_path))
    try:
        _configure_library_symbols(lib)
        return lib
    except AttributeError:
        lib_path = build_native_extension(force=True)
        lib = ctypes.CDLL(str(lib_path))
        _configure_library_symbols(lib)
        return lib


def _configure_library_symbols(lib: ctypes.CDLL) -> None:
    lib.thrawn_binpack_open_many.argtypes = [
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_double,
    ]
    lib.thrawn_binpack_open_many.restype = ctypes.c_void_p
    lib.thrawn_binpack_close.argtypes = [ctypes.c_void_p]
    lib.thrawn_binpack_close.restype = None
    lib.thrawn_binpack_next_batch.argtypes = [ctypes.c_void_p, ctypes.c_int32]
    lib.thrawn_binpack_next_batch.restype = ctypes.POINTER(_BatchView)
    lib.thrawn_batch_free.argtypes = [ctypes.POINTER(_BatchView)]
    lib.thrawn_batch_free.restype = None
    lib.thrawn_inspect_binpack.argtypes = [ctypes.c_char_p, ctypes.POINTER(_InspectStats)]
    lib.thrawn_inspect_binpack.restype = ctypes.c_int32
    lib.thrawn_write_fixture_binpack.argtypes = [ctypes.c_char_p]
    lib.thrawn_write_fixture_binpack.restype = ctypes.c_int32
    lib.thrawn_last_error.argtypes = []
    lib.thrawn_last_error.restype = ctypes.c_char_p


def _last_error(lib: ctypes.CDLL) -> str:
    raw = lib.thrawn_last_error()
    if not raw:
        return ""
    return raw.decode("utf-8")


def _native_sources_newer(source_root: Path, built_library: Path) -> bool:
    library_mtime = built_library.stat().st_mtime
    for pattern in ("CMakeLists.txt", "**/*.cpp", "**/*.h"):
        for candidate in source_root.glob(pattern):
            if candidate.is_file() and candidate.stat().st_mtime > library_mtime:
                return True
    return False


def _safe_fraction(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count) / float(total)


def _inspection_worker_count(file_count: int, jobs: int | None) -> int:
    if file_count <= 0:
        return 1
    if jobs is not None:
        if jobs <= 0:
            raise ValueError("jobs must be >= 1")
        return min(file_count, jobs)
    return min(file_count, max(1, os.cpu_count() or 1))


def _safe_std_from_parts(sum_value: float, sum_sq: float, total: int) -> float:
    if total <= 0:
        return 0.0
    mean = sum_value / float(total)
    variance = (sum_sq / float(total)) - (mean * mean)
    return math.sqrt(max(0.0, variance))


def _bucket_summary(counts, labels: list[str], total: int) -> dict[str, dict[str, int | float]]:
    return {
        label: {
            "count": int(counts[index]),
            "fraction": _safe_fraction(int(counts[index]), total),
        }
        for index, label in enumerate(labels)
    }


def _phase_summary(stats: _InspectStats, total: int) -> dict[str, dict[str, int | float]]:
    summary: dict[str, dict[str, int | float]] = {}
    for index, label in enumerate(_PHASE_LABELS):
        count = int(stats.phase_counts[index])
        summary[label] = {
            "count": count,
            "fraction": _safe_fraction(count, total),
            "mean_score": float(stats.phase_mean_score[index]),
            "mean_abs_score": float(stats.phase_mean_abs_score[index]),
            "result_mean": float(stats.phase_result_mean[index]),
        }
    return summary


def _piece_counts_by_type(counts) -> dict[str, int]:
    return {
        label: int(counts[index])
        for index, label in enumerate(_PIECE_TYPE_LABELS)
    }


def _piece_means_by_type(counts, total: int) -> dict[str, float]:
    return {
        label: float(_safe_fraction(int(counts[index]), total))
        for index, label in enumerate(_PIECE_TYPE_LABELS)
    }


def _material_summary(stats: _InspectStats, total: int) -> dict[str, object]:
    return {
        "piece_type_counts": _piece_counts_by_type(stats.piece_type_counts),
        "white_piece_type_counts": _piece_counts_by_type(stats.white_piece_counts),
        "black_piece_type_counts": _piece_counts_by_type(stats.black_piece_counts),
        "mean_piece_types_per_position": _piece_means_by_type(stats.piece_type_counts, total),
        "mean_white_piece_types_per_position": _piece_means_by_type(stats.white_piece_counts, total),
        "mean_black_piece_types_per_position": _piece_means_by_type(stats.black_piece_counts, total),
    }


def _native_wdl_scale_saturation(stats: _InspectStats, total: int) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for index, label in enumerate(_WDL_SCALE_LABELS):
        saturated_99 = int(stats.wdl_scale_saturated_99[index])
        saturated_999 = int(stats.wdl_scale_saturated_999[index])
        summary[label] = {
            "signed_target_mean": float(stats.wdl_scale_signed_target_mean[index]),
            "signed_target_std": float(stats.wdl_scale_signed_target_std[index]),
            "abs_target_mean": float(stats.wdl_scale_abs_target_mean[index]),
            "signed_target_le_0.01_or_ge_0.99_count": saturated_99,
            "signed_target_le_0.01_or_ge_0.99_fraction": _safe_fraction(saturated_99, total),
            "signed_target_le_0.001_or_ge_0.999_count": saturated_999,
            "signed_target_le_0.001_or_ge_0.999_fraction": _safe_fraction(saturated_999, total),
        }
    return summary


def _aggregate_bucket_summaries(
    stats_list: list[dict[str, object]],
    key: str,
    labels: list[str],
    total: int,
) -> dict[str, dict[str, int | float]]:
    return {
        label: {
            "count": sum(int(stats[key][label]["count"]) for stats in stats_list),
            "fraction": _safe_fraction(sum(int(stats[key][label]["count"]) for stats in stats_list), total),
        }
        for label in labels
    }


def _aggregate_phase_summaries(
    stats_list: list[dict[str, object]],
    total: int,
) -> dict[str, dict[str, int | float]]:
    summary: dict[str, dict[str, int | float]] = {}
    for label in _PHASE_LABELS:
        count = sum(int(stats["phase_buckets"][label]["count"]) for stats in stats_list)

        def phase_weighted_mean(metric: str) -> float:
            if count <= 0:
                return 0.0
            return sum(
                float(stats["phase_buckets"][label][metric]) * int(stats["phase_buckets"][label]["count"])
                for stats in stats_list
            ) / float(count)

        summary[label] = {
            "count": count,
            "fraction": _safe_fraction(count, total),
            "mean_score": phase_weighted_mean("mean_score"),
            "mean_abs_score": phase_weighted_mean("mean_abs_score"),
            "result_mean": phase_weighted_mean("result_mean"),
        }
    return summary


def _aggregate_alignment_summaries(
    stats_list: list[dict[str, object]],
    total: int,
) -> dict[str, object]:
    keys = [
        "decisive_result_score_agree",
        "decisive_result_score_disagree",
        "decisive_result_near_zero_score",
        "draw_high_abs_score",
    ]
    summary: dict[str, object] = {
        key: sum(int(stats["result_score_alignment"][key]) for stats in stats_list)
        for key in keys
    }
    summary["fractions"] = {
        key: _safe_fraction(int(summary[key]), total)
        for key in keys
    }
    return summary


def _aggregate_score_by_result(stats_list: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    result_count_keys = {"wins": "wins", "draws": "draws", "losses": "losses"}
    summary: dict[str, dict[str, float]] = {}
    for label, count_key in result_count_keys.items():
        count = sum(int(stats[count_key]) for stats in stats_list)

        def weighted(metric: str) -> float:
            if count <= 0:
                return 0.0
            return sum(
                float(stats["score_by_result"][label][metric]) * int(stats[count_key])
                for stats in stats_list
            ) / float(count)

        summary[label] = {
            "mean_score": weighted("mean_score"),
            "mean_abs_score": weighted("mean_abs_score"),
        }
    return summary


def _aggregate_material_summaries(
    stats_list: list[dict[str, object]],
    total: int,
) -> dict[str, object]:
    def sum_counts(key: str) -> dict[str, int]:
        return {
            label: sum(int(stats["material"][key][label]) for stats in stats_list)
            for label in _PIECE_TYPE_LABELS
        }

    piece_counts = sum_counts("piece_type_counts")
    white_counts = sum_counts("white_piece_type_counts")
    black_counts = sum_counts("black_piece_type_counts")
    return {
        "piece_type_counts": piece_counts,
        "white_piece_type_counts": white_counts,
        "black_piece_type_counts": black_counts,
        "mean_piece_types_per_position": {
            label: _safe_fraction(piece_counts[label], total)
            for label in _PIECE_TYPE_LABELS
        },
        "mean_white_piece_types_per_position": {
            label: _safe_fraction(white_counts[label], total)
            for label in _PIECE_TYPE_LABELS
        },
        "mean_black_piece_types_per_position": {
            label: _safe_fraction(black_counts[label], total)
            for label in _PIECE_TYPE_LABELS
        },
    }


def _aggregate_wdl_scale_saturation(
    stats_list: list[dict[str, object]],
    total: int,
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for label in _WDL_SCALE_LABELS:
        signed_sum = sum(
            float(stats["wdl_scale_saturation"][label]["signed_target_mean"]) * int(stats["entries_read"])
            for stats in stats_list
        )
        signed_sum_sq = sum(
            (
                float(stats["wdl_scale_saturation"][label]["signed_target_std"]) ** 2
                + float(stats["wdl_scale_saturation"][label]["signed_target_mean"]) ** 2
            )
            * int(stats["entries_read"])
            for stats in stats_list
        )
        abs_sum = sum(
            float(stats["wdl_scale_saturation"][label]["abs_target_mean"]) * int(stats["entries_read"])
            for stats in stats_list
        )
        saturated_99 = sum(
            int(stats["wdl_scale_saturation"][label]["signed_target_le_0.01_or_ge_0.99_count"])
            for stats in stats_list
        )
        saturated_999 = sum(
            int(stats["wdl_scale_saturation"][label]["signed_target_le_0.001_or_ge_0.999_count"])
            for stats in stats_list
        )
        summary[label] = {
            "signed_target_mean": signed_sum / float(total),
            "signed_target_std": _safe_std_from_parts(signed_sum, signed_sum_sq, total),
            "abs_target_mean": abs_sum / float(total),
            "signed_target_le_0.01_or_ge_0.99_count": saturated_99,
            "signed_target_le_0.01_or_ge_0.99_fraction": _safe_fraction(saturated_99, total),
            "signed_target_le_0.001_or_ge_0.999_count": saturated_999,
            "signed_target_le_0.001_or_ge_0.999_fraction": _safe_fraction(saturated_999, total),
        }
    return summary


def _aggregate_inspection_stats(stats_list: list[dict[str, object]]) -> dict[str, object]:
    total_entries = sum(int(stats["entries_read"]) for stats in stats_list)
    if total_entries <= 0:
        raise ValueError("Inspection aggregate requires at least one dataset entry")

    def weighted_mean(key: str) -> float:
        numerator = sum(float(stats[key]) * int(stats["entries_read"]) for stats in stats_list)
        return numerator / total_entries

    def aggregate_std(mean_key: str, std_key: str) -> float:
        total_sum = sum(float(stats[mean_key]) * int(stats["entries_read"]) for stats in stats_list)
        total_sum_sq = sum(
            ((float(stats[std_key]) ** 2) + (float(stats[mean_key]) ** 2)) * int(stats["entries_read"])
            for stats in stats_list
        )
        return _safe_std_from_parts(total_sum, total_sum_sq, total_entries)

    aggregate: dict[str, object] = {
        "file_bytes": sum(int(stats["file_bytes"]) for stats in stats_list),
        "entries_read": total_entries,
        "white_to_move": sum(int(stats["white_to_move"]) for stats in stats_list),
        "black_to_move": sum(int(stats["black_to_move"]) for stats in stats_list),
        "wins": sum(int(stats["wins"]) for stats in stats_list),
        "draws": sum(int(stats["draws"]) for stats in stats_list),
        "losses": sum(int(stats["losses"]) for stats in stats_list),
        "min_score": min(float(stats["min_score"]) for stats in stats_list),
        "max_score": max(float(stats["max_score"]) for stats in stats_list),
        "min_ply": min(int(stats["min_ply"]) for stats in stats_list),
        "max_ply": max(int(stats["max_ply"]) for stats in stats_list),
        "mean_score": weighted_mean("mean_score"),
        "score_std": aggregate_std("mean_score", "score_std"),
        "mean_abs_score": weighted_mean("mean_abs_score"),
        "abs_score_std": aggregate_std("mean_abs_score", "abs_score_std"),
        "mean_ply": weighted_mean("mean_ply"),
        "ply_std": aggregate_std("mean_ply", "ply_std"),
        "mean_piece_count": weighted_mean("mean_piece_count"),
        "piece_count_std": aggregate_std("mean_piece_count", "piece_count_std"),
        "mean_non_king_piece_count": weighted_mean("mean_non_king_piece_count"),
        "result_mean": weighted_mean("result_mean"),
        "score_result_correlation": weighted_mean("score_result_correlation"),
        "abs_score_threshold_counts": {
            threshold: sum(int(stats["abs_score_threshold_counts"][threshold]) for stats in stats_list)
            for threshold in ["ge_1000", "ge_2000", "ge_4000", "ge_8000", "ge_16000"]
        },
    }
    aggregate["result_percentages"] = {
        "wins": _safe_fraction(int(aggregate["wins"]), total_entries),
        "draws": _safe_fraction(int(aggregate["draws"]), total_entries),
        "losses": _safe_fraction(int(aggregate["losses"]), total_entries),
    }
    aggregate["abs_score_threshold_fractions"] = {
        key: _safe_fraction(value, total_entries)
        for key, value in aggregate["abs_score_threshold_counts"].items()
    }
    aggregate["score_buckets"] = _aggregate_bucket_summaries(stats_list, "score_buckets", _SCORE_BUCKET_LABELS, total_entries)
    aggregate["abs_score_buckets"] = _aggregate_bucket_summaries(
        stats_list,
        "abs_score_buckets",
        _ABS_SCORE_BUCKET_LABELS,
        total_entries,
    )
    aggregate["ply_buckets"] = _aggregate_bucket_summaries(stats_list, "ply_buckets", _PLY_BUCKET_LABELS, total_entries)
    aggregate["piece_count_buckets"] = _aggregate_bucket_summaries(
        stats_list,
        "piece_count_buckets",
        _PIECE_COUNT_BUCKET_LABELS,
        total_entries,
    )
    aggregate["phase_buckets"] = _aggregate_phase_summaries(stats_list, total_entries)
    aggregate["result_score_alignment"] = _aggregate_alignment_summaries(stats_list, total_entries)
    aggregate["score_by_result"] = _aggregate_score_by_result(stats_list)
    aggregate["material"] = _aggregate_material_summaries(stats_list, total_entries)
    aggregate["wdl_scale_saturation"] = _aggregate_wdl_scale_saturation(stats_list, total_entries)

    # Exact percentiles cannot be reconstructed from per-file summaries alone, so
    # aggregate percentiles are conservative extrema of the file-level estimates.
    score_lower_percentiles = ["p01", "p05", "p10", "p25"]
    score_upper_percentiles = ["p50", "p75", "p90", "p95", "p99", "p999"]
    aggregate["score_percentiles"] = {
        **{
            percentile: min(float(stats["score_percentiles"][percentile]) for stats in stats_list)
            for percentile in score_lower_percentiles
        },
        **{
            percentile: max(float(stats["score_percentiles"][percentile]) for stats in stats_list)
            for percentile in score_upper_percentiles
        },
    }
    aggregate["abs_score_percentiles"] = {
        percentile: max(float(stats["abs_score_percentiles"][percentile]) for stats in stats_list)
        for percentile in ["p50", "p75", "p90", "p95", "p99", "p999"]
    }
    aggregate["ply_percentiles"] = {
        **{
            percentile: min(float(stats["ply_percentiles"][percentile]) for stats in stats_list)
            for percentile in ["p05", "p25"]
        },
        **{
            percentile: max(float(stats["ply_percentiles"][percentile]) for stats in stats_list)
            for percentile in ["p50", "p75", "p95", "p99"]
        },
    }
    aggregate["piece_count_percentiles"] = {
        **{
            percentile: min(float(stats["piece_count_percentiles"][percentile]) for stats in stats_list)
            for percentile in ["p05", "p25"]
        },
        **{
            percentile: max(float(stats["piece_count_percentiles"][percentile]) for stats in stats_list)
            for percentile in ["p50", "p75", "p95"]
        },
    }
    aggregate["wdl_scale_diagnostics"] = _wdl_scale_diagnostics(aggregate)
    aggregate["recommendation"] = _inspect_recommendation(aggregate)
    aggregate["aggregate_notes"] = [
        "entries, result counts, threshold counts, minima, maxima, and means are exact aggregates",
        "percentiles are conservative extrema of per-file percentiles, not exact merged percentiles",
    ]
    return aggregate


def _wdl_scale_diagnostics(stats: dict[str, object]) -> dict[str, dict[str, float]]:
    mean_abs_score = float(stats["mean_abs_score"])
    candidate_scales = [197.0, 410.0, 600.0, 1000.0, 2000.0, 4000.0, 8000.0]
    diagnostics: dict[str, dict[str, float]] = {}
    for scale in candidate_scales:
        transformed = _wdl_target(mean_abs_score, scale)
        p95_transformed = _wdl_target(float(stats["abs_score_percentiles"]["p95"]), scale)
        native_scale = stats.get("wdl_scale_saturation", {}).get(str(int(scale)), {})
        diagnostics[str(int(scale))] = {
            "mean_abs_score_target": float(transformed),
            "p95_abs_score_target": float(p95_transformed),
            "high_saturation_proxy": float(transformed >= 0.98 or p95_transformed >= 0.995),
            "signed_target_le_0.01_or_ge_0.99_fraction": float(
                native_scale.get("signed_target_le_0.01_or_ge_0.99_fraction", 0.0)
            ),
            "signed_target_le_0.001_or_ge_0.999_fraction": float(
                native_scale.get("signed_target_le_0.001_or_ge_0.999_fraction", 0.0)
            ),
        }
    return diagnostics


def _wdl_target(abs_score: float, effective_raw_wdl_scale: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(abs_score / effective_raw_wdl_scale))))


def _recommend_wdl_scale(mean_abs_score: float, abs_p95: float) -> float:
    for candidate in [410.0, 1000.0, 2000.0, 4000.0, 8000.0]:
        mean_target = _wdl_target(mean_abs_score, candidate)
        p95_target = _wdl_target(abs_p95, candidate)
        if mean_target < 0.95 and p95_target < 0.995:
            return candidate
    return 8000.0


def _teacher_target_collapse_risk(mean_abs_score: float, abs_p95: float, effective_raw_wdl_scale: float) -> bool:
    mean_target = _wdl_target(mean_abs_score, effective_raw_wdl_scale)
    p95_target = _wdl_target(abs_p95, effective_raw_wdl_scale)
    return mean_target < 0.55 and p95_target < 0.60


def _inspect_recommendation(stats: dict[str, object]) -> dict[str, object]:
    entries_read = int(stats.get("entries_read", 0))
    abs_p95 = float(stats["abs_score_percentiles"]["p95"])
    abs_p99 = float(stats["abs_score_percentiles"]["p99"])
    mean_abs_score = float(stats["mean_abs_score"])
    diagnostics = stats["wdl_scale_diagnostics"]

    if entries_read == 0:
        return {
            "saturated_at_default_wdl_scale": False,
            "recommended_wdl_scale": 410.0,
            "recommended_score_clip": 0.0,
            "effective_raw_wdl_scale": 410.0,
            "effective_mean_abs_score_target": 0.5,
            "effective_p95_abs_score_target": 0.5,
            "teacher_target_collapse_risk": False,
            "notes": ["dataset is empty or unreadable; verify the binpack path before using this recommendation"],
        }

    recommended_score_clip = 0.0
    if abs_p99 >= 16000:
        recommended_score_clip = 16000.0
    elif abs_p99 >= 8000:
        recommended_score_clip = 8000.0

    recommended_wdl_scale = _recommend_wdl_scale(mean_abs_score, abs_p95)
    effective_raw_wdl_scale = recommended_wdl_scale
    effective_mean_abs_score_target = _wdl_target(mean_abs_score, effective_raw_wdl_scale)
    effective_p95_abs_score_target = _wdl_target(abs_p95, effective_raw_wdl_scale)
    teacher_target_collapse_risk = _teacher_target_collapse_risk(
        mean_abs_score,
        abs_p95,
        effective_raw_wdl_scale,
    )

    saturated_at_410 = bool(diagnostics["410"]["high_saturation_proxy"] >= 0.5)
    notes = []
    if saturated_at_410:
        notes.append("dataset appears highly saturated for wdl_scale=410")
    if recommended_score_clip > 0.0:
        notes.append("extreme score tails suggest enabling score clipping")
    if teacher_target_collapse_risk:
        notes.append("recommended wdl_scale still keeps teacher targets too close to 0.5")
    if not notes:
        notes.append("current score distribution looks usable without aggressive normalization")

    return {
        "saturated_at_default_wdl_scale": saturated_at_410,
        "recommended_wdl_scale": recommended_wdl_scale,
        "recommended_score_clip": recommended_score_clip,
        "effective_raw_wdl_scale": effective_raw_wdl_scale,
        "effective_mean_abs_score_target": effective_mean_abs_score_target,
        "effective_p95_abs_score_target": effective_p95_abs_score_target,
        "teacher_target_collapse_risk": teacher_target_collapse_risk,
        "notes": notes,
    }
