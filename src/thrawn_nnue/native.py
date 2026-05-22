from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import ctypes
import math
import os
import subprocess

import numpy as np

from .features import FEATURES, HALFKA_V2_HM_MAX_ACTIVE_FEATURES, HALFKA_V2_HM_NUM_FEATURES, HALFKA_V2_HM_PS_NB


class NativeError(RuntimeError):
    pass


class _BatchView(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int32),
        ("max_active_features", ctypes.c_int32),
        ("white_indices", ctypes.POINTER(ctypes.c_int32)),
        ("black_indices", ctypes.POINTER(ctypes.c_int32)),
        ("stm", ctypes.POINTER(ctypes.c_float)),
        ("score", ctypes.POINTER(ctypes.c_float)),
        ("result_wdl", ctypes.POINTER(ctypes.c_float)),
    ]


class _InspectStats(ctypes.Structure):
    _fields_ = [
        ("entries_seen", ctypes.c_uint64),
        ("entries_skipped", ctypes.c_uint64),
        ("entries_read", ctypes.c_uint64),
        ("root_entries", ctypes.c_uint64),
        ("continuation_entries", ctypes.c_uint64),
        ("chunk_count", ctypes.c_uint64),
        ("chunk_payload_bytes", ctypes.c_uint64),
        ("min_chunk_payload_bytes", ctypes.c_uint32),
        ("max_chunk_payload_bytes", ctypes.c_uint32),
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
        ("mean_raw_score", ctypes.c_double),
        ("raw_score_std", ctypes.c_double),
        ("mean_abs_score", ctypes.c_double),
        ("abs_score_std", ctypes.c_double),
        ("mean_ply", ctypes.c_double),
        ("ply_std", ctypes.c_double),
        ("mean_piece_count", ctypes.c_double),
        ("piece_count_std", ctypes.c_double),
        ("mean_non_king_piece_count", ctypes.c_double),
        ("result_mean", ctypes.c_double),
        ("result_wdl_mean", ctypes.c_double),
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
        ("capture_positions", ctypes.c_uint64),
        ("in_check_positions", ctypes.c_uint64),
        ("move_type_counts", ctypes.c_uint64 * 4),
        ("piece_type_counts", ctypes.c_uint64 * 6),
        ("white_piece_counts", ctypes.c_uint64 * 6),
        ("black_piece_counts", ctypes.c_uint64 * 6),
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
_MOVE_TYPE_LABELS = ["normal", "promotion", "castle", "en_passant"]
_STOCKFISH_INTERNAL_TO_CP = 100.0 / 208.0
_SPLIT_ROLE_IDS = {
    "all": 0,
    "train": 1,
    "validation": 2,
}
@dataclass(slots=True)
class NativeBatch:
    white_indices: np.ndarray
    black_indices: np.ndarray
    stm: np.ndarray
    score: np.ndarray
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


def inspect_binpack(
    path: str | Path,
    *,
    skip_wdl_score_mismatch: bool = False,
    sample_entries: int = 0,
) -> dict[str, object]:
    lib = _load_library()
    return _inspect_binpack_with_library(
        path,
        lib,
        skip_wdl_score_mismatch=skip_wdl_score_mismatch,
        sample_entries=sample_entries,
    )


def _inspect_binpack_with_library(
    path: str | Path,
    lib: ctypes.CDLL,
    *,
    skip_wdl_score_mismatch: bool = False,
    sample_entries: int = 0,
) -> dict[str, object]:
    if sample_entries < 0:
        raise ValueError("sample_entries must be >= 0")
    stats = _InspectStats()
    resolved = Path(path).resolve()
    ok = lib.thrawn_inspect_binpack_with_options(
        os.fsencode(resolved),
        ctypes.byref(stats),
        1 if skip_wdl_score_mismatch else 0,
        int(sample_entries),
    )
    if ok != 1:
        raise NativeError(_last_error(lib))
    entries_read = int(stats.entries_read)
    file_bytes = resolved.stat().st_size
    result = {
        "path": str(resolved),
        "file_bytes": file_bytes,
        "format": _binpack_format_summary(),
        "file": _file_summary(stats, file_bytes),
        "filters": {
            "skip_wdl_score_mismatch": bool(skip_wdl_score_mismatch),
            "sample_entries": int(sample_entries),
        },
        "sampled": sample_entries > 0,
        "entries_seen": int(stats.entries_seen),
        "entries_skipped_by_filters": int(stats.entries_skipped),
        "entries_read": int(stats.entries_read),
        "root_entries": int(stats.root_entries),
        "continuation_entries": int(stats.continuation_entries),
        "white_to_move": int(stats.white_to_move),
        "black_to_move": int(stats.black_to_move),
        "wins": int(stats.wins),
        "draws": int(stats.draws),
        "losses": int(stats.losses),
        "min_score": float(stats.min_score),
        "max_score": float(stats.max_score),
        "min_raw_score": _cp_to_raw_score(float(stats.min_score)),
        "max_raw_score": _cp_to_raw_score(float(stats.max_score)),
        "min_ply": int(stats.min_ply),
        "max_ply": int(stats.max_ply),
        "mean_score": float(stats.mean_score),
        "score_std": float(stats.score_std),
        "mean_raw_score": float(stats.mean_raw_score),
        "raw_score_std": float(stats.raw_score_std),
        "mean_abs_score": float(stats.mean_abs_score),
        "abs_score_std": float(stats.abs_score_std),
        "mean_ply": float(stats.mean_ply),
        "ply_std": float(stats.ply_std),
        "mean_piece_count": float(stats.mean_piece_count),
        "piece_count_std": float(stats.piece_count_std),
        "mean_non_king_piece_count": float(stats.mean_non_king_piece_count),
        "result_mean": float(stats.result_mean),
        "result_wdl_mean": float(stats.result_wdl_mean),
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
        "move_types": _move_type_summary(stats, entries_read),
        "position_flags": {
            "capture_positions": {
                "count": int(stats.capture_positions),
                "fraction": _safe_fraction(int(stats.capture_positions), entries_read),
            },
            "in_check_positions": {
                "count": int(stats.in_check_positions),
                "fraction": _safe_fraction(int(stats.in_check_positions), entries_read),
            },
        },
        "material": _material_summary(stats, entries_read),
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
    result["raw_score_percentiles"] = {
        key: _cp_to_raw_score(value)
        for key, value in result["score_percentiles"].items()
    }
    result["abs_raw_score_percentiles"] = {
        key: _cp_to_raw_score(value)
        for key, value in result["abs_score_percentiles"].items()
    }
    result["result_score_alignment"]["fractions"] = {
        key: _safe_fraction(value, entries_read)
        for key, value in result["result_score_alignment"].items()
    }
    if sample_entries > 0:
        sampled_scan_bytes = int(stats.chunk_payload_bytes) + int(stats.chunk_count) * 8
        file_summary = result["file"]
        file_summary["sample_scan_bytes"] = sampled_scan_bytes
        file_summary["bytes_per_entry_seen"] = _safe_fraction(sampled_scan_bytes, int(stats.entries_seen))
        file_summary["bytes_per_entry_read"] = _safe_fraction(sampled_scan_bytes, entries_read)
    result["wdl"] = _wdl_summary(result)
    return result


def inspect_binpack_collection(
    paths: list[str | Path],
    *,
    jobs: int | None = None,
    skip_wdl_score_mismatch: bool = False,
    sample_entries: int = 0,
) -> dict[str, object]:
    resolved_paths = [Path(path).resolve() for path in paths]
    if not resolved_paths:
        raise ValueError("At least one .binpack path is required")
    if sample_entries < 0:
        raise ValueError("sample_entries must be >= 0")

    sorted_paths = sorted(resolved_paths)
    worker_count = _inspection_worker_count(len(sorted_paths), jobs)
    lib = _load_library()

    def inspect_path(path: Path) -> dict[str, object]:
        return _inspect_binpack_with_library(
            path,
            lib,
            skip_wdl_score_mismatch=skip_wdl_score_mismatch,
            sample_entries=sample_entries,
        )

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
        "format": _binpack_format_summary(),
        "filters": {
            "skip_wdl_score_mismatch": bool(skip_wdl_score_mismatch),
            "sample_entries": int(sample_entries),
        },
        "sampled": sample_entries > 0,
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
        skip_wdl_score_mismatch: bool = False,
        skip_tactical_positions: bool = False,
        random_fen_skipping: int = 0,
        early_fen_skipping: int = -1,
        soft_early_fen_skipping: int = 0,
        simple_eval_skipping: int = -1,
        param_index: int = 0,
        pc_y0: float = 0.0,
        pc_y1: float = 0.4,
        pc_y2: float = 1.0,
        pc_y3: float = 1.0,
        pc_y4: float = 0.75,
        ply_x1: float = 0.0,
        ply_y1: float = 0.1,
        ply_x2: float = 6.0,
        ply_y2: float = 0.15,
        ply_x3: float = 10.0,
        ply_y3: float = 0.25,
        ply_x4: float = 18.0,
        ply_y4: float = 0.75,
        split_role: str = "all",
        validation_split_fraction: float = 0.0,
    ):
        if not paths:
            raise ValueError("At least one dataset path is required")
        if validation_split_fraction < 0.0 or validation_split_fraction >= 1.0:
            raise ValueError("validation_split_fraction must be >= 0 and < 1")
        if random_fen_skipping < 0:
            raise ValueError("random_fen_skipping must be >= 0")
        if early_fen_skipping < -1:
            raise ValueError("early_fen_skipping must be >= -1")
        if simple_eval_skipping < -1:
            raise ValueError("simple_eval_skipping must be >= -1")
        if param_index < 0:
            raise ValueError("param_index must be >= 0")
        for name, value in (
            ("pc_y0", pc_y0),
            ("pc_y1", pc_y1),
            ("pc_y2", pc_y2),
            ("pc_y3", pc_y3),
            ("pc_y4", pc_y4),
            ("ply_x1", ply_x1),
            ("ply_y1", ply_y1),
            ("ply_x2", ply_x2),
            ("ply_y2", ply_y2),
            ("ply_x3", ply_x3),
            ("ply_y3", ply_y3),
            ("ply_x4", ply_x4),
            ("ply_y4", ply_y4),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if split_role not in _SPLIT_ROLE_IDS:
            allowed = ", ".join(sorted(_SPLIT_ROLE_IDS))
            raise ValueError(f"split_role must be one of: {allowed}")
        self._handle = None
        self._lib = _load_library()
        encoded_paths = [os.fsencode(Path(path).resolve()) for path in paths]
        self._path_array = (ctypes.c_char_p * len(encoded_paths))(*encoded_paths)
        self._handle = self._lib.thrawn_binpack_open_many(
            self._path_array,
            len(encoded_paths),
            num_threads,
            1 if cyclic else 0,
            1 if skip_wdl_score_mismatch else 0,
            1 if skip_tactical_positions else 0,
            int(random_fen_skipping),
            int(early_fen_skipping),
            int(soft_early_fen_skipping),
            int(simple_eval_skipping),
            int(param_index),
            float(pc_y0),
            float(pc_y1),
            float(pc_y2),
            float(pc_y3),
            float(pc_y4),
            float(ply_x1),
            float(ply_y1),
            float(ply_x2),
            float(ply_y2),
            float(ply_x3),
            float(ply_y3),
            float(ply_x4),
            float(ply_y4),
            _SPLIT_ROLE_IDS[split_role],
            float(validation_split_fraction),
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
            score = np.ctypeslib.as_array(view.score, shape=(size,)).copy()
            result_wdl = np.ctypeslib.as_array(view.result_wdl, shape=(size,)).copy()
            return NativeBatch(
                white_indices=white_indices,
                black_indices=black_indices,
                stm=stm,
                score=score,
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
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int32,
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
    lib.thrawn_inspect_binpack_with_options.argtypes = [
        ctypes.c_char_p,
        ctypes.POINTER(_InspectStats),
        ctypes.c_int32,
        ctypes.c_uint64,
    ]
    lib.thrawn_inspect_binpack_with_options.restype = ctypes.c_int32
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


def _cp_to_raw_score(score_cp: float) -> float:
    return float(score_cp) / _STOCKFISH_INTERNAL_TO_CP


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


def _binpack_format_summary() -> dict[str, object]:
    return {
        "name": "nnue-pytorch binpack",
        "container": {
            "chunk_magic": "BINP",
            "chunk_header_bytes": 8,
            "chunk_payload_size_bytes": 4,
            "chunk_payload_size_endianness": "little",
            "writer_target_chunk_payload_bytes": 1024 * 1024,
            "max_supported_chunk_payload_bytes": 100 * 1024 * 1024,
        },
        "entry": {
            "root_entry_bytes": 32,
            "continuation_count_bytes": 2,
            "continuation_encoding": "bit-packed legal move ids plus variable-length signed score deltas",
            "score_raw_type": "int16 Stockfish internal eval",
            "score_cp_multiplier": _STOCKFISH_INTERNAL_TO_CP,
            "score_cp_formula": "score_cp = raw_score * 100 / 208",
            "ply_bits": 14,
            "result_raw_values": {"win": 1, "draw": 0, "loss": -1},
            "result_wdl_values": {"win": 1.0, "draw": 0.5, "loss": 0.0},
        },
        "features": {
            "feature_set": FEATURES,
            "feature_count": HALFKA_V2_HM_NUM_FEATURES,
            "piece_square_features": HALFKA_V2_HM_PS_NB,
            "max_active_features_per_perspective": HALFKA_V2_HM_MAX_ACTIVE_FEATURES,
            "perspectives": ["white", "black"],
            "king_pieces_in_feature_lists": True,
        },
    }


def _file_summary(stats: _InspectStats, file_bytes: int) -> dict[str, int | float]:
    chunk_count = int(stats.chunk_count)
    chunk_payload_bytes = int(stats.chunk_payload_bytes)
    entries_seen = int(stats.entries_seen)
    entries_read = int(stats.entries_read)
    return {
        "bytes": int(file_bytes),
        "chunk_count": chunk_count,
        "chunk_payload_bytes": chunk_payload_bytes,
        "chunk_header_overhead_bytes": chunk_count * 8,
        "min_chunk_payload_bytes": int(stats.min_chunk_payload_bytes),
        "max_chunk_payload_bytes": int(stats.max_chunk_payload_bytes),
        "mean_chunk_payload_bytes": _safe_fraction(chunk_payload_bytes, chunk_count),
        "bytes_per_entry_seen": _safe_fraction(int(file_bytes), entries_seen),
        "bytes_per_entry_read": _safe_fraction(int(file_bytes), entries_read),
    }


def _move_type_summary(stats: _InspectStats, total: int) -> dict[str, dict[str, int | float]]:
    return _bucket_summary(stats.move_type_counts, _MOVE_TYPE_LABELS, total)


def _bucket_summary(counts, labels: list[str], total: int) -> dict[str, dict[str, int | float]]:
    return {
        label: {
            "count": int(counts[index]),
            "fraction": _safe_fraction(int(counts[index]), total),
        }
        for index, label in enumerate(labels)
    }


def _wdl_summary(stats: dict[str, object]) -> dict[str, object]:
    entries_read = int(stats["entries_read"])
    return {
        "stored_result_targets": {
            "wins": {
                "count": int(stats["wins"]),
                "fraction": _safe_fraction(int(stats["wins"]), entries_read),
                "wdl_value": 1.0,
                "signed_value": 1.0,
            },
            "draws": {
                "count": int(stats["draws"]),
                "fraction": _safe_fraction(int(stats["draws"]), entries_read),
                "wdl_value": 0.5,
                "signed_value": 0.0,
            },
            "losses": {
                "count": int(stats["losses"]),
                "fraction": _safe_fraction(int(stats["losses"]), entries_read),
                "wdl_value": 0.0,
                "signed_value": -1.0,
            },
        },
        "result_wdl_mean": float(stats["result_wdl_mean"]),
        "signed_result_mean": float(stats["result_mean"]),
        "teacher_target_formula": {
            "score_input": "raw_score",
            "win": "sigmoid((raw_score - offset) / scale)",
            "loss": "sigmoid((-raw_score - offset) / scale)",
            "draw": "1 - win - loss",
            "expectation": "win + 0.5 * draw",
        },
        "score_result_filter": {
            "name": "skip_wdl_score_mismatch",
            "enabled": bool(stats["filters"]["skip_wdl_score_mismatch"]),
            "model": "TrainingDataEntry::score_result_prob() Stockfish win/draw/loss model over raw score and ply",
            "effect": "when enabled, training entries are stochastically downsampled and validation inspection uses a deterministic hash draw",
        },
        "score_result_alignment": stats["result_score_alignment"],
        "score_by_result": stats["score_by_result"],
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


def _aggregate_position_flags(
    stats_list: list[dict[str, object]],
    total: int,
) -> dict[str, dict[str, int | float]]:
    keys = ["capture_positions", "in_check_positions"]
    return {
        key: {
            "count": sum(int(stats["position_flags"][key]["count"]) for stats in stats_list),
            "fraction": _safe_fraction(
                sum(int(stats["position_flags"][key]["count"]) for stats in stats_list),
                total,
            ),
        }
        for key in keys
    }


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


def _aggregate_inspection_stats(stats_list: list[dict[str, object]]) -> dict[str, object]:
    total_entries = sum(int(stats["entries_read"]) for stats in stats_list)
    total_seen = sum(int(stats["entries_seen"]) for stats in stats_list)

    def weighted_mean(key: str) -> float:
        if total_entries <= 0:
            return 0.0
        numerator = sum(float(stats[key]) * int(stats["entries_read"]) for stats in stats_list)
        return numerator / total_entries

    def aggregate_std(mean_key: str, std_key: str) -> float:
        if total_entries <= 0:
            return 0.0
        total_sum = sum(float(stats[mean_key]) * int(stats["entries_read"]) for stats in stats_list)
        total_sum_sq = sum(
            ((float(stats[std_key]) ** 2) + (float(stats[mean_key]) ** 2)) * int(stats["entries_read"])
            for stats in stats_list
        )
        return _safe_std_from_parts(total_sum, total_sum_sq, total_entries)

    chunk_count = sum(int(stats["file"]["chunk_count"]) for stats in stats_list)
    chunk_payload_bytes = sum(int(stats["file"]["chunk_payload_bytes"]) for stats in stats_list)
    sampled_scan_bytes = sum(int(stats["file"].get("sample_scan_bytes", 0)) for stats in stats_list)
    sampled = any(bool(stats.get("sampled")) for stats in stats_list)
    entry_byte_basis = sampled_scan_bytes if sampled and sampled_scan_bytes > 0 else sum(
        int(stats["file_bytes"]) for stats in stats_list
    )
    non_empty_chunk_mins = [
        int(stats["file"]["min_chunk_payload_bytes"])
        for stats in stats_list
        if int(stats["file"]["chunk_count"]) > 0
    ]

    aggregate: dict[str, object] = {
        "file_bytes": sum(int(stats["file_bytes"]) for stats in stats_list),
        "filters": dict(stats_list[0].get("filters", {})),
        "file": {
            "bytes": sum(int(stats["file_bytes"]) for stats in stats_list),
            "chunk_count": chunk_count,
            "chunk_payload_bytes": chunk_payload_bytes,
            "chunk_header_overhead_bytes": chunk_count * 8,
            "min_chunk_payload_bytes": min(non_empty_chunk_mins) if non_empty_chunk_mins else 0,
            "max_chunk_payload_bytes": max(int(stats["file"]["max_chunk_payload_bytes"]) for stats in stats_list),
            "mean_chunk_payload_bytes": _safe_fraction(chunk_payload_bytes, chunk_count),
            "bytes_per_entry_seen": _safe_fraction(entry_byte_basis, total_seen),
            "bytes_per_entry_read": _safe_fraction(entry_byte_basis, total_entries),
        },
        "entries_seen": total_seen,
        "entries_skipped_by_filters": sum(int(stats["entries_skipped_by_filters"]) for stats in stats_list),
        "entries_read": total_entries,
        "root_entries": sum(int(stats["root_entries"]) for stats in stats_list),
        "continuation_entries": sum(int(stats["continuation_entries"]) for stats in stats_list),
        "white_to_move": sum(int(stats["white_to_move"]) for stats in stats_list),
        "black_to_move": sum(int(stats["black_to_move"]) for stats in stats_list),
        "wins": sum(int(stats["wins"]) for stats in stats_list),
        "draws": sum(int(stats["draws"]) for stats in stats_list),
        "losses": sum(int(stats["losses"]) for stats in stats_list),
        "min_score": min(float(stats["min_score"]) for stats in stats_list),
        "max_score": max(float(stats["max_score"]) for stats in stats_list),
        "min_raw_score": min(float(stats["min_raw_score"]) for stats in stats_list),
        "max_raw_score": max(float(stats["max_raw_score"]) for stats in stats_list),
        "min_ply": min(int(stats["min_ply"]) for stats in stats_list),
        "max_ply": max(int(stats["max_ply"]) for stats in stats_list),
        "mean_score": weighted_mean("mean_score"),
        "score_std": aggregate_std("mean_score", "score_std"),
        "mean_raw_score": weighted_mean("mean_raw_score"),
        "raw_score_std": aggregate_std("mean_raw_score", "raw_score_std"),
        "mean_abs_score": weighted_mean("mean_abs_score"),
        "abs_score_std": aggregate_std("mean_abs_score", "abs_score_std"),
        "mean_ply": weighted_mean("mean_ply"),
        "ply_std": aggregate_std("mean_ply", "ply_std"),
        "mean_piece_count": weighted_mean("mean_piece_count"),
        "piece_count_std": aggregate_std("mean_piece_count", "piece_count_std"),
        "mean_non_king_piece_count": weighted_mean("mean_non_king_piece_count"),
        "result_mean": weighted_mean("result_mean"),
        "result_wdl_mean": weighted_mean("result_wdl_mean"),
        "score_result_correlation": weighted_mean("score_result_correlation"),
        "abs_score_threshold_counts": {
            threshold: sum(int(stats["abs_score_threshold_counts"][threshold]) for stats in stats_list)
            for threshold in ["ge_1000", "ge_2000", "ge_4000", "ge_8000", "ge_16000"]
        },
    }
    if sampled:
        aggregate["file"]["sample_scan_bytes"] = sampled_scan_bytes
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
    aggregate["move_types"] = _aggregate_bucket_summaries(stats_list, "move_types", _MOVE_TYPE_LABELS, total_entries)
    aggregate["position_flags"] = _aggregate_position_flags(stats_list, total_entries)
    aggregate["material"] = _aggregate_material_summaries(stats_list, total_entries)

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
    aggregate["raw_score_percentiles"] = {
        percentile: _cp_to_raw_score(value)
        for percentile, value in aggregate["score_percentiles"].items()
    }
    aggregate["abs_raw_score_percentiles"] = {
        percentile: _cp_to_raw_score(value)
        for percentile, value in aggregate["abs_score_percentiles"].items()
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
    aggregate["wdl"] = _wdl_summary(aggregate)
    aggregate["aggregate_notes"] = [
        "entries, result counts, threshold counts, minima, maxima, and means are exact aggregates",
        "percentiles are conservative extrema of per-file percentiles, not exact merged percentiles",
    ]
    return aggregate
