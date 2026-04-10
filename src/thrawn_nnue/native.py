from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ctypes
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
        ("white_counts", ctypes.POINTER(ctypes.c_int32)),
        ("black_counts", ctypes.POINTER(ctypes.c_int32)),
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
        ("min_score", ctypes.c_int16),
        ("max_score", ctypes.c_int16),
        ("min_ply", ctypes.c_uint16),
        ("max_ply", ctypes.c_uint16),
        ("mean_score", ctypes.c_double),
        ("mean_abs_score", ctypes.c_double),
        ("mean_piece_count", ctypes.c_double),
        ("score_p01", ctypes.c_double),
        ("score_p05", ctypes.c_double),
        ("score_p50", ctypes.c_double),
        ("score_p95", ctypes.c_double),
        ("score_p99", ctypes.c_double),
        ("abs_score_p50", ctypes.c_double),
        ("abs_score_p90", ctypes.c_double),
        ("abs_score_p95", ctypes.c_double),
        ("abs_score_p99", ctypes.c_double),
        ("ply_p50", ctypes.c_double),
        ("ply_p95", ctypes.c_double),
        ("abs_score_ge_1000", ctypes.c_uint64),
        ("abs_score_ge_2000", ctypes.c_uint64),
        ("abs_score_ge_4000", ctypes.c_uint64),
        ("abs_score_ge_8000", ctypes.c_uint64),
        ("abs_score_ge_16000", ctypes.c_uint64),
    ]


@dataclass(slots=True)
class NativeBatch:
    white_indices: np.ndarray
    black_indices: np.ndarray
    white_counts: np.ndarray
    black_counts: np.ndarray
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
    stats = _InspectStats()
    ok = lib.thrawn_inspect_binpack(os.fsencode(Path(path).resolve()), ctypes.byref(stats))
    if ok != 1:
        raise NativeError(_last_error(lib))
    entries_read = int(stats.entries_read)
    result = {
        "entries_read": int(stats.entries_read),
        "white_to_move": int(stats.white_to_move),
        "black_to_move": int(stats.black_to_move),
        "wins": int(stats.wins),
        "draws": int(stats.draws),
        "losses": int(stats.losses),
        "min_score": int(stats.min_score),
        "max_score": int(stats.max_score),
        "min_ply": int(stats.min_ply),
        "max_ply": int(stats.max_ply),
        "mean_score": float(stats.mean_score),
        "mean_abs_score": float(stats.mean_abs_score),
        "mean_piece_count": float(stats.mean_piece_count),
        "score_percentiles": {
            "p01": float(stats.score_p01),
            "p05": float(stats.score_p05),
            "p50": float(stats.score_p50),
            "p95": float(stats.score_p95),
            "p99": float(stats.score_p99),
        },
        "abs_score_percentiles": {
            "p50": float(stats.abs_score_p50),
            "p90": float(stats.abs_score_p90),
            "p95": float(stats.abs_score_p95),
            "p99": float(stats.abs_score_p99),
        },
        "ply_percentiles": {
            "p50": float(stats.ply_p50),
            "p95": float(stats.ply_p95),
        },
        "abs_score_threshold_counts": {
            "ge_1000": int(stats.abs_score_ge_1000),
            "ge_2000": int(stats.abs_score_ge_2000),
            "ge_4000": int(stats.abs_score_ge_4000),
            "ge_8000": int(stats.abs_score_ge_8000),
            "ge_16000": int(stats.abs_score_ge_16000),
        },
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
    result["wdl_scale_diagnostics"] = _wdl_scale_diagnostics(result)
    result["recommendation"] = _inspect_recommendation(result)
    return result


def inspect_binpack_collection(paths: list[str | Path]) -> dict[str, object]:
    resolved_paths = [Path(path).resolve() for path in paths]
    if not resolved_paths:
        raise ValueError("At least one .binpack path is required")

    per_file: list[dict[str, object]] = []
    for path in sorted(resolved_paths):
        stats = inspect_binpack(path)
        per_file.append({"path": str(path), "stats": stats})

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
    ):
        if not paths:
            raise ValueError("At least one dataset path is required")

        self._handle = None
        self._lib = _load_library()
        encoded_paths = [os.fsencode(Path(path).resolve()) for path in paths]
        self._path_array = (ctypes.c_char_p * len(encoded_paths))(*encoded_paths)
        self._handle = self._lib.thrawn_binpack_open_many(
            self._path_array,
            len(encoded_paths),
            num_threads,
            1 if cyclic else 0,
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
            white_counts = np.ctypeslib.as_array(view.white_counts, shape=(size,)).copy()
            black_counts = np.ctypeslib.as_array(view.black_counts, shape=(size,)).copy()
            stm = np.ctypeslib.as_array(view.stm, shape=(size,)).copy()
            score_cp = np.ctypeslib.as_array(view.score_cp, shape=(size,)).copy()
            result_wdl = np.ctypeslib.as_array(view.result_wdl, shape=(size,)).copy()
            return NativeBatch(
                white_indices=white_indices,
                black_indices=black_indices,
                white_counts=white_counts,
                black_counts=black_counts,
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


def _aggregate_inspection_stats(stats_list: list[dict[str, object]]) -> dict[str, object]:
    total_entries = sum(int(stats["entries_read"]) for stats in stats_list)
    if total_entries <= 0:
        raise ValueError("Inspection aggregate requires at least one dataset entry")

    def weighted_mean(key: str) -> float:
        numerator = sum(float(stats[key]) * int(stats["entries_read"]) for stats in stats_list)
        return numerator / total_entries

    aggregate: dict[str, object] = {
        "entries_read": total_entries,
        "white_to_move": sum(int(stats["white_to_move"]) for stats in stats_list),
        "black_to_move": sum(int(stats["black_to_move"]) for stats in stats_list),
        "wins": sum(int(stats["wins"]) for stats in stats_list),
        "draws": sum(int(stats["draws"]) for stats in stats_list),
        "losses": sum(int(stats["losses"]) for stats in stats_list),
        "min_score": min(int(stats["min_score"]) for stats in stats_list),
        "max_score": max(int(stats["max_score"]) for stats in stats_list),
        "min_ply": min(int(stats["min_ply"]) for stats in stats_list),
        "max_ply": max(int(stats["max_ply"]) for stats in stats_list),
        "mean_score": weighted_mean("mean_score"),
        "mean_abs_score": weighted_mean("mean_abs_score"),
        "mean_piece_count": weighted_mean("mean_piece_count"),
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

    # Exact percentiles cannot be reconstructed from per-file summaries alone, so
    # aggregate percentiles are conservative maxima of the file-level estimates.
    aggregate["score_percentiles"] = {
        percentile: max(float(stats["score_percentiles"][percentile]) for stats in stats_list)
        for percentile in ["p01", "p05", "p50", "p95", "p99"]
    }
    aggregate["abs_score_percentiles"] = {
        percentile: max(float(stats["abs_score_percentiles"][percentile]) for stats in stats_list)
        for percentile in ["p50", "p90", "p95", "p99"]
    }
    aggregate["ply_percentiles"] = {
        percentile: max(float(stats["ply_percentiles"][percentile]) for stats in stats_list)
        for percentile in ["p50", "p95"]
    }
    aggregate["wdl_scale_diagnostics"] = _wdl_scale_diagnostics(aggregate)
    aggregate["recommendation"] = _inspect_recommendation(aggregate)
    aggregate["aggregate_notes"] = [
        "entries, result counts, threshold counts, minima, maxima, and means are exact aggregates",
        "percentiles are conservative maxima of per-file percentiles, not exact merged percentiles",
    ]
    return aggregate


def _wdl_scale_diagnostics(stats: dict[str, object]) -> dict[str, dict[str, float]]:
    mean_abs_score = float(stats["mean_abs_score"])
    candidate_scales = [410.0, 1000.0, 2000.0, 4000.0, 8000.0]
    diagnostics: dict[str, dict[str, float]] = {}
    for scale in candidate_scales:
        transformed = _wdl_target(mean_abs_score, scale)
        p95_transformed = _wdl_target(float(stats["abs_score_percentiles"]["p95"]), scale)
        diagnostics[str(int(scale))] = {
            "mean_abs_score_target": float(transformed),
            "p95_abs_score_target": float(p95_transformed),
            "high_saturation_proxy": float(transformed >= 0.98 or p95_transformed >= 0.995),
        }
    return diagnostics


def _wdl_target(abs_score: float, effective_raw_wdl_scale: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(abs_score / effective_raw_wdl_scale))))


def _recommend_wdl_scale(mean_abs_score: float, abs_p95: float, score_scale: float) -> float:
    normalized_mean_abs_score = mean_abs_score / score_scale
    normalized_abs_p95 = abs_p95 / score_scale
    for candidate in [410.0, 1000.0, 2000.0, 4000.0, 8000.0]:
        mean_target = _wdl_target(normalized_mean_abs_score, candidate)
        p95_target = _wdl_target(normalized_abs_p95, candidate)
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
            "recommended_score_scale": 1.0,
            "effective_raw_wdl_scale": 410.0,
            "effective_mean_abs_score_target": 0.5,
            "effective_p95_abs_score_target": 0.5,
            "teacher_target_collapse_risk": False,
            "raw_space_pair_collapse_risk": False,
            "notes": ["dataset is empty or unreadable; verify the binpack path before using this recommendation"],
        }

    recommended_score_clip = 0.0
    if abs_p99 >= 16000:
        recommended_score_clip = 16000.0
    elif abs_p99 >= 8000:
        recommended_score_clip = 8000.0

    recommended_score_scale = 1.0
    if mean_abs_score >= 6000:
        recommended_score_scale = 10.0
    elif mean_abs_score >= 2000:
        recommended_score_scale = 4.0

    raw_space_recommended_wdl_scale = _recommend_wdl_scale(mean_abs_score, abs_p95, 1.0)
    recommended_wdl_scale = _recommend_wdl_scale(mean_abs_score, abs_p95, recommended_score_scale)
    effective_raw_wdl_scale = recommended_score_scale * recommended_wdl_scale
    effective_mean_abs_score_target = _wdl_target(mean_abs_score, effective_raw_wdl_scale)
    effective_p95_abs_score_target = _wdl_target(abs_p95, effective_raw_wdl_scale)
    teacher_target_collapse_risk = _teacher_target_collapse_risk(
        mean_abs_score,
        abs_p95,
        effective_raw_wdl_scale,
    )
    raw_space_effective_raw_wdl_scale = recommended_score_scale * raw_space_recommended_wdl_scale
    raw_space_pair_collapse_risk = _teacher_target_collapse_risk(
        mean_abs_score,
        abs_p95,
        raw_space_effective_raw_wdl_scale,
    )

    saturated_at_410 = bool(diagnostics["410"]["high_saturation_proxy"] >= 0.5)
    notes = []
    if saturated_at_410:
        notes.append("dataset appears highly saturated for wdl_scale=410")
    if recommended_score_clip > 0.0:
        notes.append("extreme score tails suggest enabling score clipping")
    if recommended_score_scale != 1.0:
        notes.append("score magnitudes look unusually large for direct centipawn-style use")
    if raw_space_pair_collapse_risk:
        notes.append("combining score_scale with a raw-space wdl_scale would collapse teacher targets")
    if teacher_target_collapse_risk:
        notes.append("recommended score_scale and wdl_scale still keep teacher targets too close to 0.5")
    if not notes:
        notes.append("current score distribution looks usable without aggressive normalization")

    return {
        "saturated_at_default_wdl_scale": saturated_at_410,
        "recommended_wdl_scale": recommended_wdl_scale,
        "recommended_score_clip": recommended_score_clip,
        "recommended_score_scale": recommended_score_scale,
        "effective_raw_wdl_scale": effective_raw_wdl_scale,
        "effective_mean_abs_score_target": effective_mean_abs_score_target,
        "effective_p95_abs_score_target": effective_p95_abs_score_target,
        "teacher_target_collapse_risk": teacher_target_collapse_risk,
        "raw_space_pair_collapse_risk": raw_space_pair_collapse_risk,
        "notes": notes,
    }
