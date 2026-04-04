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
        ("mean_abs_score", ctypes.c_double),
        ("mean_piece_count", ctypes.c_double),
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

    def to_torch(self, device: str):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError("PyTorch is required for training commands") from exc

        return {
            "white_indices": torch.from_numpy(self.white_indices).to(device=device, dtype=torch.long),
            "black_indices": torch.from_numpy(self.black_indices).to(device=device, dtype=torch.long),
            "white_counts": torch.from_numpy(self.white_counts).to(device=device, dtype=torch.int32),
            "black_counts": torch.from_numpy(self.black_counts).to(device=device, dtype=torch.int32),
            "stm": torch.from_numpy(self.stm).to(device=device, dtype=torch.float32).unsqueeze(1),
            "score_cp": torch.from_numpy(self.score_cp).to(device=device, dtype=torch.float32).unsqueeze(1),
            "result_wdl": torch.from_numpy(self.result_wdl).to(device=device, dtype=torch.float32).unsqueeze(1),
        }


def build_native_extension(force: bool = False) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = repo_root / "build" / "native"
    build_dir.mkdir(parents=True, exist_ok=True)

    existing = _find_library(build_dir)
    if existing is not None and not force:
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
    return {
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
        "mean_abs_score": float(stats.mean_abs_score),
        "mean_piece_count": float(stats.mean_piece_count),
    }


def write_fixture_binpack(path: str | Path) -> None:
    lib = _load_library()
    ok = lib.thrawn_write_fixture_binpack(os.fsencode(Path(path).resolve()))
    if ok != 1:
        raise NativeError(_last_error(lib))


class BinpackStream:
    def __init__(self, paths: list[str | Path], *, num_threads: int = 1, cyclic: bool = False):
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
    return lib


def _last_error(lib: ctypes.CDLL) -> str:
    raw = lib.thrawn_last_error()
    if not raw:
        return ""
    return raw.decode("utf-8")
