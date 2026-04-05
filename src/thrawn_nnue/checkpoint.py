from __future__ import annotations

from pathlib import Path
import random

import numpy as np


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for checkpoint commands") from exc
    return torch


def capture_rng_state() -> dict[str, object]:
    torch = _require_torch()
    state: dict[str, object] = {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, object]) -> None:
    torch = _require_torch()
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_checkpoint(
    path: str | Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    config: dict[str, object],
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    best_validation_loss: float | None = None,
    best_validation_step: int | None = None,
) -> Path:
    torch = _require_torch()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "rng_state": capture_rng_state(),
        "best_validation_loss": best_validation_loss,
        "best_validation_step": best_validation_step,
    }
    torch.save(payload, output_path)
    return output_path


def load_checkpoint(path: str | Path, *, map_location: str = "cpu") -> dict[str, object]:
    torch = _require_torch()
    # Checkpoints store optimizer state and RNG snapshots, so they must be loaded
    # as full objects rather than with the newer weights-only default.
    return torch.load(Path(path), map_location=map_location, weights_only=False)
