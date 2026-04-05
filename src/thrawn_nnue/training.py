from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

from .checkpoint import load_checkpoint, restore_rng_state, save_checkpoint
from .console import ConsoleContext, create_console_reporter
from .config import TrainConfig
from .native import BinpackStream, inspect_binpack


def _require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch is required for training commands") from exc
    return torch


@dataclass(slots=True)
class TrainState:
    model: object
    optimizer: object
    scheduler: object
    scaler: object
    config: TrainConfig
    device: str
    run_dir: Path
    metrics_path: Path
    best_validation_loss: float | None = None
    best_validation_step: int | None = None
    epoch: int = 0
    step_in_epoch: int = 0
    global_step: int = 0


def train_from_config(config: TrainConfig, *, console_mode: str | None = None) -> Path:
    if not config.train_datasets:
        raise ValueError("train_datasets must not be empty")

    if console_mode is not None:
        config.console_mode = console_mode
        config.validate()
    state = _create_state(config)
    _run_training_loop(state)
    final_checkpoint = state.run_dir / "checkpoints" / f"step_{state.global_step:08d}.pt"
    _save_training_checkpoint(state, final_checkpoint)
    return final_checkpoint


def resume_training(checkpoint_path: str | Path, *, console_mode: str | None = None) -> Path:
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    config = TrainConfig.from_dict(dict(payload["config"]))
    if console_mode is not None:
        config.console_mode = console_mode
        config.validate()
    state = _create_state(config)
    state.model.load_state_dict(payload["model_state"])
    state.optimizer.load_state_dict(payload["optimizer_state"])
    if state.scheduler is not None and payload["scheduler_state"] is not None:
        state.scheduler.load_state_dict(payload["scheduler_state"])
    if state.scaler is not None and payload["scaler_state"] is not None:
        state.scaler.load_state_dict(payload["scaler_state"])
    restore_rng_state(payload["rng_state"])
    state.best_validation_loss = payload.get("best_validation_loss")
    state.best_validation_step = payload.get("best_validation_step")
    state.epoch = int(payload["epoch"])
    state.step_in_epoch = int(payload["step_in_epoch"])
    state.global_step = int(payload["global_step"])
    state.model.to(state.device)
    _run_training_loop(state)
    final_checkpoint = state.run_dir / "checkpoints" / f"step_{state.global_step:08d}.pt"
    _save_training_checkpoint(state, final_checkpoint)
    return final_checkpoint


def _create_state(config: TrainConfig) -> TrainState:
    torch = _require_torch()
    from .model import DualPerspectiveA768NNUE

    _resolve_runtime_config(config)
    device = _select_device(config.device, torch)
    run_dir = Path(config.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    model = DualPerspectiveA768NNUE(
        num_features=config.num_features,
        ft_size=config.ft_size,
        hidden_size=config.hidden_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.max_epochs * config.steps_per_epoch,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=_cuda_amp_enabled(config, device))

    return TrainState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        config=config,
        device=device,
        run_dir=run_dir,
        metrics_path=metrics_path,
    )


def _run_training_loop(state: TrainState) -> None:
    torch = _require_torch()
    autocast_enabled = _cuda_amp_enabled(state.config, state.device)
    reporter = create_console_reporter(state.config.console_mode)
    reporter.startup(
        ConsoleContext(
            run_name=state.config.run_name,
            device=state.device,
            train_shards=len(state.config.train_datasets),
            validation_shards=len(state.config.validation_datasets),
            total_steps=state.config.max_epochs * state.config.steps_per_epoch,
            initial_global_step=state.global_step,
            max_epochs=state.config.max_epochs,
            steps_per_epoch=state.config.steps_per_epoch,
            log_every=state.config.log_every,
        )
    )

    try:
        with BinpackStream(
            state.config.train_datasets,
            num_threads=state.config.num_loader_threads,
            cyclic=True,
        ) as train_stream:
            for epoch in range(state.epoch, state.config.max_epochs):
                start_step = state.step_in_epoch if epoch == state.epoch else 0
                for step_in_epoch in range(start_step, state.config.steps_per_epoch):
                    batch = train_stream.next_batch(state.config.batch_size)
                    if batch is None:
                        raise RuntimeError("Cyclic training stream unexpectedly returned EOF")

                    tensors = batch.to_torch(state.device)
                    state.model.train()
                    state.optimizer.zero_grad(set_to_none=True)
                    with torch.cuda.amp.autocast(enabled=autocast_enabled):
                        prediction_cp = state.model(
                            tensors["white_indices"],
                            tensors["black_indices"],
                            tensors["stm"],
                        )
                        normalized_scores = _normalize_teacher_scores(tensors["score_cp"], state.config, torch)
                        loss, eval_loss, result_loss = _blended_loss(
                            prediction_cp,
                            normalized_scores,
                            tensors["result_wdl"],
                            state.config.wdl_scale,
                            state.config.result_lambda,
                            torch,
                        )

                    state.scaler.scale(loss).backward()
                    state.scaler.unscale_(state.optimizer)
                    torch.nn.utils.clip_grad_norm_(state.model.parameters(), state.config.clip_grad_norm)
                    state.scaler.step(state.optimizer)
                    state.scaler.update()
                    state.scheduler.step()

                    state.global_step += 1
                    state.step_in_epoch = step_in_epoch + 1
                    current_loss = float(loss.detach().cpu().item())
                    current_lr = float(state.optimizer.param_groups[0]["lr"])
                    reporter.update_train(
                        epoch=epoch,
                        step_in_epoch=state.step_in_epoch,
                        global_step=state.global_step,
                        loss=current_loss,
                        lr=current_lr,
                    )

                    if state.global_step % state.config.log_every == 0:
                        _log_metrics(
                            state,
                            {
                                "event": "train",
                                "epoch": epoch,
                                "step_in_epoch": step_in_epoch,
                                "global_step": state.global_step,
                                "loss": current_loss,
                                "eval_loss": float(eval_loss.detach().cpu().item()),
                                "result_loss": float(result_loss.detach().cpu().item()),
                                "lr": current_lr,
                            },
                        )

                    if state.global_step % state.config.checkpoint_every == 0:
                        checkpoint_path = state.run_dir / "checkpoints" / f"step_{state.global_step:08d}.pt"
                        _save_training_checkpoint(state, checkpoint_path, epoch=epoch, step_in_epoch=state.step_in_epoch)
                        reporter.checkpoint_saved(str(checkpoint_path), is_best=False)

                    if _should_run_validation(state):
                        reporter.validation_started(global_step=state.global_step)
                        metrics = _run_validation(state)
                        is_best = _maybe_update_best_checkpoint(state, float(metrics["validation_loss"]))
                        metrics["is_best"] = is_best
                        _log_metrics(state, metrics)
                        reporter.validation_finished(metrics, is_best=is_best)
                        if is_best:
                            reporter.checkpoint_saved(
                                str(state.run_dir / "checkpoints" / "best.pt"),
                                is_best=True,
                            )

                state.epoch = epoch + 1
                state.step_in_epoch = 0
    finally:
        reporter.close()


def _save_training_checkpoint(
    state: TrainState,
    checkpoint_path: Path,
    *,
    epoch: int | None = None,
    step_in_epoch: int | None = None,
) -> None:
    save_checkpoint(
        checkpoint_path,
        model=state.model,
        optimizer=state.optimizer,
        scheduler=state.scheduler,
        scaler=state.scaler,
        config=state.config.to_dict(),
        epoch=state.epoch if epoch is None else epoch,
        step_in_epoch=state.step_in_epoch if step_in_epoch is None else step_in_epoch,
        global_step=state.global_step,
        best_validation_loss=state.best_validation_loss,
        best_validation_step=state.best_validation_step,
    )


def _resolve_runtime_config(config: TrainConfig) -> None:
    if config.steps_per_epoch == 0:
        config.steps_per_epoch = _auto_steps_for_dataset(config.train_datasets, config.batch_size)
    if config.validation_datasets and config.validation_steps == 0:
        config.validation_steps = _auto_steps_for_dataset(config.validation_datasets, config.batch_size)
    config.validate()


def _auto_steps_for_dataset(paths: list[str], batch_size: int) -> int:
    total_entries = 0
    for path in paths:
        stats = inspect_binpack(path)
        total_entries += int(stats["entries_read"])
    if total_entries <= 0:
        raise ValueError("Auto-sized datasets must contain at least one entry")
    return max(1, math.ceil(total_entries / batch_size))


def _should_run_validation(state: TrainState) -> bool:
    if not state.config.validation_datasets:
        return False
    if state.config.validation_every == 0:
        return state.step_in_epoch >= state.config.steps_per_epoch
    return state.global_step % state.config.validation_every == 0


def _blended_loss(prediction_cp, target_cp, result_wdl, wdl_scale: float, result_lambda: float, torch):
    pred_wdl = torch.sigmoid(prediction_cp / wdl_scale)
    target_wdl = torch.sigmoid(target_cp / wdl_scale)
    eval_loss = torch.mean((pred_wdl - target_wdl) ** 2)
    result_loss = torch.mean((pred_wdl - result_wdl) ** 2)
    loss = result_lambda * eval_loss + (1.0 - result_lambda) * result_loss
    return loss, eval_loss, result_loss


def _normalize_teacher_scores(scores, config: TrainConfig, torch):
    normalized = scores
    if config.score_clip > 0.0:
        normalized = torch.clamp(normalized, -config.score_clip, config.score_clip)
    if config.score_scale != 1.0:
        normalized = normalized / config.score_scale
    return normalized


def _run_validation(state: TrainState) -> dict[str, object]:
    torch = _require_torch()
    autocast_enabled = _cuda_amp_enabled(state.config, state.device)

    total_loss = 0.0
    total_eval_loss = 0.0
    total_result_loss = 0.0
    batches = 0

    was_training = state.model.training
    state.model.eval()
    with BinpackStream(
        state.config.validation_datasets,
        num_threads=state.config.num_loader_threads,
        cyclic=False,
    ) as validation_stream:
        with torch.no_grad():
            for _ in range(state.config.validation_steps):
                batch = validation_stream.next_batch(state.config.batch_size)
                if batch is None:
                    break

                tensors = batch.to_torch(state.device)
                with torch.cuda.amp.autocast(enabled=autocast_enabled):
                    prediction_cp = state.model(
                        tensors["white_indices"],
                        tensors["black_indices"],
                        tensors["stm"],
                    )
                    normalized_scores = _normalize_teacher_scores(tensors["score_cp"], state.config, torch)
                    loss, eval_loss, result_loss = _blended_loss(
                        prediction_cp,
                        normalized_scores,
                        tensors["result_wdl"],
                        state.config.wdl_scale,
                        state.config.result_lambda,
                        torch,
                    )

                total_loss += float(loss.detach().cpu().item())
                total_eval_loss += float(eval_loss.detach().cpu().item())
                total_result_loss += float(result_loss.detach().cpu().item())
                batches += 1

    if was_training:
        state.model.train()

    average_loss = math.inf if batches == 0 else total_loss / batches
    average_eval_loss = math.inf if batches == 0 else total_eval_loss / batches
    average_result_loss = math.inf if batches == 0 else total_result_loss / batches

    return {
        "event": "validation",
        "global_step": state.global_step,
        "validation_loss": average_loss,
        "validation_eval_loss": average_eval_loss,
        "validation_result_loss": average_result_loss,
        "validation_batches": batches,
    }


def _maybe_update_best_checkpoint(state: TrainState, validation_loss: float) -> bool:
    if not math.isfinite(validation_loss):
        return False

    if state.best_validation_loss is not None and validation_loss >= state.best_validation_loss:
        return False

    state.best_validation_loss = validation_loss
    state.best_validation_step = state.global_step
    best_checkpoint_path = state.run_dir / "checkpoints" / "best.pt"
    _save_training_checkpoint(state, best_checkpoint_path)
    return True


def _log_metrics(state: TrainState, metrics: dict[str, object]) -> None:
    line = json.dumps(metrics)
    with state.metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def _select_device(requested: str, torch) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available(torch):
            return "mps"
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested, but CUDA is not available in this PyTorch build")
        return "cuda"
    if requested == "mps":
        if not _mps_available(torch):
            raise RuntimeError("device='mps' was requested, but Apple Metal (MPS) is not available in this PyTorch build")
        return "mps"
    if requested == "cpu":
        return "cpu"
    return requested


def _mps_available(torch) -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def _cuda_amp_enabled(config: TrainConfig, device: str) -> bool:
    return config.amp and device == "cuda"
