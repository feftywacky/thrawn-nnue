from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import json
import math
from queue import Full, Queue
from threading import Event, Thread
import time

from .checkpoint import load_checkpoint, restore_rng_state, save_checkpoint
from .console import ConsoleContext, create_console_reporter
from .config import TrainConfig
from .export import MATERIAL_SANITY_POSITIONS
from .native import BinpackStream

SANITY_ANCHOR_POSITIONS = [
    ("starting_position_white", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"),
    ("starting_position_black", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b - - 0 1"),
    ("bare_kings_white", "8/2k5/8/8/8/8/5K2/8 w - - 0 1"),
    ("bare_kings_black", "8/2k5/8/8/8/8/5K2/8 b - - 0 1"),
]


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
    best_validation_positions: int | None = None
    global_step: int = 0
    positions_seen: int = 0
    epoch_index: int = 0


@dataclass(slots=True)
class _PreparedBatch:
    batch_positions: int
    tensors: dict[str, object]


@dataclass(slots=True)
class _ProducerException:
    exception: BaseException


_PREFETCH_EOF = object()


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
    state.best_validation_positions = _payload_best_validation_positions(payload)
    state.global_step = int(payload["global_step"])
    state.positions_seen = _payload_positions_seen(payload)
    state.epoch_index = _payload_epoch_index(payload)
    state.model.to(state.device)
    _run_training_loop(state)
    final_checkpoint = state.run_dir / "checkpoints" / f"step_{state.global_step:08d}.pt"
    _save_training_checkpoint(state, final_checkpoint)
    return final_checkpoint


def _create_state(config: TrainConfig) -> TrainState:
    torch = _require_torch()
    from .model import HalfKPNNUE

    _resolve_runtime_config(config)
    device = _select_device(config.device, torch)
    run_dir = Path(config.output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    model = HalfKPNNUE(
        num_features=config.num_features,
        num_factor_features=config.num_factor_features,
        ft_size=config.ft_size,
        l1_size=config.l1_size,
        l2_size=config.l2_size,
    ).to(device)
    optimizer = _create_optimizer(config, model, torch)
    scheduler = _create_scheduler(config, optimizer, torch)
    scaler = _create_grad_scaler(torch, config, device)

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


class _PreparedBatchSource:
    def __init__(
        self,
        stream,
        *,
        batch_size: int,
        total_positions: int | None,
        prefetch_batches: int,
        torch,
    ) -> None:
        self._stream = stream
        self._batch_size = batch_size
        self._remaining_positions = total_positions
        self._prefetch_batches = prefetch_batches
        self._torch = torch
        self._queue: Queue[object] | None = None
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._closed = False

    def __enter__(self) -> "_PreparedBatchSource":
        if self._prefetch_batches > 0:
            self._queue = Queue(maxsize=self._prefetch_batches)
            self._thread = Thread(target=self._run_producer, name="thrawn-batch-prefetch", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __iter__(self) -> "_PreparedBatchSource":
        return self

    def __next__(self) -> _PreparedBatch:
        if self._prefetch_batches <= 0:
            return self._next_sync()
        return self._next_prefetched()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.5)

    def _next_sync(self) -> _PreparedBatch:
        requested_batch_size = _requested_batch_size(self._batch_size, self._remaining_positions)
        if requested_batch_size is None:
            raise StopIteration
        batch = self._stream.next_batch(requested_batch_size)
        if batch is None:
            raise StopIteration
        prepared = _prepare_batch(batch, self._torch)
        self._remaining_positions = _consume_positions(self._remaining_positions, prepared.batch_positions)
        return prepared

    def _next_prefetched(self) -> _PreparedBatch:
        if self._queue is None:
            raise StopIteration
        item = self._queue.get()
        if item is _PREFETCH_EOF:
            raise StopIteration
        if isinstance(item, _ProducerException):
            self.close()
            raise item.exception
        return item

    def _run_producer(self) -> None:
        try:
            while not self._stop_event.is_set():
                requested_batch_size = _requested_batch_size(self._batch_size, self._remaining_positions)
                if requested_batch_size is None:
                    break
                batch = self._stream.next_batch(requested_batch_size)
                if batch is None:
                    break
                prepared = _prepare_batch(batch, self._torch)
                self._remaining_positions = _consume_positions(self._remaining_positions, prepared.batch_positions)
                if not self._queue_put(prepared):
                    return
        except BaseException as exc:
            self._queue_put(_ProducerException(exc))
            return
        self._queue_put(_PREFETCH_EOF)

    def _queue_put(self, item: object) -> bool:
        if self._queue is None:
            return False
        while not self._stop_event.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False


def _run_training_loop(state: TrainState) -> None:
    torch = _require_torch()
    autocast_enabled = _amp_enabled(torch, state.config, state.device)
    reporter = create_console_reporter(state.config.console_mode)
    validation_interval_positions = _effective_validation_interval_positions(state.config)
    next_validation_positions = (
        None
        if not state.config.validation_datasets
        else _next_validation_positions(state.positions_seen, validation_interval_positions)
    )
    last_validation_positions: int | None = None

    reporter.startup(
        ConsoleContext(
            run_name=state.config.run_name,
            device=state.device,
            train_shards=len(state.config.train_datasets),
            validation_shards=len(state.config.validation_datasets),
            total_train_positions=state.config.total_train_positions,
            initial_positions_seen=state.positions_seen,
            batch_size=state.config.batch_size,
            epoch_positions=state.config.epoch_positions,
            validation_interval_positions=validation_interval_positions,
            log_every=state.config.log_every,
            prefetch_batches=state.config.prefetch_batches,
        )
    )

    try:
        with BinpackStream(
            state.config.train_datasets,
            num_threads=state.config.num_loader_threads,
            cyclic=True,
            **_binpack_filter_options(state.config),
        ) as train_stream:
            remaining_positions = state.config.total_train_positions - state.positions_seen
            with _PreparedBatchSource(
                train_stream,
                batch_size=state.config.batch_size,
                total_positions=remaining_positions,
                prefetch_batches=state.config.prefetch_batches,
                torch=torch,
            ) as train_batches:
                while state.positions_seen < state.config.total_train_positions:
                    step_started_at = time.monotonic()
                    try:
                        prepared_batch = next(train_batches)
                    except StopIteration:
                        raise RuntimeError("Cyclic training stream unexpectedly returned EOF") from None

                    batch_positions = prepared_batch.batch_positions
                    positions_before_step = state.positions_seen
                    losses = _run_train_step(
                        state,
                        prepared_batch.tensors,
                        torch,
                        autocast_enabled=autocast_enabled,
                    )
                    state.global_step += 1
                    state.positions_seen += batch_positions
                    state.epoch_index = _epoch_index(state.positions_seen, state.config.epoch_positions)
                    _advance_scheduler_for_epoch_boundaries(
                        state,
                        positions_before_step=positions_before_step,
                        positions_after_step=state.positions_seen,
                    )
                    current_loss = float(losses["loss"].detach().cpu().item())
                    current_lr = float(state.optimizer.param_groups[0]["lr"])
                    step_seconds = max(0.0, time.monotonic() - step_started_at)
                    train_positions_per_second = (
                        0.0 if step_seconds <= 0.0 else float(batch_positions) / step_seconds
                    )
                    reporter.update_train(
                        global_step=state.global_step,
                        positions_seen=state.positions_seen,
                        epoch_index=state.epoch_index,
                        loss=current_loss,
                        lr=current_lr,
                        step_seconds=step_seconds,
                        train_positions_per_second=train_positions_per_second,
                    )

                    if state.global_step % state.config.log_every == 0:
                        _log_metrics(
                            state,
                            {
                                "event": "train",
                                "global_step": state.global_step,
                                "positions_seen": state.positions_seen,
                                "epoch_index": state.epoch_index,
                                "batch_positions": batch_positions,
                                "loss": current_loss,
                                "cp_loss": float(losses["cp_loss"].detach().cpu().item()),
                                "wdl_loss": float(losses["wdl_loss"].detach().cpu().item()),
                                "output_reg_loss": float(losses["output_reg_loss"].detach().cpu().item()),
                                "sanity_anchor_loss": float(
                                    losses["sanity_anchor_loss"].detach().cpu().item()
                                ),
                                "wdl_lambda": state.config.wdl_lambda,
                                "lr": current_lr,
                                "step_seconds": step_seconds,
                                "train_positions_per_second": train_positions_per_second,
                            },
                        )

                    if next_validation_positions is not None and state.positions_seen >= next_validation_positions:
                        _run_validation_and_report(state, reporter)
                        last_validation_positions = state.positions_seen
                        next_validation_positions = _next_validation_positions(
                            state.positions_seen,
                            validation_interval_positions,
                        )

                    if state.global_step % state.config.checkpoint_every == 0:
                        checkpoint_path = state.run_dir / "checkpoints" / f"step_{state.global_step:08d}.pt"
                        _save_training_checkpoint(state, checkpoint_path)
                        reporter.checkpoint_saved(str(checkpoint_path), is_best=False)

        if state.config.validation_datasets and last_validation_positions != state.positions_seen:
            _run_validation_and_report(state, reporter)
    finally:
        reporter.close()


def _save_training_checkpoint(state: TrainState, checkpoint_path: Path) -> None:
    save_checkpoint(
        checkpoint_path,
        model=state.model,
        optimizer=state.optimizer,
        scheduler=state.scheduler,
        scaler=state.scaler,
        config=state.config.to_dict(),
        global_step=state.global_step,
        positions_seen=state.positions_seen,
        epoch_index=state.epoch_index,
        best_validation_loss=state.best_validation_loss,
        best_validation_positions=state.best_validation_positions,
    )


def _resolve_runtime_config(config: TrainConfig) -> None:
    config.validate()


def _create_scheduler(config: TrainConfig, optimizer, torch):
    total_epochs = max(1, config.total_train_positions // config.epoch_positions)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=config.learning_rate * 0.01,
    )


def _create_optimizer(config: TrainConfig, model, torch):
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _run_train_step(state: TrainState, tensors, torch, *, autocast_enabled: bool):
    tensors = _move_tensors_to_device(tensors, state.device)
    score_cp_stm, result_wdl_stm = _stm_oriented_targets(
        tensors["score_cp"],
        tensors["result_wdl"],
        tensors["stm"],
        torch,
    )
    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)
    with _autocast_context(torch, state.device, autocast_enabled):
        normalized_scores = _normalize_teacher_scores(score_cp_stm, state.config, torch)
        prediction_cp = state.model(
            tensors["white_indices"],
            tensors["black_indices"],
            tensors["stm"],
        )
        losses = _scalar_head_loss(
            prediction_cp,
            normalized_scores,
            result_wdl_stm,
            wdl_eval_weight=state.config.wdl_lambda,
            wdl_in_offset=state.config.wdl_in_offset,
            wdl_out_offset=state.config.wdl_out_offset,
            wdl_in_scaling=state.config.wdl_in_scaling,
            wdl_out_scaling=state.config.wdl_out_scaling,
            wdl_loss_power=state.config.wdl_loss_power,
            output_regularization=state.config.output_regularization,
            torch=torch,
        )
        losses = _add_sanity_anchor_loss(losses, state.model, state.config, torch, state.device)

    state.scaler.scale(losses["loss"]).backward()
    state.scaler.unscale_(state.optimizer)
    torch.nn.utils.clip_grad_norm_(state.model.parameters(), state.config.clip_grad_norm)
    state.scaler.step(state.optimizer)
    _clip_model_weights(state.model, state.config)
    state.scaler.update()
    return losses


def _run_validation_and_report(state: TrainState, reporter) -> None:
    reporter.validation_started(global_step=state.global_step, positions_seen=state.positions_seen)
    metrics = _run_validation(state)
    is_best = _maybe_update_best_checkpoint(state, float(metrics["validation_loss"]))
    metrics["is_best"] = is_best
    _log_metrics(state, metrics)
    reporter.validation_finished(metrics, is_best=is_best)
    if is_best:
        reporter.checkpoint_saved(
            str(_best_step_checkpoint_path(state.run_dir, state.global_step)),
            is_best=True,
        )


def _scalar_head_loss(
    prediction_cp,
    target_cp,
    result_wdl,
    *,
    wdl_eval_weight: float,
    wdl_in_offset: float,
    wdl_out_offset: float,
    wdl_in_scaling: float,
    wdl_out_scaling: float,
    wdl_loss_power: float,
    output_regularization: float,
    torch,
):
    pred_wdl = _wdl_expectation_from_cp(prediction_cp, wdl_in_offset, wdl_in_scaling, torch)
    target_wdl = _wdl_expectation_from_cp(target_cp, wdl_out_offset, wdl_out_scaling, torch)
    blended_target = wdl_eval_weight * target_wdl + (1.0 - wdl_eval_weight) * result_wdl
    cp_loss = torch.mean((pred_wdl - blended_target).abs().pow(wdl_loss_power))
    wdl_loss = torch.mean((pred_wdl - result_wdl).abs().pow(wdl_loss_power))
    output_reg_loss = torch.mean(prediction_cp.square())
    loss = cp_loss + output_regularization * output_reg_loss
    return {
        "loss": loss,
        "cp_loss": cp_loss,
        "wdl_loss": wdl_loss,
        "output_reg_loss": output_reg_loss,
        "sanity_anchor_loss": loss.new_zeros(()),
        "predicted_cp": prediction_cp,
        "predicted_wdl": pred_wdl,
    }


def _add_sanity_anchor_loss(losses, model, config: TrainConfig, torch, device: str):
    if config.sanity_anchor_weight <= 0.0:
        return losses

    losses = dict(losses)
    anchor_loss = _sanity_anchor_loss(model, config, torch, device)
    losses["sanity_anchor_loss"] = anchor_loss
    losses["loss"] = losses["loss"] + config.sanity_anchor_weight * anchor_loss
    return losses


def _sanity_anchor_loss(model, config: TrainConfig, torch, device: str):
    tensors = _sanity_anchor_tensors(config, torch, device)
    prediction_cp = model(
        tensors["white_indices"],
        tensors["black_indices"],
        tensors["stm"],
    )
    return torch.mean((prediction_cp / config.wdl_out_scaling).square())


def _clip_model_weights(model, config: TrainConfig) -> None:
    dense_limit = (127.0 - 0.5) / max(config.export_dense_scale, 1.0)
    torch = _require_torch()
    with torch.no_grad():
        model.l1.weight.clamp_(-dense_limit, dense_limit)
        model.l2.weight.clamp_(-dense_limit, dense_limit)


def _normalize_teacher_scores(scores, config: TrainConfig, torch):
    normalized = scores
    if config.score_clip > 0.0:
        normalized = torch.clamp(normalized, -config.score_clip, config.score_clip)
    return normalized


def _stm_oriented_targets(score_cp, result_wdl, stm, torch):
    # Binpack teacher scores and results are already side-to-move oriented.
    return score_cp, result_wdl


def _run_validation(state: TrainState) -> dict[str, object]:
    torch = _require_torch()
    autocast_enabled = _amp_enabled(torch, state.config, state.device)
    started_at = time.monotonic()

    total_loss = 0.0
    total_cp_loss = 0.0
    total_wdl_loss = 0.0
    total_output_reg_loss = 0.0
    total_sanity_anchor_loss = 0.0
    total_positions = 0
    total_correct = 0
    total_teacher_result_disagreement = 0
    total_abs_error = 0.0
    total_sq_error = 0.0
    corr_sum_x = 0.0
    corr_sum_y = 0.0
    corr_sum_x2 = 0.0
    corr_sum_y2 = 0.0
    corr_sum_xy = 0.0
    batches = 0

    was_training = state.model.training
    state.model.eval()
    with BinpackStream(
        state.config.validation_datasets,
        num_threads=state.config.num_loader_threads,
        cyclic=False,
        **_binpack_filter_options(state.config),
    ) as validation_stream:
        validation_budget = state.config.validation_positions if state.config.validation_positions > 0 else None
        with _PreparedBatchSource(
            validation_stream,
            batch_size=state.config.batch_size,
            total_positions=validation_budget,
            prefetch_batches=state.config.prefetch_batches,
            torch=torch,
        ) as validation_batches:
            with torch.no_grad():
                for prepared_batch in validation_batches:
                    batch_positions = prepared_batch.batch_positions
                    tensors = _move_tensors_to_device(prepared_batch.tensors, state.device)
                    score_cp_stm, result_wdl_stm = _stm_oriented_targets(
                        tensors["score_cp"],
                        tensors["result_wdl"],
                        tensors["stm"],
                        torch,
                    )
                    with _autocast_context(torch, state.device, autocast_enabled):
                        normalized_scores = _normalize_teacher_scores(score_cp_stm, state.config, torch)
                        prediction_cp = state.model(
                            tensors["white_indices"],
                            tensors["black_indices"],
                            tensors["stm"],
                        )
                        losses = _scalar_head_loss(
                            prediction_cp,
                            normalized_scores,
                            result_wdl_stm,
                            wdl_eval_weight=state.config.wdl_lambda,
                            wdl_in_offset=state.config.wdl_in_offset,
                            wdl_out_offset=state.config.wdl_out_offset,
                            wdl_in_scaling=state.config.wdl_in_scaling,
                            wdl_out_scaling=state.config.wdl_out_scaling,
                            wdl_loss_power=state.config.wdl_loss_power,
                            output_regularization=state.config.output_regularization,
                            torch=torch,
                        )
                        losses = _add_sanity_anchor_loss(losses, state.model, state.config, torch, state.device)

                    pred_wdl = losses["predicted_wdl"]
                    target_wdl = _wdl_expectation_from_cp(
                        normalized_scores,
                        state.config.wdl_out_offset,
                        state.config.wdl_out_scaling,
                        torch,
                    )
                    result_bucket = _wdl_bucket(result_wdl_stm)
                    pred_bucket = _wdl_bucket(pred_wdl)
                    target_bucket = _wdl_bucket(target_wdl)

                    pred_cpu = prediction_cp.detach().cpu().reshape(-1).to(dtype=torch.float64)
                    target_cpu = normalized_scores.detach().cpu().reshape(-1).to(dtype=torch.float64)
                    delta = pred_cpu - target_cpu

                    total_loss += float(losses["loss"].detach().cpu().item()) * batch_positions
                    total_cp_loss += float(losses["cp_loss"].detach().cpu().item()) * batch_positions
                    total_wdl_loss += float(losses["wdl_loss"].detach().cpu().item()) * batch_positions
                    total_output_reg_loss += float(losses["output_reg_loss"].detach().cpu().item()) * batch_positions
                    total_sanity_anchor_loss += (
                        float(losses["sanity_anchor_loss"].detach().cpu().item()) * batch_positions
                    )
                    total_correct += int((pred_bucket == result_bucket).sum().detach().cpu().item())
                    total_teacher_result_disagreement += int(
                        (target_bucket != result_bucket).sum().detach().cpu().item()
                    )
                    total_abs_error += float(delta.abs().sum().item())
                    total_sq_error += float(delta.square().sum().item())
                    corr_sum_x += float(pred_cpu.sum().item())
                    corr_sum_y += float(target_cpu.sum().item())
                    corr_sum_x2 += float(pred_cpu.square().sum().item())
                    corr_sum_y2 += float(target_cpu.square().sum().item())
                    corr_sum_xy += float((pred_cpu * target_cpu).sum().item())
                    total_positions += batch_positions
                    batches += 1

    if was_training:
        state.model.train()

    validation_seconds = max(0.0, time.monotonic() - started_at)
    average_loss = math.inf if total_positions == 0 else total_loss / total_positions
    average_cp_loss = math.inf if total_positions == 0 else total_cp_loss / total_positions
    average_wdl_loss = math.inf if total_positions == 0 else total_wdl_loss / total_positions
    average_output_reg_loss = math.inf if total_positions == 0 else total_output_reg_loss / total_positions
    average_sanity_anchor_loss = math.inf if total_positions == 0 else total_sanity_anchor_loss / total_positions
    wdl_accuracy = 0.0 if total_positions == 0 else total_correct / total_positions
    teacher_result_disagreement_rate = (
        0.0 if total_positions == 0 else total_teacher_result_disagreement / total_positions
    )
    validation_positions_per_second = (
        0.0 if total_positions == 0 or validation_seconds <= 0.0 else float(total_positions) / validation_seconds
    )
    cp_mae = 0.0 if total_positions == 0 else total_abs_error / total_positions
    cp_rmse = 0.0 if total_positions == 0 else math.sqrt(total_sq_error / total_positions)
    cp_corr = _pearson_correlation(
        total_positions,
        corr_sum_x,
        corr_sum_y,
        corr_sum_x2,
        corr_sum_y2,
        corr_sum_xy,
    )
    material_sanity = _material_sanity_snapshot(state)

    return {
        "event": "validation",
        "global_step": state.global_step,
        "positions_seen": state.positions_seen,
        "validation_loss": average_loss,
        "validation_cp_loss": average_cp_loss,
        "validation_wdl_loss": average_wdl_loss,
        "validation_output_reg_loss": average_output_reg_loss,
        "validation_sanity_anchor_loss": average_sanity_anchor_loss,
        "cp_mae": cp_mae,
        "cp_rmse": cp_rmse,
        "cp_corr": cp_corr,
        "wdl_accuracy": wdl_accuracy,
        "teacher_result_disagreement_rate": teacher_result_disagreement_rate,
        "validation_positions": total_positions,
        "validation_batches": batches,
        "validation_seconds": validation_seconds,
        "validation_positions_per_second": validation_positions_per_second,
        "material_sanity": material_sanity,
        "material_ordering_ok": bool(material_sanity["ordering_ok"]),
        "wdl_lambda": state.config.wdl_lambda,
    }


def _material_sanity_snapshot(state: TrainState) -> dict[str, object]:
    torch = _require_torch()
    from .board import BoardState
    from .features import active_feature_indices

    white_indices: list[list[int]] = []
    black_indices: list[list[int]] = []
    stm: list[float] = []
    for _, fen in MATERIAL_SANITY_POSITIONS:
        board = BoardState.from_fen(fen)
        white = active_feature_indices(board, "white")
        black = active_feature_indices(board, "black")
        white_indices.append(white + [-1] * (state.config.max_active_features - len(white)))
        black_indices.append(black + [-1] * (state.config.max_active_features - len(black)))
        stm.append(1.0 if board.side_to_move == "w" else 0.0)

    white_tensor = torch.tensor(white_indices, dtype=torch.long, device=state.device)
    black_tensor = torch.tensor(black_indices, dtype=torch.long, device=state.device)
    stm_tensor = torch.tensor(stm, dtype=torch.float32, device=state.device).unsqueeze(1)
    predictions = state.model(white_tensor, black_tensor, stm_tensor)
    values = [float(value) for value in predictions.detach().cpu().reshape(-1).tolist()]
    named = {
        name: value
        for (name, _), value in zip(MATERIAL_SANITY_POSITIONS, values, strict=True)
    }
    named["ordering_ok"] = (
        named["starting_position"]
        < named["white_up_pawn"]
        < named["white_up_knight"]
        < named["white_up_rook"]
        < named["white_up_queen"]
    )
    named["starting_position_near_zero"] = abs(named["starting_position"]) <= 50.0
    return named


def _sanity_anchor_tensors(config: TrainConfig, torch, device: str) -> dict[str, object]:
    from .board import BoardState
    from .features import active_feature_indices

    white_indices: list[list[int]] = []
    black_indices: list[list[int]] = []
    stm: list[float] = []
    for _, fen in SANITY_ANCHOR_POSITIONS:
        board = BoardState.from_fen(fen)
        white = active_feature_indices(board, "white")
        black = active_feature_indices(board, "black")
        white_indices.append(white + [-1] * (config.max_active_features - len(white)))
        black_indices.append(black + [-1] * (config.max_active_features - len(black)))
        stm.append(1.0 if board.side_to_move == "w" else 0.0)

    return {
        "white_indices": torch.tensor(white_indices, dtype=torch.long, device=device),
        "black_indices": torch.tensor(black_indices, dtype=torch.long, device=device),
        "stm": torch.tensor(stm, dtype=torch.float32, device=device).unsqueeze(1),
    }


def _pearson_correlation(
    count: int,
    sum_x: float,
    sum_y: float,
    sum_x2: float,
    sum_y2: float,
    sum_xy: float,
) -> float:
    if count < 2:
        return 0.0
    numerator = count * sum_xy - sum_x * sum_y
    denom_x = count * sum_x2 - sum_x * sum_x
    denom_y = count * sum_y2 - sum_y * sum_y
    if denom_x <= 1e-12 or denom_y <= 1e-12:
        return 0.0
    value = numerator / math.sqrt(denom_x * denom_y)
    if not math.isfinite(value):
        return 0.0
    return float(value)


def _maybe_update_best_checkpoint(state: TrainState, validation_loss: float) -> bool:
    if not math.isfinite(validation_loss):
        return False

    if state.best_validation_loss is not None and validation_loss >= state.best_validation_loss:
        return False

    state.best_validation_loss = validation_loss
    state.best_validation_positions = state.positions_seen
    _write_best_checkpoint(state)
    return True


def _write_best_checkpoint(state: TrainState) -> Path:
    checkpoints_dir = state.run_dir / "checkpoints"
    alias_path = checkpoints_dir / "best.pt"
    stamped_path = _best_step_checkpoint_path(state.run_dir, state.global_step)
    _save_training_checkpoint(state, alias_path)
    _save_training_checkpoint(state, stamped_path)
    for candidate in checkpoints_dir.glob("best_step_*.pt"):
        if candidate != stamped_path:
            candidate.unlink()
    return stamped_path


def _best_step_checkpoint_path(run_dir: Path, global_step: int) -> Path:
    return run_dir / "checkpoints" / f"best_step_{global_step:08d}.pt"


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


class _NullScaledLoss:
    def __init__(self, loss):
        self._loss = loss

    def backward(self) -> None:
        self._loss.backward()


class _NullGradScaler:
    def scale(self, loss):
        return _NullScaledLoss(loss)

    def unscale_(self, optimizer) -> None:
        return None

    def step(self, optimizer) -> None:
        optimizer.step()

    def update(self) -> None:
        return None

    def state_dict(self) -> dict[str, object]:
        return {}

    def load_state_dict(self, state: dict[str, object]) -> None:
        return None


def _amp_enabled(torch, config: TrainConfig, device: str) -> bool:
    if not config.amp:
        return False
    if device not in {"cuda", "mps", "cpu", "xpu"}:
        return False
    if not hasattr(torch, "autocast"):
        return device == "cuda" and hasattr(getattr(torch, "cuda", None), "amp")

    autocast_mode = getattr(getattr(torch, "amp", None), "autocast_mode", None)
    if autocast_mode is not None and hasattr(autocast_mode, "is_autocast_available"):
        try:
            return bool(autocast_mode.is_autocast_available(device))
        except Exception:
            return False

    return device == "cuda"


def _autocast_context(torch, device: str, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "autocast"):
        try:
            return torch.autocast(device_type=device, enabled=True)
        except TypeError:
            return torch.autocast(device, enabled=True)
    if device == "cuda" and hasattr(getattr(torch, "cuda", None), "amp"):
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def _create_grad_scaler(torch, config: TrainConfig, device: str):
    enabled = _amp_enabled(torch, config, device)
    amp_namespace = getattr(torch, "amp", None)
    if amp_namespace is not None and hasattr(amp_namespace, "GradScaler"):
        try:
            return amp_namespace.GradScaler(device=device, enabled=enabled)
        except TypeError:
            try:
                return amp_namespace.GradScaler(device, enabled=enabled)
            except TypeError:
                pass
        except Exception:
            if not enabled:
                return _NullGradScaler()

    if device == "cuda" and hasattr(getattr(torch, "cuda", None), "amp"):
        return torch.cuda.amp.GradScaler(enabled=enabled)
    return _NullGradScaler()


def _effective_validation_interval_positions(config: TrainConfig) -> int:
    if config.validation_interval_positions > 0:
        return config.validation_interval_positions
    return config.epoch_positions


def _next_validation_positions(current_positions: int, interval_positions: int) -> int:
    return ((current_positions // interval_positions) + 1) * interval_positions


def _epoch_index(positions_seen: int, epoch_positions: int) -> int:
    return positions_seen // epoch_positions


def _advance_scheduler_for_epoch_boundaries(
    state: TrainState,
    *,
    positions_before_step: int,
    positions_after_step: int,
) -> None:
    if state.scheduler is None:
        return
    previous_epoch_index = _epoch_index(positions_before_step, state.config.epoch_positions)
    new_epoch_index = _epoch_index(positions_after_step, state.config.epoch_positions)
    completed_epochs = max(0, new_epoch_index - previous_epoch_index)
    for _ in range(completed_epochs):
        state.scheduler.step()


def _requested_batch_size(batch_size: int, remaining_positions: int | None) -> int | None:
    if remaining_positions is not None and remaining_positions <= 0:
        return None
    if remaining_positions is None:
        return batch_size
    return min(batch_size, remaining_positions)


def _consume_positions(remaining_positions: int | None, batch_positions: int) -> int | None:
    if remaining_positions is None:
        return None
    return max(0, remaining_positions - batch_positions)


def _prepare_batch(batch, torch) -> _PreparedBatch:
    return _PreparedBatch(
        batch_positions=_batch_size(batch),
        tensors={
            "white_indices": torch.from_numpy(batch.white_indices).to(dtype=torch.long),
            "black_indices": torch.from_numpy(batch.black_indices).to(dtype=torch.long),
            "stm": torch.from_numpy(batch.stm).to(dtype=torch.float32).unsqueeze(1),
            "score_cp": torch.from_numpy(batch.score_cp).to(dtype=torch.float32).unsqueeze(1),
            "result_wdl": torch.from_numpy(batch.result_wdl).to(dtype=torch.float32).unsqueeze(1),
        },
    )


def _move_tensors_to_device(tensors: dict[str, object], device: str) -> dict[str, object]:
    if device == "cpu":
        return tensors
    return {name: tensor.to(device=device) for name, tensor in tensors.items()}


def _batch_size(batch) -> int:
    return int(batch.stm.shape[0])


def _binpack_filter_options(config: TrainConfig) -> dict[str, object]:
    return {
        "skip_capture_positions": config.skip_capture_positions,
        "skip_decisive_score_mismatch": config.skip_decisive_score_mismatch,
        "decisive_score_mismatch_margin": config.decisive_score_mismatch_margin,
        "skip_draw_score_mismatch": config.skip_draw_score_mismatch,
        "draw_score_mismatch_margin": config.draw_score_mismatch_margin,
        "max_abs_score": config.max_abs_score,
    }


def _wdl_expectation_from_cp(values, offset: float, scaling: float, torch):
    win = torch.sigmoid((values - offset) / scaling)
    loss = torch.sigmoid((-values - offset) / scaling)
    draw = 1.0 - win - loss
    return (win + 0.5 * draw).clamp(0.0, 1.0)


def _wdl_bucket(values):
    return values.mul(3.0).clamp_min(0.0).clamp_max(2.999999).to(dtype=values.dtype).to(dtype=values.long().dtype)


def _payload_positions_seen(payload: dict[str, object]) -> int:
    return int(payload["positions_seen"])


def _payload_epoch_index(payload: dict[str, object]) -> int:
    return int(payload["epoch_index"])


def _payload_best_validation_positions(payload: dict[str, object]) -> int | None:
    if payload.get("best_validation_positions") is not None:
        return int(payload["best_validation_positions"])
    return None
