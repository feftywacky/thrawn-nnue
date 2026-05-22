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
from .native import BinpackStream


_STOCKFISH_INTERNAL_TO_CP = 100.0 / 208.0


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
    best_checkpoint_metric_name: str = "score_mae"
    best_checkpoint_metric_value: float | None = None
    best_checkpoint_positions: int | None = None
    best_checkpoint_sort_key: tuple[float, ...] | None = None
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


def train_from_config(
    config: TrainConfig,
    *,
    console_mode: str | None = None,
    init_checkpoint: str | Path | None = None,
) -> Path:
    if not config.datasets:
        raise ValueError("datasets must not be empty")

    if console_mode is not None:
        config.console_mode = console_mode
        config.validate()
    state = _create_state(config)
    if init_checkpoint is not None:
        _load_initial_model_state(state, init_checkpoint)
    _run_training_loop(state)
    final_checkpoint = state.run_dir / "checkpoints" / f"step_{state.global_step:08d}.pt"
    _save_training_checkpoint(state, final_checkpoint)
    return final_checkpoint


def resume_training(checkpoint_path: str | Path, *, console_mode: str | None = None) -> Path:
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    config = TrainConfig.from_dict(_checkpoint_config_dict(payload["config"]))
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
    state.best_checkpoint_metric_name = "score_mae"
    state.best_checkpoint_metric_value = _payload_optional_float(payload, "best_checkpoint_metric_value")
    state.best_checkpoint_positions = _payload_optional_int(payload, "best_checkpoint_positions")
    state.best_checkpoint_sort_key = _payload_optional_float_tuple(payload, "best_checkpoint_sort_key")
    _restore_best_checkpoint_tracking_from_metrics(state)
    state.global_step = int(payload["global_step"])
    state.positions_seen = _payload_positions_seen(payload)
    state.epoch_index = _payload_epoch_index(payload)
    state.model.to(state.device)
    _run_training_loop(state)
    final_checkpoint = state.run_dir / "checkpoints" / f"step_{state.global_step:08d}.pt"
    _save_training_checkpoint(state, final_checkpoint)
    return final_checkpoint


def _checkpoint_config_dict(raw_config: object) -> dict[str, object]:
    config = dict(raw_config)  # type: ignore[arg-type]
    if "soft_early_fen_skipping" not in config:
        config.setdefault("early_fen_skipping", -1)
        config.setdefault("soft_early_fen_skipping", 0)
        config.setdefault("simple_eval_skipping", -1)
        config.setdefault("param_index", 0)
        config.setdefault("pc_y0", 0.0)
        config.setdefault("pc_y1", 0.4)
        config.setdefault("pc_y2", 1.0)
        config.setdefault("pc_y3", 1.0)
        config.setdefault("pc_y4", 0.75)
        config.setdefault("ply_x1", 0.0)
        config.setdefault("ply_y1", 0.1)
        config.setdefault("ply_x2", 6.0)
        config.setdefault("ply_y2", 0.15)
        config.setdefault("ply_x3", 10.0)
        config.setdefault("ply_y3", 0.25)
        config.setdefault("ply_x4", 18.0)
        config.setdefault("ply_y4", 0.75)
    return config


def _create_state(config: TrainConfig) -> TrainState:
    torch = _require_torch()
    from .model import HalfKAv2HmNNUE

    _resolve_runtime_config(config)
    _seed_torch(config, torch)
    device = _select_device(config.accelerator, torch)
    _configure_torch_backend(config, device, torch)
    run_dir = Path(config.default_root_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"

    model = HalfKAv2HmNNUE(
        num_features=config.num_features,
        ft_size=config.ft_size,
        hidden_size=config.hidden_size,
        forward_size=config.forward_size,
        fc1_output_size=config.fc1_output_size,
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


def _load_initial_model_state(state: TrainState, checkpoint_path: str | Path) -> None:
    payload = load_checkpoint(checkpoint_path, map_location="cpu")
    state.model.load_state_dict(payload["model_state"])
    state.model.to(state.device)


class _PreparedBatchSource:
    def __init__(
        self,
        stream,
        *,
        batch_size: int,
        total_positions: int | None,
        queue_size: int,
        pin_memory: bool,
        torch,
    ) -> None:
        self._stream = stream
        self._batch_size = batch_size
        self._remaining_positions = total_positions
        self._queue_size = queue_size
        self._pin_memory = pin_memory
        self._torch = torch
        self._queue: Queue[object] | None = None
        self._stop_event = Event()
        self._thread: Thread | None = None
        self._closed = False

    def __enter__(self) -> "_PreparedBatchSource":
        if self._queue_size > 0:
            self._queue = Queue(maxsize=self._queue_size)
            self._thread = Thread(target=self._run_producer, name="thrawn-batch-prefetch", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __iter__(self) -> "_PreparedBatchSource":
        return self

    def __next__(self) -> _PreparedBatch:
        if self._queue_size <= 0:
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
        prepared = _prepare_batch(batch, self._torch, pin_memory=self._pin_memory)
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
                prepared = _prepare_batch(batch, self._torch, pin_memory=self._pin_memory)
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
    total_positions = state.config.total_positions
    log_every_n_steps = _log_every_n_steps(state.config)
    last_validation_epoch: int | None = None

    reporter.startup(
        ConsoleContext(
            run_name=state.config.run_name,
            accelerator=state.device,
            train_shards=len(state.config.datasets),
            validation_shards=len(_validation_datasets(state.config)),
            max_epochs=state.config.max_epochs,
            epoch_size=state.config.epoch_size,
            total_positions=total_positions,
            initial_positions_seen=state.positions_seen,
            batch_size=state.config.batch_size,
            log_every_n_steps=log_every_n_steps,
            nnue2score=state.config.nnue2score,
            data_loader_queue_size=state.config.data_loader_queue_size,
        )
    )

    try:
        with BinpackStream(
            state.config.datasets,
            num_threads=state.config.num_workers,
            cyclic=True,
            **_binpack_filter_options(state.config, split_role="train"),
        ) as train_stream:
            remaining_positions = total_positions - state.positions_seen
            with _PreparedBatchSource(
                train_stream,
                batch_size=state.config.batch_size,
                total_positions=remaining_positions,
                queue_size=state.config.data_loader_queue_size,
                pin_memory=_pin_memory_enabled(state),
                torch=torch,
            ) as train_batches:
                while state.positions_seen < total_positions:
                    step_started_at = time.monotonic()
                    try:
                        prepared_batch = next(train_batches)
                    except StopIteration:
                        raise RuntimeError("Cyclic training stream unexpectedly returned EOF") from None

                    batch_positions = prepared_batch.batch_positions
                    positions_before_step = state.positions_seen
                    epoch_before_step = _epoch_index(positions_before_step, state.config.epoch_size)
                    losses = _run_train_step(
                        state,
                        prepared_batch.tensors,
                        torch,
                        autocast_enabled=autocast_enabled,
                    )
                    state.global_step += 1
                    state.positions_seen += batch_positions
                    state.epoch_index = _epoch_index(state.positions_seen, state.config.epoch_size)
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

                    if state.global_step % log_every_n_steps == 0:
                        lambda_weight = _current_lambda(state.config, epoch_before_step)
                        _log_metrics(
                            state,
                            {
                                "event": "train",
                                "global_step": state.global_step,
                                "positions_seen": state.positions_seen,
                                "epoch_index": state.epoch_index,
                                "batch_positions": batch_positions,
                                "loss": current_loss,
                                "wdl_loss": float(losses["wdl_loss"].detach().cpu().item()),
                                "teacher_wdl_loss": float(losses["teacher_wdl_loss"].detach().cpu().item()),
                                "result_wdl_loss": float(losses["result_wdl_loss"].detach().cpu().item()),
                                "output_reg_loss": float(losses["output_reg_loss"].detach().cpu().item()),
                                "lambda": lambda_weight,
                                "start_lambda": state.config.start_lambda,
                                "end_lambda": state.config.end_lambda,
                                "nnue2score": state.config.nnue2score,
                                "in_offset": state.config.in_offset,
                                "in_scaling": state.config.in_scaling,
                                "out_offset": state.config.out_offset,
                                "out_scaling": state.config.out_scaling,
                                "filtered": state.config.filtered,
                                "wld_filtered": state.config.wld_filtered,
                                "random_fen_skipping": state.config.random_fen_skipping,
                                "early_fen_skipping": state.config.early_fen_skipping,
                                "soft_early_fen_skipping": state.config.soft_early_fen_skipping,
                                "simple_eval_skipping": state.config.simple_eval_skipping,
                                "param_index": state.config.param_index,
                                "pc_y0": state.config.pc_y0,
                                "pc_y1": state.config.pc_y1,
                                "pc_y2": state.config.pc_y2,
                                "pc_y3": state.config.pc_y3,
                                "pc_y4": state.config.pc_y4,
                                "ply_x1": state.config.ply_x1,
                                "ply_y1": state.config.ply_y1,
                                "ply_x2": state.config.ply_x2,
                                "ply_y2": state.config.ply_y2,
                                "ply_x3": state.config.ply_x3,
                                "ply_y3": state.config.ply_y3,
                                "ply_x4": state.config.ply_x4,
                                "ply_y4": state.config.ply_y4,
                                "max_epochs": state.config.max_epochs,
                                "epoch_size": state.config.epoch_size,
                                "lr": current_lr,
                                "step_seconds": step_seconds,
                                "train_positions_per_second": train_positions_per_second,
                            },
                        )

                    for completed_epoch in range(epoch_before_step + 1, state.epoch_index + 1):
                        if _should_validate_epoch(state.config, completed_epoch):
                            _run_validation_and_report(state, reporter)
                            last_validation_epoch = completed_epoch
                        if _should_checkpoint_epoch(state.config, completed_epoch):
                            checkpoint_path = _epoch_checkpoint_path(
                                state.run_dir,
                                completed_epoch,
                                state.global_step,
                            )
                            _save_training_checkpoint(state, checkpoint_path)
                            reporter.checkpoint_saved(str(checkpoint_path), is_best=False)

        if (
            state.config.save_last_network
            and state.config.validation_size > 0
            and _should_validate_epoch(state.config, state.epoch_index)
            and last_validation_epoch != state.epoch_index
        ):
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
        best_checkpoint_metric_name=state.best_checkpoint_metric_name,
        best_checkpoint_metric_value=state.best_checkpoint_metric_value,
        best_checkpoint_positions=state.best_checkpoint_positions,
        best_checkpoint_sort_key=state.best_checkpoint_sort_key,
    )


def _resolve_runtime_config(config: TrainConfig) -> None:
    config.validate()


def _seed_torch(config: TrainConfig, torch) -> None:
    torch.manual_seed(config.seed)
    if hasattr(torch, "cuda"):
        try:
            torch.cuda.manual_seed_all(config.seed)
        except Exception:
            pass


def _create_scheduler(config: TrainConfig, optimizer, torch):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma)


def _create_optimizer(config: TrainConfig, model, torch):
    kwargs = {
        "lr": config.lr,
        "weight_decay": config.weight_decay,
    }
    if _model_device_type(model) == "cuda":
        if config.fused_optimizer:
            try:
                return torch.optim.AdamW(model.parameters(), **kwargs, fused=True)
            except (RuntimeError, TypeError):
                pass
        return torch.optim.AdamW(model.parameters(), **kwargs, foreach=True)
    return torch.optim.AdamW(model.parameters(), **kwargs, foreach=False)


def _configure_torch_backend(config: TrainConfig, device: str, torch) -> None:
    if device != "cuda":
        return
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if config.tf32 else "highest")

    cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
    matmul_backend = getattr(cuda_backend, "matmul", None)
    if matmul_backend is not None and hasattr(matmul_backend, "allow_tf32"):
        matmul_backend.allow_tf32 = bool(config.tf32)

    cudnn_backend = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn_backend is not None and hasattr(cudnn_backend, "allow_tf32"):
        cudnn_backend.allow_tf32 = bool(config.tf32)


def _model_device_type(model) -> str:
    try:
        parameter = next(model.parameters())
    except StopIteration:
        return "cpu"
    return str(parameter.device.type)


def _run_train_step(state: TrainState, tensors, torch, *, autocast_enabled: bool):
    tensors = _move_tensors_to_device(tensors, state.device, non_blocking=_pin_memory_enabled(state))
    score_stm, result_wdl_stm = _stm_oriented_targets(
        tensors["score"],
        tensors["result_wdl"],
        tensors["stm"],
        torch,
    )
    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)
    lambda_weight = _current_lambda(
        state.config,
        _epoch_index(state.positions_seen, state.config.epoch_size),
    )
    with _autocast_context(torch, state.device, autocast_enabled):
        prediction_raw = state.model(
            tensors["white_indices"],
            tensors["black_indices"],
            tensors["stm"],
        )
        prediction_score = _model_output_to_score(prediction_raw, state.config.nnue2score)
        losses = _scalar_head_loss(
            prediction_score,
            score_stm,
            result_wdl_stm,
            lambda_weight=lambda_weight,
            in_offset=state.config.in_offset,
            out_offset=state.config.out_offset,
            in_scaling=state.config.in_scaling,
            out_scaling=state.config.out_scaling,
            pow_exp=state.config.pow_exp,
            qp_asymmetry=state.config.qp_asymmetry,
            w1=state.config.w1,
            w2=state.config.w2,
            torch=torch,
        )

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
    is_best = _maybe_update_best_checkpoint(state, metrics)
    metrics["is_best"] = is_best
    _log_metrics(state, metrics)
    reporter.validation_finished(metrics, is_best=is_best)
    if is_best:
        reporter.checkpoint_saved(
            str(_best_checkpoint_path(state.run_dir, state.epoch_index)),
            is_best=True,
        )


def _scalar_head_loss(
    prediction_score,
    target_score,
    result_wdl,
    *,
    lambda_weight: float,
    in_offset: float,
    out_offset: float,
    in_scaling: float,
    out_scaling: float,
    pow_exp: float,
    qp_asymmetry: float,
    w1: float,
    w2: float,
    torch,
):
    # Match Stockfish nnue-pytorch naming: "in" applies to the network score,
    # "out" applies to the search-score target read from the training data.
    pred_wdl = _wdl_expectation_from_score(prediction_score, in_offset, in_scaling, torch)
    target_wdl = _wdl_expectation_from_score(target_score, out_offset, out_scaling, torch)
    blended_target = lambda_weight * target_wdl + (1.0 - lambda_weight) * result_wdl
    per_sample_loss = (pred_wdl - blended_target).abs().pow(pow_exp)
    if qp_asymmetry != 0.0:
        per_sample_loss = per_sample_loss * ((pred_wdl > blended_target) * qp_asymmetry + 1.0)
    weights = 1.0 + (2.0**w1 - 1.0) * torch.pow(
        (target_wdl - 0.5).square() * target_wdl * (1.0 - target_wdl),
        w2,
    )
    wdl_loss = (per_sample_loss * weights).sum() / weights.sum()
    teacher_wdl_loss = torch.mean((pred_wdl - target_wdl).abs().pow(pow_exp))
    result_wdl_loss = torch.mean((pred_wdl - result_wdl).abs().pow(pow_exp))
    output_reg_loss = torch.mean(prediction_score.square())
    return {
        "loss": wdl_loss,
        "wdl_loss": wdl_loss,
        "teacher_wdl_loss": teacher_wdl_loss,
        "result_wdl_loss": result_wdl_loss,
        "output_reg_loss": output_reg_loss,
        "predicted_score": prediction_score,
        "predicted_wdl": pred_wdl,
    }


def _model_output_to_score(prediction, nnue2score: float):
    return prediction * float(nnue2score)


def _clip_model_weights(model, config: TrainConfig) -> None:
    dense_limit = (127.0 - 0.5) / max(config.export_dense_scale, 1.0)
    torch = _require_torch()
    with torch.no_grad():
        model.fc0.weight.clamp_(-dense_limit, dense_limit)
        model.fc1.weight.clamp_(-dense_limit, dense_limit)
        model.fc2.weight.clamp_(-dense_limit, dense_limit)


def _stm_oriented_targets(score, result_wdl, stm, torch):
    # Binpack teacher scores and results are already side-to-move oriented.
    return score, result_wdl


def _run_validation(state: TrainState) -> dict[str, object]:
    torch = _require_torch()
    autocast_enabled = _amp_enabled(torch, state.config, state.device)
    started_at = time.monotonic()

    total_loss = 0.0
    total_wdl_loss = 0.0
    total_teacher_wdl_loss = 0.0
    total_result_wdl_loss = 0.0
    total_output_reg_loss = 0.0
    total_positions = 0
    total_correct = 0
    total_teacher_result_disagreement = 0
    total_score_abs_error = 0.0
    total_score_sq_error = 0.0
    corr_sum_x = 0.0
    corr_sum_y = 0.0
    corr_sum_x2 = 0.0
    corr_sum_y2 = 0.0
    corr_sum_xy = 0.0
    batches = 0
    lambda_weight = _current_lambda(state.config, state.epoch_index)

    was_training = state.model.training
    state.model.eval()
    with BinpackStream(
        _validation_datasets(state.config),
        num_threads=state.config.num_workers,
        cyclic=False,
        **_binpack_filter_options(state.config, split_role="validation"),
    ) as validation_stream:
        validation_budget = state.config.validation_size if state.config.validation_size > 0 else None
        with _PreparedBatchSource(
            validation_stream,
            batch_size=state.config.batch_size,
            total_positions=validation_budget,
            queue_size=state.config.data_loader_queue_size,
            pin_memory=_pin_memory_enabled(state),
            torch=torch,
        ) as validation_batches:
            with torch.no_grad():
                for prepared_batch in validation_batches:
                    batch_positions = prepared_batch.batch_positions
                    tensors = _move_tensors_to_device(
                        prepared_batch.tensors,
                        state.device,
                        non_blocking=_pin_memory_enabled(state),
                    )
                    score_stm, result_wdl_stm = _stm_oriented_targets(
                        tensors["score"],
                        tensors["result_wdl"],
                        tensors["stm"],
                        torch,
                    )
                    with _autocast_context(torch, state.device, autocast_enabled):
                        prediction_raw = state.model(
                            tensors["white_indices"],
                            tensors["black_indices"],
                            tensors["stm"],
                        )
                        prediction_score = _model_output_to_score(prediction_raw, state.config.nnue2score)
                        losses = _scalar_head_loss(
                            prediction_score,
                            score_stm,
                            result_wdl_stm,
                            lambda_weight=lambda_weight,
                            in_offset=state.config.in_offset,
                            out_offset=state.config.out_offset,
                            in_scaling=state.config.in_scaling,
                            out_scaling=state.config.out_scaling,
                            pow_exp=state.config.pow_exp,
                            qp_asymmetry=state.config.qp_asymmetry,
                            w1=state.config.w1,
                            w2=state.config.w2,
                            torch=torch,
                        )

                    pred_wdl = losses["predicted_wdl"]
                    target_wdl = _wdl_expectation_from_score(
                        score_stm,
                        state.config.out_offset,
                        state.config.out_scaling,
                        torch,
                    )
                    result_bucket = _wdl_bucket(result_wdl_stm)
                    pred_bucket = _wdl_bucket(pred_wdl)
                    target_bucket = _wdl_bucket(target_wdl)

                    pred_cpu = prediction_score.detach().cpu().reshape(-1).to(dtype=torch.float64)
                    target_cpu = score_stm.detach().cpu().reshape(-1).to(dtype=torch.float64)
                    delta = pred_cpu - target_cpu

                    total_loss += float(losses["loss"].detach().cpu().item()) * batch_positions
                    total_wdl_loss += float(losses["wdl_loss"].detach().cpu().item()) * batch_positions
                    total_teacher_wdl_loss += float(losses["teacher_wdl_loss"].detach().cpu().item()) * batch_positions
                    total_result_wdl_loss += float(losses["result_wdl_loss"].detach().cpu().item()) * batch_positions
                    total_output_reg_loss += float(losses["output_reg_loss"].detach().cpu().item()) * batch_positions
                    total_correct += int((pred_bucket == result_bucket).sum().detach().cpu().item())
                    total_teacher_result_disagreement += int(
                        (target_bucket != result_bucket).sum().detach().cpu().item()
                    )
                    total_score_abs_error += float(delta.abs().sum().item())
                    total_score_sq_error += float(delta.square().sum().item())
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
    average_wdl_loss = math.inf if total_positions == 0 else total_wdl_loss / total_positions
    average_teacher_wdl_loss = math.inf if total_positions == 0 else total_teacher_wdl_loss / total_positions
    average_result_wdl_loss = math.inf if total_positions == 0 else total_result_wdl_loss / total_positions
    average_output_reg_loss = math.inf if total_positions == 0 else total_output_reg_loss / total_positions
    wdl_accuracy = 0.0 if total_positions == 0 else total_correct / total_positions
    teacher_result_disagreement_rate = (
        0.0 if total_positions == 0 else total_teacher_result_disagreement / total_positions
    )
    validation_positions_per_second = (
        0.0 if total_positions == 0 or validation_seconds <= 0.0 else float(total_positions) / validation_seconds
    )
    score_mae = 0.0 if total_positions == 0 else total_score_abs_error / total_positions
    score_rmse = 0.0 if total_positions == 0 else math.sqrt(total_score_sq_error / total_positions)
    score_corr = _pearson_correlation(
        total_positions,
        corr_sum_x,
        corr_sum_y,
        corr_sum_x2,
        corr_sum_y2,
        corr_sum_xy,
    )
    cp_mae = score_mae * _STOCKFISH_INTERNAL_TO_CP
    cp_rmse = score_rmse * _STOCKFISH_INTERNAL_TO_CP

    return {
        "event": "validation",
        "global_step": state.global_step,
        "positions_seen": state.positions_seen,
        "validation_loss": average_loss,
        "validation_wdl_loss": average_wdl_loss,
        "validation_teacher_wdl_loss": average_teacher_wdl_loss,
        "validation_result_wdl_loss": average_result_wdl_loss,
        "validation_output_reg_loss": average_output_reg_loss,
        "score_mae": score_mae,
        "score_rmse": score_rmse,
        "score_corr": score_corr,
        "cp_mae": cp_mae,
        "cp_rmse": cp_rmse,
        "cp_corr": score_corr,
        "wdl_accuracy": wdl_accuracy,
        "teacher_result_disagreement_rate": teacher_result_disagreement_rate,
        "validation_positions": total_positions,
        "validation_batches": batches,
        "validation_seconds": validation_seconds,
        "validation_positions_per_second": validation_positions_per_second,
        "lambda": lambda_weight,
        "start_lambda": state.config.start_lambda,
        "end_lambda": state.config.end_lambda,
        "nnue2score": state.config.nnue2score,
        "in_offset": state.config.in_offset,
        "in_scaling": state.config.in_scaling,
        "out_offset": state.config.out_offset,
        "out_scaling": state.config.out_scaling,
        "filtered": state.config.filtered,
        "wld_filtered": state.config.wld_filtered,
        "random_fen_skipping": state.config.random_fen_skipping,
        "early_fen_skipping": state.config.early_fen_skipping,
        "soft_early_fen_skipping": state.config.soft_early_fen_skipping,
        "simple_eval_skipping": state.config.simple_eval_skipping,
        "param_index": state.config.param_index,
        "pc_y0": state.config.pc_y0,
        "pc_y1": state.config.pc_y1,
        "pc_y2": state.config.pc_y2,
        "pc_y3": state.config.pc_y3,
        "pc_y4": state.config.pc_y4,
        "ply_x1": state.config.ply_x1,
        "ply_y1": state.config.ply_y1,
        "ply_x2": state.config.ply_x2,
        "ply_y2": state.config.ply_y2,
        "ply_x3": state.config.ply_x3,
        "ply_y3": state.config.ply_y3,
        "ply_x4": state.config.ply_x4,
        "ply_y4": state.config.ply_y4,
        "max_epochs": state.config.max_epochs,
        "epoch_size": state.config.epoch_size,
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


def _maybe_update_best_checkpoint(state: TrainState, metrics: dict[str, object]) -> bool:
    score_mae = float(metrics["score_mae"])
    if not math.isfinite(score_mae):
        return False

    is_best = False
    candidate_key = _best_checkpoint_sort_key(metrics)
    if state.best_checkpoint_sort_key is None or candidate_key < state.best_checkpoint_sort_key:
        state.best_checkpoint_metric_name = "score_mae"
        state.best_checkpoint_metric_value = score_mae
        state.best_checkpoint_positions = state.positions_seen
        state.best_checkpoint_sort_key = candidate_key
        is_best = True

    if is_best:
        _write_best_checkpoint(state)

    return is_best


def _write_best_checkpoint(state: TrainState) -> Path:
    checkpoints_dir = state.run_dir / "checkpoints"
    stamped_path = _best_checkpoint_path(state.run_dir, state.epoch_index)
    _save_training_checkpoint(state, stamped_path)
    for candidate in checkpoints_dir.glob("epoch_*_best.pt"):
        if candidate != stamped_path:
            candidate.unlink(missing_ok=True)
    return stamped_path


def _best_checkpoint_path(run_dir: Path, epoch_index: int) -> Path:
    return run_dir / "checkpoints" / f"epoch_{epoch_index:04d}_best.pt"


def _best_checkpoint_sort_key(metrics: dict[str, object]) -> tuple[float, ...]:
    return (
        float(metrics["score_mae"]),
        float(metrics["validation_loss"]),
        float(metrics["validation_teacher_wdl_loss"]),
        -float(metrics["score_corr"]),
    )


def _restore_best_checkpoint_tracking_from_metrics(state: TrainState) -> None:
    if state.best_checkpoint_sort_key is not None:
        return

    if not state.metrics_path.exists():
        return

    best_record: dict[str, object] | None = None
    best_key: tuple[float, ...] | None = None
    with state.metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("event") != "validation":
                continue
            try:
                candidate_key = _best_checkpoint_sort_key(record)
            except (KeyError, TypeError, ValueError):
                continue
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_record = record

    if best_record is None or best_key is None:
        return

    state.best_checkpoint_metric_name = "score_mae"
    state.best_checkpoint_sort_key = best_key
    state.best_checkpoint_metric_value = float(best_record["score_mae"])
    state.best_checkpoint_positions = int(best_record["positions_seen"])


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


def _epoch_index(positions_seen: int, epoch_size: int) -> int:
    return positions_seen // epoch_size


def _advance_scheduler_for_epoch_boundaries(
    state: TrainState,
    *,
    positions_before_step: int,
    positions_after_step: int,
) -> None:
    if state.scheduler is None:
        return
    previous_epoch_index = _epoch_index(positions_before_step, state.config.epoch_size)
    new_epoch_index = _epoch_index(positions_after_step, state.config.epoch_size)
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


def _prepare_batch(batch, torch, *, pin_memory: bool = False) -> _PreparedBatch:
    tensors = {
        "white_indices": torch.from_numpy(batch.white_indices).to(dtype=torch.long),
        "black_indices": torch.from_numpy(batch.black_indices).to(dtype=torch.long),
        "stm": torch.from_numpy(batch.stm).to(dtype=torch.float32).unsqueeze(1),
        "score": torch.from_numpy(batch.score).to(dtype=torch.float32).unsqueeze(1),
        "result_wdl": torch.from_numpy(batch.result_wdl).to(dtype=torch.float32).unsqueeze(1),
    }
    if pin_memory:
        tensors = {name: tensor.pin_memory() for name, tensor in tensors.items()}
    return _PreparedBatch(
        batch_positions=_batch_size(batch),
        tensors=tensors,
    )


def _move_tensors_to_device(
    tensors: dict[str, object],
    device: str,
    *,
    non_blocking: bool = False,
) -> dict[str, object]:
    if device == "cpu":
        return tensors
    return {name: tensor.to(device=device, non_blocking=non_blocking) for name, tensor in tensors.items()}


def _pin_memory_enabled(state: TrainState) -> bool:
    return state.device == "cuda" and state.config.pin_memory


def _batch_size(batch) -> int:
    return int(batch.stm.shape[0])


def _binpack_filter_options(config: TrainConfig, *, split_role: str = "all") -> dict[str, object]:
    return {
        "skip_tactical_positions": config.filtered,
        "skip_wdl_score_mismatch": config.wld_filtered,
        "random_fen_skipping": config.random_fen_skipping if split_role == "train" else 0,
        "early_fen_skipping": config.early_fen_skipping,
        "soft_early_fen_skipping": config.soft_early_fen_skipping,
        "simple_eval_skipping": config.simple_eval_skipping,
        "param_index": config.param_index,
        "pc_y0": config.pc_y0,
        "pc_y1": config.pc_y1,
        "pc_y2": config.pc_y2,
        "pc_y3": config.pc_y3,
        "pc_y4": config.pc_y4,
        "ply_x1": config.ply_x1,
        "ply_y1": config.ply_y1,
        "ply_x2": config.ply_x2,
        "ply_y2": config.ply_y2,
        "ply_x3": config.ply_x3,
        "ply_y3": config.ply_y3,
        "ply_x4": config.ply_x4,
        "ply_y4": config.ply_y4,
        "split_role": split_role,
        "validation_split_fraction": 0.0,
    }


def _current_lambda(config: TrainConfig, current_epoch: int) -> float:
    start_lambda = float(config.start_lambda if config.start_lambda is not None else config.lambda_)
    end_lambda = float(config.end_lambda if config.end_lambda is not None else config.lambda_)
    progress = max(0.0, min(1.0, float(current_epoch) / float(config.max_epochs)))
    return float(start_lambda + (end_lambda - start_lambda) * progress)


def _validation_datasets(config: TrainConfig) -> list[str]:
    return config.validation_datasets or config.datasets


def _should_validate_epoch(config: TrainConfig, epoch_index: int) -> bool:
    return (
        config.validation_size > 0
        and bool(_validation_datasets(config))
        and epoch_index > 0
        and epoch_index % config.check_val_every_n_epoch == 0
    )


def _should_checkpoint_epoch(config: TrainConfig, epoch_index: int) -> bool:
    return config.network_save_period > 0 and epoch_index > 0 and epoch_index % config.network_save_period == 0


def _log_every_n_steps(config: TrainConfig) -> int:
    return max(1, (config.num_batches_per_epoch + 4) // 5)


def _epoch_checkpoint_path(run_dir: Path, epoch_index: int, global_step: int) -> Path:
    return run_dir / "checkpoints" / f"epoch_{epoch_index:04d}_step_{global_step:08d}.pt"


def _wdl_expectation_from_score(values, offset: float, scaling: float, torch):
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


def _payload_optional_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    return int(value)


def _payload_optional_float(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    return float(value)


def _payload_optional_float_tuple(payload: dict[str, object], key: str) -> tuple[float, ...] | None:
    value = payload.get(key)
    if value is None:
        return None
    return tuple(float(item) for item in value)
