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
    superbatch_index: int = 0


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
    state.best_validation_positions = _payload_best_validation_positions(payload, config)
    state.global_step = int(payload["global_step"])
    state.positions_seen = _payload_positions_seen(payload, config)
    state.superbatch_index = _payload_superbatch_index(payload, config)
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
        output_buckets=config.output_buckets,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
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
            superbatch_positions=state.config.superbatch_positions,
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
                    loss, teacher_loss, result_loss = _run_train_step(
                        state,
                        prepared_batch.tensors,
                        torch,
                        autocast_enabled=autocast_enabled,
                    )
                    if state.scheduler is not None:
                        state.scheduler.step()

                    state.global_step += 1
                    state.positions_seen += batch_positions
                    state.superbatch_index = _superbatch_index(state.positions_seen, state.config.superbatch_positions)
                    current_loss = float(loss.detach().cpu().item())
                    current_lr = float(state.optimizer.param_groups[0]["lr"])
                    step_seconds = max(0.0, time.monotonic() - step_started_at)
                    train_positions_per_second = (
                        0.0 if step_seconds <= 0.0 else float(batch_positions) / step_seconds
                    )
                    reporter.update_train(
                        global_step=state.global_step,
                        positions_seen=state.positions_seen,
                        superbatch_index=state.superbatch_index,
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
                                "superbatch_index": state.superbatch_index,
                                "batch_positions": batch_positions,
                                "loss": current_loss,
                                "teacher_loss": float(teacher_loss.detach().cpu().item()),
                                "result_loss": float(result_loss.detach().cpu().item()),
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
        superbatch_index=state.superbatch_index,
        best_validation_loss=state.best_validation_loss,
        best_validation_positions=state.best_validation_positions,
    )


def _resolve_runtime_config(config: TrainConfig) -> None:
    config.validate()


def _create_scheduler(config: TrainConfig, optimizer, torch):
    total_steps = _total_optimizer_steps(config)
    milestones = sorted(
        {
            milestone
            for milestone in (
                int(math.floor(total_steps * float(fraction)))
                for fraction in config.lr_drop_fractions
            )
            if 0 < milestone < total_steps
        }
    )
    return torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=config.lr_drop_factor,
    )


def _run_train_step(state: TrainState, tensors, torch, *, autocast_enabled: bool):
    tensors = _move_tensors_to_device(tensors, state.device)
    score_cp_stm, result_wdl_stm = _stm_oriented_targets(
        tensors["score_cp"],
        tensors["result_wdl"],
        tensors["stm"],
        torch,
    )
    bucket_indices = _output_bucket_indices(tensors["white_counts"], state.config.output_buckets, torch)
    state.model.train()
    state.optimizer.zero_grad(set_to_none=True)
    with _autocast_context(torch, state.device, autocast_enabled):
        prediction_cp = state.model(
            tensors["white_indices"],
            tensors["black_indices"],
            tensors["stm"],
            bucket_indices,
        )
        normalized_scores = _normalize_teacher_scores(score_cp_stm, state.config, torch)
        loss, teacher_loss, result_loss = _blended_loss(
            prediction_cp,
            normalized_scores,
            result_wdl_stm,
            state.config.wdl_scale,
            state.config.eval_lambda,
            torch,
        )

    state.scaler.scale(loss).backward()
    state.scaler.unscale_(state.optimizer)
    torch.nn.utils.clip_grad_norm_(state.model.parameters(), state.config.clip_grad_norm)
    state.scaler.step(state.optimizer)
    state.scaler.update()
    return loss, teacher_loss, result_loss


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


def _blended_loss(prediction_cp, target_cp, result_wdl, wdl_scale: float, eval_lambda: float, torch):
    pred_wdl = _wdl_from_cp(prediction_cp, wdl_scale, torch)
    target_wdl = _wdl_from_cp(target_cp, wdl_scale, torch)
    teacher_loss = torch.mean((pred_wdl - target_wdl) ** 2)
    result_loss = torch.mean((pred_wdl - result_wdl) ** 2)
    loss = eval_lambda * teacher_loss + (1.0 - eval_lambda) * result_loss
    return loss, teacher_loss, result_loss


def _normalize_teacher_scores(scores, config: TrainConfig, torch):
    normalized = scores
    if config.score_clip > 0.0:
        normalized = torch.clamp(normalized, -config.score_clip, config.score_clip)
    if config.score_scale != 1.0:
        normalized = normalized / config.score_scale
    return normalized


def _stm_oriented_targets(score_cp, result_wdl, stm, torch):
    stm_white = stm.ge(0.5)
    # Binpack teacher scores are already side-to-move oriented.
    score_cp_stm = score_cp
    result_wdl_stm = torch.where(stm_white, result_wdl, 1.0 - result_wdl)
    return score_cp_stm, result_wdl_stm


def _run_validation(state: TrainState) -> dict[str, object]:
    torch = _require_torch()
    autocast_enabled = _amp_enabled(torch, state.config, state.device)
    started_at = time.monotonic()

    total_loss = 0.0
    total_teacher_loss = 0.0
    total_result_loss = 0.0
    total_positions = 0
    total_correct = 0
    total_teacher_result_disagreement = 0
    batches = 0

    was_training = state.model.training
    state.model.eval()
    with BinpackStream(
        state.config.validation_datasets,
        num_threads=state.config.num_loader_threads,
        cyclic=False,
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
                    bucket_indices = _output_bucket_indices(
                        tensors["white_counts"],
                        state.config.output_buckets,
                        torch,
                    )
                    with _autocast_context(torch, state.device, autocast_enabled):
                        prediction_cp = state.model(
                            tensors["white_indices"],
                            tensors["black_indices"],
                            tensors["stm"],
                            bucket_indices,
                        )
                        normalized_scores = _normalize_teacher_scores(score_cp_stm, state.config, torch)
                        loss, teacher_loss, result_loss = _blended_loss(
                            prediction_cp,
                            normalized_scores,
                            result_wdl_stm,
                            state.config.wdl_scale,
                            state.config.eval_lambda,
                            torch,
                        )

                    pred_wdl = _wdl_from_cp(prediction_cp, state.config.wdl_scale, torch)
                    target_wdl = _wdl_from_cp(normalized_scores, state.config.wdl_scale, torch)
                    result_bucket = _wdl_bucket(result_wdl_stm)
                    pred_bucket = _wdl_bucket(pred_wdl)
                    target_bucket = _wdl_bucket(target_wdl)

                    total_loss += float(loss.detach().cpu().item()) * batch_positions
                    total_teacher_loss += float(teacher_loss.detach().cpu().item()) * batch_positions
                    total_result_loss += float(result_loss.detach().cpu().item()) * batch_positions
                    total_correct += int((pred_bucket == result_bucket).sum().detach().cpu().item())
                    total_teacher_result_disagreement += int(
                        (target_bucket != result_bucket).sum().detach().cpu().item()
                    )
                    total_positions += batch_positions
                    batches += 1

    if was_training:
        state.model.train()

    validation_seconds = max(0.0, time.monotonic() - started_at)
    average_loss = math.inf if total_positions == 0 else total_loss / total_positions
    average_teacher_loss = math.inf if total_positions == 0 else total_teacher_loss / total_positions
    average_result_loss = math.inf if total_positions == 0 else total_result_loss / total_positions
    wdl_accuracy = 0.0 if total_positions == 0 else total_correct / total_positions
    teacher_result_disagreement_rate = (
        0.0 if total_positions == 0 else total_teacher_result_disagreement / total_positions
    )
    validation_positions_per_second = (
        0.0 if total_positions == 0 or validation_seconds <= 0.0 else float(total_positions) / validation_seconds
    )

    return {
        "event": "validation",
        "global_step": state.global_step,
        "positions_seen": state.positions_seen,
        "validation_loss": average_loss,
        "validation_teacher_loss": average_teacher_loss,
        "validation_result_loss": average_result_loss,
        "wdl_accuracy": wdl_accuracy,
        "teacher_result_disagreement_rate": teacher_result_disagreement_rate,
        "validation_positions": total_positions,
        "validation_batches": batches,
        "validation_seconds": validation_seconds,
        "validation_positions_per_second": validation_positions_per_second,
    }


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


def _output_bucket_indices(piece_counts, output_buckets: int, torch):
    if output_buckets <= 1:
        return None
    clamped_counts = piece_counts.clamp(min=2, max=32).to(dtype=torch.long)
    phase_progress = 32 - clamped_counts
    return torch.clamp((phase_progress * output_buckets) // 31, max=output_buckets - 1)


def _total_optimizer_steps(config: TrainConfig) -> int:
    return max(1, math.ceil(config.total_train_positions / config.batch_size))


def _effective_validation_interval_positions(config: TrainConfig) -> int:
    if config.validation_interval_positions > 0:
        return config.validation_interval_positions
    return config.superbatch_positions


def _next_validation_positions(current_positions: int, interval_positions: int) -> int:
    return ((current_positions // interval_positions) + 1) * interval_positions


def _superbatch_index(positions_seen: int, superbatch_positions: int) -> int:
    return positions_seen // superbatch_positions


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
            "white_counts": torch.from_numpy(batch.white_counts).to(dtype=torch.int32),
            "black_counts": torch.from_numpy(batch.black_counts).to(dtype=torch.int32),
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


def _wdl_from_cp(values, wdl_scale: float, torch):
    return torch.sigmoid(values / wdl_scale)


def _wdl_bucket(values):
    return values.mul(3.0).clamp_min(0.0).clamp_max(2.999999).to(dtype=values.dtype).to(dtype=values.long().dtype)


def _payload_positions_seen(payload: dict[str, object], config: TrainConfig) -> int:
    if "positions_seen" in payload:
        return int(payload["positions_seen"])
    return int(payload["global_step"]) * config.batch_size


def _payload_superbatch_index(payload: dict[str, object], config: TrainConfig) -> int:
    if "superbatch_index" in payload:
        return int(payload["superbatch_index"])
    return _superbatch_index(_payload_positions_seen(payload, config), config.superbatch_positions)


def _payload_best_validation_positions(payload: dict[str, object], config: TrainConfig) -> int | None:
    if payload.get("best_validation_positions") is not None:
        return int(payload["best_validation_positions"])
    if payload.get("best_validation_step") is not None:
        return int(payload["best_validation_step"]) * config.batch_size
    return None
