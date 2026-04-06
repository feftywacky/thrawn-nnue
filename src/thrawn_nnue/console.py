from __future__ import annotations

from dataclasses import dataclass
import sys
import time


@dataclass(slots=True)
class ConsoleContext:
    run_name: str
    device: str
    train_shards: int
    validation_shards: int
    total_train_positions: int
    initial_positions_seen: int
    batch_size: int
    superbatch_positions: int
    validation_interval_positions: int
    log_every: int


class _BaseReporter:
    def startup(self, context: ConsoleContext) -> None:
        raise NotImplementedError

    def update_train(
        self,
        *,
        global_step: int,
        positions_seen: int,
        superbatch_index: int,
        loss: float,
        lr: float,
    ) -> None:
        raise NotImplementedError

    def validation_started(self, *, global_step: int, positions_seen: int) -> None:
        raise NotImplementedError

    def validation_finished(self, metrics: dict[str, object], *, is_best: bool) -> None:
        raise NotImplementedError

    def checkpoint_saved(self, checkpoint_path: str, *, is_best: bool) -> None:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class TextReporter(_BaseReporter):
    def __init__(self) -> None:
        self._started_at = time.monotonic()
        self._latest_validation_loss: float | None = None
        self._log_every = 1
        self._total_train_positions = 0

    def startup(self, context: ConsoleContext) -> None:
        self._log_every = context.log_every
        self._total_train_positions = context.total_train_positions
        print(
            "training start:"
            f" run={context.run_name}"
            f" device={context.device}"
            f" train_shards={context.train_shards}"
            f" validation_shards={context.validation_shards}"
            f" total_positions={_format_count(context.total_train_positions)}"
            f" batch_size={_format_count(context.batch_size)}"
        )
        print(
            "training stream:"
            " combined cyclic stream across all train_datasets;"
            f" superbatch_positions={_format_count(context.superbatch_positions)};"
            f" validation_interval_positions={_format_count(context.validation_interval_positions)};"
            f" validation uses non-cyclic passes across {context.validation_shards} validation shard(s)"
        )

    def update_train(
        self,
        *,
        global_step: int,
        positions_seen: int,
        superbatch_index: int,
        loss: float,
        lr: float,
    ) -> None:
        if global_step % self._log_every != 0:
            return
        elapsed = time.monotonic() - self._started_at
        pieces = [
            f"step={global_step}",
            f"positions={_format_progress(positions_seen, self._total_train_positions)}",
            f"superbatch={superbatch_index}",
            f"loss={loss:.6f}",
            f"lr={lr:.8f}",
            f"elapsed={_format_seconds(elapsed)}",
        ]
        if self._latest_validation_loss is not None:
            pieces.append(f"val={self._latest_validation_loss:.6f}")
        print("train " + " ".join(pieces))

    def validation_started(self, *, global_step: int, positions_seen: int) -> None:
        print(
            "validation start:"
            f" step={global_step}"
            f" positions={_format_count(positions_seen)}"
        )

    def validation_finished(self, metrics: dict[str, object], *, is_best: bool) -> None:
        self._latest_validation_loss = float(metrics["validation_loss"])
        suffix = " best=true" if is_best else ""
        print(
            "validation done:"
            f" step={int(metrics['global_step'])}"
            f" positions={_format_count(int(metrics['positions_seen']))}"
            f" loss={float(metrics['validation_loss']):.6f}"
            f" teacher={float(metrics['validation_teacher_loss']):.6f}"
            f" result={float(metrics['validation_result_loss']):.6f}"
            f" wdl_acc={float(metrics['wdl_accuracy']):.4f}"
            f" teacher_result_disagree={float(metrics['teacher_result_disagreement_rate']):.4f}"
            f" eval_positions={_format_count(int(metrics['validation_positions']))}"
            f" batches={int(metrics['validation_batches'])}"
            f"{suffix}"
        )

    def checkpoint_saved(self, checkpoint_path: str, *, is_best: bool) -> None:
        label = "best checkpoint" if is_best else "checkpoint"
        print(f"{label} saved: {checkpoint_path}")

    def close(self) -> None:
        return None


class ProgressReporter(_BaseReporter):
    def __init__(self) -> None:
        self._tqdm = _load_tqdm()
        self._bar = None
        self._latest_validation_loss: float | None = None

    def startup(self, context: ConsoleContext) -> None:
        self._bar = self._tqdm(
            total=context.total_train_positions,
            initial=context.initial_positions_seen,
            dynamic_ncols=True,
            file=sys.stdout,
            unit="pos",
        )
        self._bar.write(
            "training start:"
            f" run={context.run_name}"
            f" device={context.device}"
            f" train_shards={context.train_shards}"
            f" validation_shards={context.validation_shards}"
            f" total_positions={_format_count(context.total_train_positions)}"
            f" batch_size={_format_count(context.batch_size)}"
        )
        self._bar.write(
            "training stream:"
            " combined cyclic stream across all train_datasets;"
            f" superbatch_positions={_format_count(context.superbatch_positions)};"
            f" validation_interval_positions={_format_count(context.validation_interval_positions)};"
            f" validation uses non-cyclic passes across {context.validation_shards} validation shard(s)"
        )

    def update_train(
        self,
        *,
        global_step: int,
        positions_seen: int,
        superbatch_index: int,
        loss: float,
        lr: float,
    ) -> None:
        if self._bar is None:
            return
        if positions_seen > self._bar.n:
            self._bar.update(positions_seen - self._bar.n)
        postfix = {
            "step": global_step,
            "sb": superbatch_index,
            "loss": f"{loss:.4f}",
            "lr": f"{lr:.2e}",
        }
        if self._latest_validation_loss is not None:
            postfix["val"] = f"{self._latest_validation_loss:.4f}"
        self._bar.set_postfix(postfix, refresh=False)

    def validation_started(self, *, global_step: int, positions_seen: int) -> None:
        if self._bar is not None:
            self._bar.write(
                "validation start:"
                f" step={global_step}"
                f" positions={_format_count(positions_seen)}"
            )

    def validation_finished(self, metrics: dict[str, object], *, is_best: bool) -> None:
        self._latest_validation_loss = float(metrics["validation_loss"])
        if self._bar is not None:
            suffix = " best=true" if is_best else ""
            self._bar.write(
                "validation done:"
                f" step={int(metrics['global_step'])}"
                f" positions={_format_count(int(metrics['positions_seen']))}"
                f" loss={float(metrics['validation_loss']):.6f}"
                f" teacher={float(metrics['validation_teacher_loss']):.6f}"
                f" result={float(metrics['validation_result_loss']):.6f}"
                f" wdl_acc={float(metrics['wdl_accuracy']):.4f}"
                f" teacher_result_disagree={float(metrics['teacher_result_disagreement_rate']):.4f}"
                f" eval_positions={_format_count(int(metrics['validation_positions']))}"
                f" batches={int(metrics['validation_batches'])}"
                f"{suffix}"
            )

    def checkpoint_saved(self, checkpoint_path: str, *, is_best: bool) -> None:
        if self._bar is not None:
            label = "best checkpoint" if is_best else "checkpoint"
            self._bar.write(f"{label} saved: {checkpoint_path}")

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


def create_console_reporter(mode: str) -> _BaseReporter:
    if mode == "text":
        return TextReporter()
    try:
        return ProgressReporter()
    except ModuleNotFoundError:
        return TextReporter()


def _load_tqdm():
    from tqdm import tqdm  # type: ignore

    return tqdm


def _format_seconds(value: float) -> str:
    total = int(max(0, round(value)))
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def _format_count(value: int) -> str:
    return f"{int(value):,}"


def _format_progress(current: int, total: int) -> str:
    if total <= 0:
        return _format_count(current)
    fraction = (float(current) / float(total)) * 100.0
    return f"{_format_count(current)}/{_format_count(total)} ({fraction:.2f}%)"
