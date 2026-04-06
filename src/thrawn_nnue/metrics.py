from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math


@dataclass(slots=True)
class MetricsRun:
    run_dir: Path
    metrics_path: Path
    train_records: list[dict[str, object]]
    validation_records: list[dict[str, object]]
    best_validation_loss: float | None
    best_validation_positions: int | None
    best_checkpoint_exists: bool
    checkpoint_config: dict[str, object] | None
    checkpoint_global_step: int | None
    checkpoint_positions_seen: int | None


def load_metrics_run(run_dir: str | Path) -> MetricsRun:
    run_path = Path(run_dir).resolve()
    metrics_path = run_path / "metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.jsonl found under {run_path}")

    records = _load_jsonl(metrics_path)
    train_records = [record for record in records if record.get("event") == "train"]
    validation_records = [record for record in records if record.get("event") == "validation"]
    train_records.sort(key=lambda record: int(record.get("global_step", 0)))
    validation_records.sort(key=lambda record: int(record.get("global_step", 0)))

    checkpoint = _checkpoint_diagnostics(run_path)
    batch_size = _as_int(None if checkpoint["config"] is None else checkpoint["config"].get("batch_size"))
    best_validation_loss = checkpoint["best_validation_loss"]
    best_validation_positions = checkpoint["best_validation_positions"]
    if best_validation_loss is None and validation_records:
        best_record = min(validation_records, key=lambda record: float(record["validation_loss"]))
        best_validation_loss = float(best_record["validation_loss"])
        best_validation_positions = _record_positions(best_record, batch_size)

    return MetricsRun(
        run_dir=run_path,
        metrics_path=metrics_path,
        train_records=train_records,
        validation_records=validation_records,
        best_validation_loss=best_validation_loss,
        best_validation_positions=best_validation_positions,
        best_checkpoint_exists=(run_path / "checkpoints" / "best.pt").exists(),
        checkpoint_config=checkpoint["config"],
        checkpoint_global_step=checkpoint["global_step"],
        checkpoint_positions_seen=checkpoint["positions_seen"],
    )


def summarize_run(run: MetricsRun) -> dict[str, object]:
    latest_train = run.train_records[-1] if run.train_records else None
    latest_validation = run.validation_records[-1] if run.validation_records else None
    status = "validated" if run.validation_records else "train-only"
    if not run.train_records and not run.validation_records:
        status = "missing-metrics"

    batch_size = _as_int(_config_value(run, "batch_size"))
    total_train_positions = _as_int(_config_value(run, "total_train_positions"))
    superbatch_positions = _as_int(_config_value(run, "superbatch_positions"))
    configured_validation_interval = _as_int(_config_value(run, "validation_interval_positions"))
    effective_validation_interval = configured_validation_interval
    if effective_validation_interval in (None, 0):
        effective_validation_interval = superbatch_positions
    score_clip = _as_float(_config_value(run, "score_clip"))
    score_scale = _as_float(_config_value(run, "score_scale"))
    wdl_scale = _as_float(_config_value(run, "wdl_scale"))
    eval_lambda = _as_float(_config_value(run, "eval_lambda"))

    latest_train_step = _record_step(latest_train)
    latest_train_positions = _record_positions(latest_train, batch_size)
    latest_validation_step = _record_step(latest_validation)
    latest_validation_positions = _record_positions(latest_validation, batch_size)
    total_optimizer_steps = (
        None
        if total_train_positions is None or batch_size in (None, 0)
        else math.ceil(total_train_positions / batch_size)
    )
    latest_lr = _metric_value(latest_train, "lr")
    initial_lr = _metric_value(run.train_records[0] if run.train_records else None, "lr")

    train_log_interval_steps = _infer_interval(run.train_records, "global_step", batch_size=batch_size)
    validation_interval_positions = _infer_interval(
        run.validation_records,
        "positions_seen",
        batch_size=batch_size,
    )
    latest_train_at_validation = _closest_train_record(
        run.train_records,
        latest_validation_positions,
        batch_size=batch_size,
    )

    best_validation_gap = None
    positions_since_best = None
    best_is_latest_validation = None
    if latest_validation is not None and run.best_validation_loss is not None:
        best_validation_gap = float(latest_validation["validation_loss"]) - float(run.best_validation_loss)
        if run.best_validation_positions is not None and latest_validation_positions is not None:
            positions_since_best = latest_validation_positions - run.best_validation_positions
            best_is_latest_validation = positions_since_best == 0

    train_validation_gap = _gap(
        _metric_value(latest_train_at_validation, "loss"),
        _metric_value(latest_validation, "validation_loss"),
    )
    teacher_gap = _gap(
        _metric_value(latest_train_at_validation, "teacher_loss", "eval_loss"),
        _metric_value(latest_validation, "validation_teacher_loss", "validation_eval_loss"),
    )
    result_gap = _gap(
        _metric_value(latest_train_at_validation, "result_loss"),
        _metric_value(latest_validation, "validation_result_loss"),
    )

    train_teacher_to_result_ratio = _ratio(
        _metric_value(latest_train, "teacher_loss", "eval_loss"),
        _metric_value(latest_train, "result_loss"),
    )
    validation_teacher_to_result_ratio = _ratio(
        _metric_value(latest_validation, "validation_teacher_loss", "validation_eval_loss"),
        _metric_value(latest_validation, "validation_result_loss"),
    )
    teacher_signal_collapsed = False
    ratios = [value for value in (train_teacher_to_result_ratio, validation_teacher_to_result_ratio) if value is not None]
    if ratios and min(ratios) < 0.02:
        teacher_signal_collapsed = True

    latest_position_fraction = None
    if total_train_positions is not None and total_train_positions > 0 and latest_train_positions is not None:
        latest_position_fraction = latest_train_positions / total_train_positions

    latest_lr_fraction_of_initial = None
    if latest_lr is not None and initial_lr is not None and initial_lr > 0.0:
        latest_lr_fraction_of_initial = latest_lr / initial_lr
    lr_near_zero = latest_lr is not None and latest_lr <= 1e-8
    scheduler_exhausted = bool(
        lr_near_zero
        and latest_position_fraction is not None
        and latest_position_fraction >= 0.98
    )

    resume_recommendation = _resume_recommendation(
        validation_records=run.validation_records,
        latest_validation_loss=_metric_value(latest_validation, "validation_loss"),
        best_validation_loss=run.best_validation_loss,
        latest_validation_positions=latest_validation_positions,
        best_validation_positions=run.best_validation_positions,
    )

    summary = {
        "run_dir": str(run.run_dir),
        "status": status,
        "train_records": len(run.train_records),
        "validation_records": len(run.validation_records),
        "latest_train_step": latest_train_step,
        "positions_seen": latest_train_positions,
        "latest_train_loss": _metric_value(latest_train, "loss"),
        "latest_train_teacher_loss": _metric_value(latest_train, "teacher_loss", "eval_loss"),
        "latest_train_result_loss": _metric_value(latest_train, "result_loss"),
        "latest_lr": latest_lr,
        "latest_validation_step": latest_validation_step,
        "latest_validation_positions": latest_validation_positions,
        "latest_validation_loss": _metric_value(latest_validation, "validation_loss"),
        "latest_validation_teacher_loss": _metric_value(
            latest_validation,
            "validation_teacher_loss",
            "validation_eval_loss",
        ),
        "latest_validation_result_loss": _metric_value(latest_validation, "validation_result_loss"),
        "latest_validation_wdl_accuracy": _metric_value(latest_validation, "wdl_accuracy"),
        "latest_validation_teacher_result_disagreement_rate": _metric_value(
            latest_validation,
            "teacher_result_disagreement_rate",
        ),
        "latest_validation_evaluated_positions": _metric_value(latest_validation, "validation_positions"),
        "best_validation_loss": run.best_validation_loss,
        "best_validation_positions": run.best_validation_positions,
        "best_checkpoint_exists": run.best_checkpoint_exists,
        "configured_total_positions": total_train_positions,
        "configured_total_steps": total_optimizer_steps,
        "latest_position_fraction": latest_position_fraction,
        "batch_size": batch_size,
        "superbatch_positions": superbatch_positions,
        "configured_validation_interval_positions": configured_validation_interval,
        "effective_validation_interval_positions": effective_validation_interval,
        "train_log_interval_steps": train_log_interval_steps,
        "validation_interval_positions": validation_interval_positions,
        "best_validation_gap": best_validation_gap,
        "positions_since_best": positions_since_best,
        "best_is_latest_validation": best_is_latest_validation,
        "resume_recommendation": resume_recommendation,
        "train_validation_gap": train_validation_gap,
        "teacher_gap": teacher_gap,
        "result_gap": result_gap,
        "train_teacher_to_result_ratio": train_teacher_to_result_ratio,
        "validation_teacher_to_result_ratio": validation_teacher_to_result_ratio,
        "teacher_signal_collapsed": teacher_signal_collapsed,
        "latest_lr_fraction_of_initial": latest_lr_fraction_of_initial,
        "lr_near_zero": lr_near_zero,
        "scheduler_exhausted": scheduler_exhausted,
        "score_clip": score_clip,
        "score_scale": score_scale,
        "wdl_scale": wdl_scale,
        "eval_lambda": eval_lambda,
    }
    summary["suggestions"] = _build_suggestions(summary)
    return summary


def render_summary_text(summary: dict[str, object]) -> str:
    lines = [
        f"run_dir: {summary['run_dir']}",
        f"status: {summary['status']}",
        f"train_records: {summary['train_records']}",
        f"validation_records: {summary['validation_records']}",
    ]
    if summary["latest_train_step"] is not None:
        lines.extend(
            [
                f"latest_train_step: {summary['latest_train_step']}",
                f"positions_seen: {_format_optional_int(summary['positions_seen'])}",
                f"latest_train_loss: {_format_optional_float(summary['latest_train_loss'])}",
                f"latest_train_teacher_loss: {_format_optional_float(summary['latest_train_teacher_loss'])}",
                f"latest_train_result_loss: {_format_optional_float(summary['latest_train_result_loss'])}",
                f"latest_lr: {_format_optional_float(summary['latest_lr'], precision=8)}",
            ]
        )
    if summary["latest_validation_step"] is not None:
        lines.extend(
            [
                f"latest_validation_step: {summary['latest_validation_step']}",
                f"latest_validation_positions: {_format_optional_int(summary['latest_validation_positions'])}",
                f"latest_validation_loss: {_format_optional_float(summary['latest_validation_loss'])}",
                f"latest_validation_teacher_loss: {_format_optional_float(summary['latest_validation_teacher_loss'])}",
                f"latest_validation_result_loss: {_format_optional_float(summary['latest_validation_result_loss'])}",
                f"latest_validation_wdl_accuracy: {_format_optional_float(summary['latest_validation_wdl_accuracy'])}",
                (
                    "latest_validation_teacher_result_disagreement_rate: "
                    f"{_format_optional_float(summary['latest_validation_teacher_result_disagreement_rate'])}"
                ),
                f"latest_validation_evaluated_positions: {_format_optional_int(summary['latest_validation_evaluated_positions'])}",
            ]
        )
    else:
        lines.append("latest_validation_step: none")

    if summary["best_validation_loss"] is not None:
        lines.extend(
            [
                f"best_validation_positions: {_format_optional_int(summary['best_validation_positions'])}",
                f"best_validation_loss: {_format_optional_float(summary['best_validation_loss'])}",
            ]
        )
    else:
        lines.append("best_validation_positions: none")

    lines.append(f"best_checkpoint_exists: {summary['best_checkpoint_exists']}")
    lines.append("")
    lines.append("Run Budget")
    lines.append(f"configured_total_positions: {_format_optional_int(summary['configured_total_positions'])}")
    lines.append(f"configured_total_steps: {_format_optional_int(summary['configured_total_steps'])}")
    lines.append(f"latest_position_fraction: {_format_optional_fraction(summary['latest_position_fraction'])}")
    lines.append(f"batch_size: {_format_optional_int(summary['batch_size'])}")
    lines.append(f"superbatch_positions: {_format_optional_int(summary['superbatch_positions'])}")
    lines.append(
        f"effective_validation_interval_positions: {_format_optional_int(summary['effective_validation_interval_positions'])}"
    )
    lines.append(f"train_log_interval_steps: {_format_optional_int(summary['train_log_interval_steps'])}")
    lines.append(f"validation_interval_positions: {_format_optional_int(summary['validation_interval_positions'])}")
    lines.append(f"latest_lr_fraction_of_initial: {_format_optional_float(summary['latest_lr_fraction_of_initial'])}")
    lines.append(f"lr_near_zero: {summary['lr_near_zero']}")
    lines.append(f"scheduler_exhausted: {summary['scheduler_exhausted']}")
    lines.append("")
    lines.append("Generalization")
    lines.append(f"best_validation_gap: {_format_optional_float(summary['best_validation_gap'])}")
    lines.append(f"positions_since_best: {_format_optional_int(summary['positions_since_best'])}")
    lines.append(f"best_is_latest_validation: {summary['best_is_latest_validation']}")
    lines.append(f"resume_recommendation: {summary['resume_recommendation']}")
    lines.append(f"train_validation_gap: {_format_optional_float(summary['train_validation_gap'])}")
    lines.append(f"teacher_gap: {_format_optional_float(summary['teacher_gap'])}")
    lines.append(f"result_gap: {_format_optional_float(summary['result_gap'])}")
    lines.append(f"train_teacher_to_result_ratio: {_format_optional_float(summary['train_teacher_to_result_ratio'])}")
    lines.append(
        f"validation_teacher_to_result_ratio: {_format_optional_float(summary['validation_teacher_to_result_ratio'])}"
    )
    lines.append(f"teacher_signal_collapsed: {summary['teacher_signal_collapsed']}")
    lines.append("")
    lines.append("Suggestions")
    for suggestion in summary["suggestions"]:
        lines.append(f"- {suggestion}")
    return "\n".join(lines)


def generate_run_plots(run: MetricsRun) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for thrawn-nnue metrics plotting") from exc

    plots_dir = run.run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    batch_size = _as_int(_config_value(run, "batch_size"))

    if run.train_records:
        outputs.append(
            _plot_losses(
                plt,
                plots_dir / "train_loss.png",
                run.train_records,
                "positions_seen",
                [
                    (("loss",), "blended loss"),
                    (("teacher_loss", "eval_loss"), "teacher target"),
                    (("result_loss",), "game result"),
                ],
                "Train Loss",
                batch_size=batch_size,
                smooth_primary=True,
                y_label="Loss",
                footer_note="blended = weighted teacher-target + game-result loss",
            )
        )
        outputs.append(
            _plot_losses(
                plt,
                plots_dir / "lr.png",
                run.train_records,
                "positions_seen",
                [(("lr",), "lr")],
                "Learning Rate",
                batch_size=batch_size,
                smooth_primary=False,
                y_label="Learning Rate",
            )
        )

    if run.validation_records:
        outputs.append(
            _plot_losses(
                plt,
                plots_dir / "validation_loss.png",
                run.validation_records,
                "positions_seen",
                [
                    (("validation_loss",), "blended loss"),
                    (("validation_teacher_loss", "validation_eval_loss"), "teacher target"),
                    (("validation_result_loss",), "game result"),
                ],
                "Validation Loss",
                batch_size=batch_size,
                smooth_primary=False,
                y_label="Loss",
                footer_note="blended = weighted teacher-target + game-result loss",
            )
        )

    if run.train_records and run.validation_records:
        outputs.append(_plot_overview(plt, plots_dir / "loss_overview.png", run, batch_size=batch_size))

    return outputs


def _plot_losses(
    plt,
    output_path: Path,
    records: list[dict[str, object]],
    x_key: str,
    series: list[tuple[tuple[str, ...], str]],
    title: str,
    *,
    batch_size: int | None,
    smooth_primary: bool,
    y_label: str,
    footer_note: str | None = None,
) -> Path:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    x_values = [_record_axis(record, x_key, batch_size=batch_size) for record in records]
    primary_values: list[float] | None = None
    primary_smoothed: list[float] | None = None
    for index, (fields, label) in enumerate(series):
        y_values = [_required_metric(record, fields) for record in records]
        if index == 0 and smooth_primary and len(y_values) >= 8:
            primary_values = y_values
            primary_smoothed = _moving_average(y_values, window=_smoothing_window(len(y_values)))
            axis.plot(x_values, y_values, label=f"{label} (raw)", alpha=0.2, linewidth=1.0)
            axis.plot(x_values, primary_smoothed, label=f"{label} (smoothed)", linewidth=2.2)
            continue
        axis.plot(x_values, y_values, label=label, linewidth=2.0 if index == 0 else 1.8)
    axis.set_title(title)
    axis.set_xlabel("Positions Seen")
    axis.set_ylabel(y_label)
    axis.grid(True, alpha=0.3)
    axis.legend()
    if primary_values is not None and primary_smoothed is not None:
        _set_focus_ylim(axis, primary_values, primary_smoothed)
    if footer_note is not None:
        axis.text(
            0.01,
            0.01,
            footer_note,
            transform=axis.transAxes,
            fontsize=9,
            alpha=0.75,
            va="bottom",
        )
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def _plot_overview(plt, output_path: Path, run: MetricsRun, *, batch_size: int | None) -> Path:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    train_positions = [_record_axis(record, "positions_seen", batch_size=batch_size) for record in run.train_records]
    train_loss = [_required_metric(record, ("loss",)) for record in run.train_records]
    validation_positions = [
        _record_axis(record, "positions_seen", batch_size=batch_size)
        for record in run.validation_records
    ]
    validation_loss = [_required_metric(record, ("validation_loss",)) for record in run.validation_records]
    smoothed_train = _moving_average(train_loss, window=_smoothing_window(len(train_loss)))

    axis.plot(
        train_positions,
        train_loss,
        label="train blended (raw)",
        alpha=0.12,
        linewidth=0.9,
    )
    axis.plot(
        train_positions,
        smoothed_train,
        label="train blended (smoothed)",
        linewidth=2.2,
    )
    axis.plot(
        validation_positions,
        validation_loss,
        label="validation blended",
        marker="o",
        markersize=3.5,
        linewidth=2.0,
    )
    axis.set_title("Loss Overview")
    axis.set_xlabel("Positions Seen")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    _set_focus_ylim(axis, train_loss + validation_loss, smoothed_train + validation_loss)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def _moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1 or len(values) <= 2:
        return values[:]
    smoothed: list[float] = []
    running_total = 0.0
    for index, value in enumerate(values):
        running_total += value
        if index >= window:
            running_total -= values[index - window]
        current_window = min(index + 1, window)
        smoothed.append(running_total / current_window)
    return smoothed


def _smoothing_window(length: int) -> int:
    return max(5, min(401, length // 50))


def _set_focus_ylim(axis, raw_values: list[float], trend_values: list[float]) -> None:
    if not raw_values or not trend_values:
        return
    trend_min = min(trend_values)
    trend_max = max(trend_values)
    spread = max(trend_max - trend_min, 1e-9)
    lower = min(raw_values)
    upper = max(raw_values)
    padded_lower = max(lower, trend_min - spread * 0.35)
    padded_upper = min(upper, trend_max + spread * 0.35)
    if padded_upper <= padded_lower:
        padded_lower = lower
        padded_upper = upper
    axis.set_ylim(padded_lower, padded_upper)


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def _checkpoint_diagnostics(run_dir: Path) -> dict[str, object]:
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        return {
            "best_validation_loss": None,
            "best_validation_positions": None,
            "config": None,
            "global_step": None,
            "positions_seen": None,
        }
    try:
        from .checkpoint import load_checkpoint

        payload = load_checkpoint(checkpoint_path, map_location="cpu")
    except Exception:
        return {
            "best_validation_loss": None,
            "best_validation_positions": None,
            "config": None,
            "global_step": None,
            "positions_seen": None,
        }

    config = payload.get("config")
    batch_size = _as_int(None if config is None else config.get("batch_size"))
    best_validation_positions = payload.get("best_validation_positions")
    if best_validation_positions is None and payload.get("best_validation_step") is not None and batch_size is not None:
        best_validation_positions = int(payload["best_validation_step"]) * batch_size

    positions_seen = payload.get("positions_seen")
    if positions_seen is None and payload.get("global_step") is not None and batch_size is not None:
        positions_seen = int(payload["global_step"]) * batch_size

    return {
        "best_validation_loss": payload.get("best_validation_loss"),
        "best_validation_positions": best_validation_positions,
        "config": config,
        "global_step": payload.get("global_step"),
        "positions_seen": positions_seen,
    }


def _record_step(record: dict[str, object] | None) -> int | None:
    return None if record is None else int(record["global_step"])


def _record_positions(record: dict[str, object] | None, batch_size: int | None) -> int | None:
    if record is None:
        return None
    if "positions_seen" in record:
        return int(record["positions_seen"])
    if batch_size is not None and "global_step" in record:
        return int(record["global_step"]) * batch_size
    return None


def _config_value(run: MetricsRun, key: str) -> object | None:
    if run.checkpoint_config is None:
        return None
    return run.checkpoint_config.get(key)


def _closest_train_record(
    train_records: list[dict[str, object]],
    positions_seen: int | None,
    *,
    batch_size: int | None,
) -> dict[str, object] | None:
    if positions_seen is None or not train_records:
        return None
    eligible = [
        record
        for record in train_records
        if (_record_positions(record, batch_size) or 0) <= positions_seen
    ]
    if eligible:
        return eligible[-1]
    return train_records[-1]


def _infer_interval(
    records: list[dict[str, object]],
    key: str,
    *,
    batch_size: int | None,
) -> int | None:
    if len(records) < 2:
        return None
    deltas = [
        _record_axis(records[index], key, batch_size=batch_size)
        - _record_axis(records[index - 1], key, batch_size=batch_size)
        for index in range(1, len(records))
    ]
    positive = [delta for delta in deltas if delta > 0]
    if not positive:
        return None
    return positive[-1]


def _record_axis(record: dict[str, object], key: str, *, batch_size: int | None) -> int:
    if key == "positions_seen":
        value = _record_positions(record, batch_size)
        if value is None:
            raise KeyError("positions_seen is unavailable and cannot be derived")
        return value
    return int(record[key])


def _metric_value(record: dict[str, object] | None, *keys: str) -> float | None:
    if record is None:
        return None
    for key in keys:
        if key in record:
            return float(record[key])
    return None


def _required_metric(record: dict[str, object], keys: tuple[str, ...]) -> float:
    value = _metric_value(record, *keys)
    if value is None:
        raise KeyError(f"Missing metric fields: {', '.join(keys)}")
    return value


def _ratio(numerator: object | None, denominator: object | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    denominator_value = float(denominator)
    if denominator_value == 0.0:
        return None
    return float(numerator) / denominator_value


def _gap(train_value: object | None, validation_value: object | None) -> float | None:
    if train_value is None or validation_value is None:
        return None
    return float(validation_value) - float(train_value)


def _resume_recommendation(
    *,
    validation_records: list[dict[str, object]],
    latest_validation_loss: float | None,
    best_validation_loss: float | None,
    latest_validation_positions: int | None,
    best_validation_positions: int | None,
) -> str:
    if len(validation_records) < 2 or latest_validation_loss is None or best_validation_loss is None:
        return "insufficient-validation"
    if latest_validation_positions is None or best_validation_positions is None:
        return "insufficient-validation"

    gap = latest_validation_loss - best_validation_loss
    positions_since_best = latest_validation_positions - best_validation_positions
    if positions_since_best <= 0 or gap <= 5e-4:
        return "continue-latest"
    return "export-best"


def _build_suggestions(summary: dict[str, object]) -> list[str]:
    suggestions: list[str] = []
    if summary["resume_recommendation"] == "continue-latest":
        suggestions.append(
            "Validation is still near its best point; resume from the latest checkpoint and extend total_train_positions."
        )
    elif summary["resume_recommendation"] == "export-best":
        suggestions.append("Best validation is materially earlier than the end; export best.pt and start the next experiment from there.")
    else:
        suggestions.append("There are too few validation points to judge continuation confidently yet.")

    if summary["teacher_signal_collapsed"]:
        suggestions.append("Teacher loss is tiny relative to result loss; revisit score normalization or eval_lambda.")
    if summary["scheduler_exhausted"]:
        suggestions.append("Learning rate is effectively exhausted; if validation is still improving, continue by increasing the position budget.")
    if summary["positions_seen"] is not None and summary["positions_seen"] < 500_000_000:
        suggestions.append("The position budget is still modest for very large binpack corpora; the run may simply need more data exposure.")
    if not suggestions:
        suggestions.append("No obvious red flags; compare this run against engine testing and nearby hyperparameter variants.")
    return suggestions


def _as_int(value: object | None) -> int | None:
    return None if value is None else int(value)


def _as_float(value: object | None) -> float | None:
    return None if value is None else float(value)


def _format_optional_int(value: object | None) -> str:
    if value is None:
        return "none"
    return str(int(value))


def _format_optional_float(value: object | None, *, precision: int = 6) -> str:
    if value is None:
        return "none"
    return f"{float(value):.{precision}f}"


def _format_optional_fraction(value: object | None) -> str:
    if value is None:
        return "none"
    return f"{float(value) * 100.0:.2f}%"
