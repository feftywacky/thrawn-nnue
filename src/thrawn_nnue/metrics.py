from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(slots=True)
class MetricsRun:
    run_dir: Path
    metrics_path: Path
    train_records: list[dict[str, object]]
    validation_records: list[dict[str, object]]
    best_validation_loss: float | None
    best_validation_step: int | None
    best_checkpoint_exists: bool
    checkpoint_config: dict[str, object] | None
    checkpoint_global_step: int | None


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
    best_validation_loss = checkpoint["best_validation_loss"]
    best_validation_step = checkpoint["best_validation_step"]
    if best_validation_loss is None and validation_records:
        best_record = min(validation_records, key=lambda record: float(record["validation_loss"]))
        best_validation_loss = float(best_record["validation_loss"])
        best_validation_step = int(best_record["global_step"])

    return MetricsRun(
        run_dir=run_path,
        metrics_path=metrics_path,
        train_records=train_records,
        validation_records=validation_records,
        best_validation_loss=best_validation_loss,
        best_validation_step=best_validation_step,
        best_checkpoint_exists=(run_path / "checkpoints" / "best.pt").exists(),
        checkpoint_config=checkpoint["config"],
        checkpoint_global_step=checkpoint["global_step"],
    )


def summarize_run(run: MetricsRun) -> dict[str, object]:
    latest_train = run.train_records[-1] if run.train_records else None
    latest_validation = run.validation_records[-1] if run.validation_records else None
    status = "validated" if run.validation_records else "train-only"
    if not run.train_records and not run.validation_records:
        status = "missing-metrics"

    batch_size = _as_int(_config_value(run, "batch_size"))
    max_epochs = _as_int(_config_value(run, "max_epochs"))
    steps_per_epoch = _as_int(_config_value(run, "steps_per_epoch"))
    score_clip = _as_float(_config_value(run, "score_clip"))
    score_scale = _as_float(_config_value(run, "score_scale"))
    wdl_scale = _as_float(_config_value(run, "wdl_scale"))
    result_lambda = _as_float(_config_value(run, "result_lambda"))

    latest_train_step = _record_step(latest_train)
    latest_validation_step = _record_step(latest_validation)
    configured_total_steps = None if max_epochs is None or steps_per_epoch is None else max_epochs * steps_per_epoch
    latest_lr = None if latest_train is None else float(latest_train["lr"])
    initial_lr = None if not run.train_records else float(run.train_records[0]["lr"])

    train_log_interval = _infer_interval(run.train_records)
    validation_interval = _infer_interval(run.validation_records)
    latest_train_at_validation = _closest_train_record(run.train_records, latest_validation_step)

    best_validation_gap = None
    steps_since_best = None
    best_is_latest_validation = None
    if latest_validation is not None and run.best_validation_loss is not None:
        best_validation_gap = float(latest_validation["validation_loss"]) - float(run.best_validation_loss)
        if run.best_validation_step is not None:
            steps_since_best = int(latest_validation["global_step"]) - int(run.best_validation_step)
            best_is_latest_validation = steps_since_best == 0

    train_validation_gap = _gap(
        None if latest_train_at_validation is None else latest_train_at_validation.get("loss"),
        None if latest_validation is None else latest_validation.get("validation_loss"),
    )
    eval_gap = _gap(
        None if latest_train_at_validation is None else latest_train_at_validation.get("eval_loss"),
        None if latest_validation is None else latest_validation.get("validation_eval_loss"),
    )
    result_gap = _gap(
        None if latest_train_at_validation is None else latest_train_at_validation.get("result_loss"),
        None if latest_validation is None else latest_validation.get("validation_result_loss"),
    )

    train_eval_to_result_ratio = _ratio(
        None if latest_train is None else latest_train.get("eval_loss"),
        None if latest_train is None else latest_train.get("result_loss"),
    )
    validation_eval_to_result_ratio = _ratio(
        None if latest_validation is None else latest_validation.get("validation_eval_loss"),
        None if latest_validation is None else latest_validation.get("validation_result_loss"),
    )
    eval_signal_collapsed = False
    ratios = [value for value in (train_eval_to_result_ratio, validation_eval_to_result_ratio) if value is not None]
    if ratios and min(ratios) < 0.02:
        eval_signal_collapsed = True

    latest_train_step_fraction = None
    if configured_total_steps is not None and configured_total_steps > 0 and latest_train_step is not None:
        latest_train_step_fraction = latest_train_step / configured_total_steps

    samples_seen = None if latest_train_step is None or batch_size is None else latest_train_step * batch_size
    latest_lr_fraction_of_initial = None
    if latest_lr is not None and initial_lr is not None and initial_lr > 0.0:
        latest_lr_fraction_of_initial = latest_lr / initial_lr
    lr_near_zero = latest_lr is not None and latest_lr <= 1e-8
    scheduler_exhausted = bool(
        lr_near_zero
        and latest_train_step_fraction is not None
        and latest_train_step_fraction >= 0.98
    )

    resume_recommendation = _resume_recommendation(
        validation_records=run.validation_records,
        latest_validation_loss=None if latest_validation is None else float(latest_validation["validation_loss"]),
        best_validation_loss=run.best_validation_loss,
        latest_validation_step=latest_validation_step,
        best_validation_step=run.best_validation_step,
    )

    summary = {
        "run_dir": str(run.run_dir),
        "status": status,
        "train_records": len(run.train_records),
        "validation_records": len(run.validation_records),
        "latest_train_step": latest_train_step,
        "latest_train_loss": None if latest_train is None else float(latest_train["loss"]),
        "latest_train_eval_loss": None if latest_train is None else float(latest_train["eval_loss"]),
        "latest_train_result_loss": None if latest_train is None else float(latest_train["result_loss"]),
        "latest_lr": latest_lr,
        "latest_validation_step": latest_validation_step,
        "latest_validation_loss": None if latest_validation is None else float(latest_validation["validation_loss"]),
        "latest_validation_eval_loss": None if latest_validation is None else float(latest_validation["validation_eval_loss"]),
        "latest_validation_result_loss": None if latest_validation is None else float(latest_validation["validation_result_loss"]),
        "best_validation_loss": run.best_validation_loss,
        "best_validation_step": run.best_validation_step,
        "best_checkpoint_exists": run.best_checkpoint_exists,
        "configured_total_steps": configured_total_steps,
        "latest_train_step_fraction": latest_train_step_fraction,
        "batch_size": batch_size,
        "samples_seen": samples_seen,
        "train_log_interval": train_log_interval,
        "validation_interval": validation_interval,
        "best_validation_gap": best_validation_gap,
        "steps_since_best": steps_since_best,
        "best_is_latest_validation": best_is_latest_validation,
        "resume_recommendation": resume_recommendation,
        "train_validation_gap": train_validation_gap,
        "eval_gap": eval_gap,
        "result_gap": result_gap,
        "train_eval_to_result_ratio": train_eval_to_result_ratio,
        "validation_eval_to_result_ratio": validation_eval_to_result_ratio,
        "eval_signal_collapsed": eval_signal_collapsed,
        "latest_lr_fraction_of_initial": latest_lr_fraction_of_initial,
        "lr_near_zero": lr_near_zero,
        "scheduler_exhausted": scheduler_exhausted,
        "score_clip": score_clip,
        "score_scale": score_scale,
        "wdl_scale": wdl_scale,
        "result_lambda": result_lambda,
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
                f"latest_train_loss: {summary['latest_train_loss']:.6f}",
                f"latest_train_eval_loss: {summary['latest_train_eval_loss']:.6f}",
                f"latest_train_result_loss: {summary['latest_train_result_loss']:.6f}",
                f"latest_lr: {summary['latest_lr']:.8f}",
            ]
        )
    if summary["latest_validation_step"] is not None:
        lines.extend(
            [
                f"latest_validation_step: {summary['latest_validation_step']}",
                f"latest_validation_loss: {summary['latest_validation_loss']:.6f}",
                f"latest_validation_eval_loss: {summary['latest_validation_eval_loss']:.6f}",
                f"latest_validation_result_loss: {summary['latest_validation_result_loss']:.6f}",
            ]
        )
    else:
        lines.append("latest_validation_step: none")

    if summary["best_validation_loss"] is not None:
        lines.extend(
            [
                f"best_validation_step: {summary['best_validation_step']}",
                f"best_validation_loss: {summary['best_validation_loss']:.6f}",
            ]
        )
    else:
        lines.append("best_validation_step: none")

    lines.append(f"best_checkpoint_exists: {summary['best_checkpoint_exists']}")
    lines.append("")
    lines.append("Run Budget")
    lines.append(f"configured_total_steps: {_format_optional_int(summary['configured_total_steps'])}")
    lines.append(f"latest_train_step_fraction: {_format_optional_fraction(summary['latest_train_step_fraction'])}")
    lines.append(f"batch_size: {_format_optional_int(summary['batch_size'])}")
    lines.append(f"samples_seen: {_format_optional_int(summary['samples_seen'])}")
    lines.append(f"train_log_interval: {_format_optional_int(summary['train_log_interval'])}")
    lines.append(f"validation_interval: {_format_optional_int(summary['validation_interval'])}")
    lines.append(f"latest_lr_fraction_of_initial: {_format_optional_float(summary['latest_lr_fraction_of_initial'])}")
    lines.append(f"lr_near_zero: {summary['lr_near_zero']}")
    lines.append(f"scheduler_exhausted: {summary['scheduler_exhausted']}")
    lines.append("")
    lines.append("Generalization")
    lines.append(f"best_validation_gap: {_format_optional_float(summary['best_validation_gap'])}")
    lines.append(f"steps_since_best: {_format_optional_int(summary['steps_since_best'])}")
    lines.append(f"best_is_latest_validation: {summary['best_is_latest_validation']}")
    lines.append(f"resume_recommendation: {summary['resume_recommendation']}")
    lines.append(f"train_validation_gap: {_format_optional_float(summary['train_validation_gap'])}")
    lines.append(f"eval_gap: {_format_optional_float(summary['eval_gap'])}")
    lines.append(f"result_gap: {_format_optional_float(summary['result_gap'])}")
    lines.append(f"train_eval_to_result_ratio: {_format_optional_float(summary['train_eval_to_result_ratio'])}")
    lines.append(
        f"validation_eval_to_result_ratio: {_format_optional_float(summary['validation_eval_to_result_ratio'])}"
    )
    lines.append(f"eval_signal_collapsed: {summary['eval_signal_collapsed']}")
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

    if run.train_records:
        outputs.append(
            _plot_losses(
                plt,
                plots_dir / "train_loss.png",
                run.train_records,
                "global_step",
                [
                    ("loss", "blended loss"),
                    ("eval_loss", "eval target"),
                    ("result_loss", "game result"),
                ],
                "Train Loss",
                smooth_primary=True,
            )
        )
        outputs.append(
            _plot_losses(
                plt,
                plots_dir / "lr.png",
                run.train_records,
                "global_step",
                [("lr", "lr")],
                "Learning Rate",
                smooth_primary=False,
            )
        )

    if run.validation_records:
        outputs.append(
            _plot_losses(
                plt,
                plots_dir / "validation_loss.png",
                run.validation_records,
                "global_step",
                [
                    ("validation_loss", "blended loss"),
                    ("validation_eval_loss", "eval target"),
                    ("validation_result_loss", "game result"),
                ],
                "Validation Loss",
                smooth_primary=False,
            )
        )

    if run.train_records and run.validation_records:
        outputs.append(_plot_overview(plt, plots_dir / "loss_overview.png", run))

    return outputs


def _plot_losses(
    plt,
    output_path: Path,
    records: list[dict[str, object]],
    x_key: str,
    series: list[tuple[str, str]],
    title: str,
    *,
    smooth_primary: bool,
) -> Path:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    x_values = [int(record[x_key]) for record in records]
    primary_values: list[float] | None = None
    primary_smoothed: list[float] | None = None
    for index, (field, label) in enumerate(series):
        y_values = [float(record[field]) for record in records]
        if index == 0 and smooth_primary and len(y_values) >= 8:
            primary_values = y_values
            primary_smoothed = _moving_average(y_values, window=_smoothing_window(len(y_values)))
            axis.plot(x_values, y_values, label=f"{label} (raw)", alpha=0.2, linewidth=1.0)
            axis.plot(x_values, primary_smoothed, label=f"{label} (smoothed)", linewidth=2.2)
            continue
        axis.plot(x_values, y_values, label=label, linewidth=2.0 if index == 0 else 1.8)
    axis.set_title(title)
    axis.set_xlabel("Global Step")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    if primary_values is not None and primary_smoothed is not None:
        _set_focus_ylim(axis, primary_values, primary_smoothed)
    axis.text(
        0.01,
        0.01,
        "blended = weighted eval-target + game-result loss",
        transform=axis.transAxes,
        fontsize=9,
        alpha=0.75,
        va="bottom",
    )
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def _plot_overview(plt, output_path: Path, run: MetricsRun) -> Path:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    train_steps = [int(record["global_step"]) for record in run.train_records]
    train_loss = [float(record["loss"]) for record in run.train_records]
    validation_steps = [int(record["global_step"]) for record in run.validation_records]
    validation_loss = [float(record["validation_loss"]) for record in run.validation_records]
    smoothed_train = _moving_average(train_loss, window=_smoothing_window(len(train_loss)))

    axis.plot(
        train_steps,
        train_loss,
        label="train blended (raw)",
        alpha=0.12,
        linewidth=0.9,
    )
    axis.plot(
        train_steps,
        smoothed_train,
        label="train blended (smoothed)",
        linewidth=2.2,
    )
    axis.plot(
        validation_steps,
        validation_loss,
        label="validation blended",
        marker="o",
        markersize=3.5,
        linewidth=2.0,
    )
    axis.set_title("Loss Overview")
    axis.set_xlabel("Global Step")
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
            "best_validation_step": None,
            "config": None,
            "global_step": None,
        }
    try:
        from .checkpoint import load_checkpoint

        payload = load_checkpoint(checkpoint_path, map_location="cpu")
    except Exception:
        return {
            "best_validation_loss": None,
            "best_validation_step": None,
            "config": None,
            "global_step": None,
        }
    return {
        "best_validation_loss": payload.get("best_validation_loss"),
        "best_validation_step": payload.get("best_validation_step"),
        "config": payload.get("config"),
        "global_step": payload.get("global_step"),
    }


def _record_step(record: dict[str, object] | None) -> int | None:
    return None if record is None else int(record["global_step"])


def _config_value(run: MetricsRun, key: str) -> object | None:
    if run.checkpoint_config is None:
        return None
    return run.checkpoint_config.get(key)


def _closest_train_record(train_records: list[dict[str, object]], step: int | None) -> dict[str, object] | None:
    if step is None or not train_records:
        return None
    eligible = [record for record in train_records if int(record["global_step"]) <= step]
    if eligible:
        return eligible[-1]
    return train_records[-1]


def _infer_interval(records: list[dict[str, object]]) -> int | None:
    if len(records) < 2:
        return None
    deltas = [
        int(records[index]["global_step"]) - int(records[index - 1]["global_step"])
        for index in range(1, len(records))
    ]
    positive = [delta for delta in deltas if delta > 0]
    if not positive:
        return None
    return positive[-1]


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
    latest_validation_step: int | None,
    best_validation_step: int | None,
) -> str:
    if len(validation_records) < 2 or latest_validation_loss is None or best_validation_loss is None:
        return "insufficient-validation"
    if latest_validation_step is None or best_validation_step is None:
        return "insufficient-validation"

    gap = latest_validation_loss - best_validation_loss
    steps_since_best = latest_validation_step - best_validation_step
    if steps_since_best <= 0 or gap <= 5e-4:
        return "continue-latest"
    return "export-best"


def _build_suggestions(summary: dict[str, object]) -> list[str]:
    suggestions: list[str] = []
    if summary["resume_recommendation"] == "continue-latest":
        suggestions.append("Validation is still near its best point; resume from the latest step checkpoint and extend max_epochs.")
    elif summary["resume_recommendation"] == "export-best":
        suggestions.append("Best validation is materially earlier than the end; export best.pt and consider a fresh run for the next experiment.")
    else:
        suggestions.append("There are too few validation points to judge continuation confidently yet.")

    if summary["eval_signal_collapsed"]:
        suggestions.append("Eval loss is tiny relative to result loss; revisit score normalization or result_lambda.")
    if summary["scheduler_exhausted"]:
        suggestions.append("Learning rate is effectively exhausted; if validation is still improving, continue by increasing the total step budget.")
    if summary["samples_seen"] is not None and summary["samples_seen"] < 500_000_000:
        suggestions.append("Sample budget is still modest for very large binpack corpora; the run may simply need more steps.")
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


def _format_optional_float(value: object | None) -> str:
    if value is None:
        return "none"
    return f"{float(value):.6f}"


def _format_optional_fraction(value: object | None) -> str:
    if value is None:
        return "none"
    return f"{float(value) * 100.0:.2f}%"
