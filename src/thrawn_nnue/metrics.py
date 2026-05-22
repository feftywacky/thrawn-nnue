from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math


LEGACY_PLOT_FILENAMES = (
    "loss_overview.png",
    "train_loss.png",
    "validation_loss.png",
    "lr.png",
    "validation_quality.png",
)


@dataclass(slots=True)
class MetricsRun:
    run_dir: Path
    metrics_path: Path
    train_records: list[dict[str, object]]
    validation_records: list[dict[str, object]]
    best_checkpoint_exists: bool
    best_checkpoint_metric_name: str | None
    best_checkpoint_metric_value: float | None
    best_checkpoint_positions: int | None
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
    best_checkpoint_metric_name = checkpoint.get("best_checkpoint_metric_name")
    best_checkpoint_metric_value = _as_float(checkpoint.get("best_checkpoint_metric_value"))
    best_checkpoint_positions = _as_int(checkpoint.get("best_checkpoint_positions"))
    if best_checkpoint_metric_value is None and validation_records:
        best_record = min(validation_records, key=lambda record: float(record["score_mae"]))
        best_checkpoint_metric_name = "score_mae"
        best_checkpoint_metric_value = float(best_record["score_mae"])
        best_checkpoint_positions = _record_positions(best_record, batch_size)

    return MetricsRun(
        run_dir=run_path,
        metrics_path=metrics_path,
        train_records=train_records,
        validation_records=validation_records,
        best_checkpoint_exists=_best_checkpoint_path(run_path) is not None,
        best_checkpoint_metric_name=None if best_checkpoint_metric_name is None else str(best_checkpoint_metric_name),
        best_checkpoint_metric_value=best_checkpoint_metric_value,
        best_checkpoint_positions=best_checkpoint_positions,
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
    max_epochs = _as_int(_config_value(run, "max_epochs"))
    epoch_size = _as_int(_config_value(run, "epoch_size"))
    configured_total_positions = (
        None
        if max_epochs is None or epoch_size is None
        else max_epochs * epoch_size
    )
    start_lambda = _as_float(_config_value(run, "start_lambda", "lambda_"))
    end_lambda = _as_float(_config_value(run, "end_lambda", "lambda_"))
    nnue2score = _as_float(_config_value(run, "nnue2score"))
    in_scaling = _as_float(_config_value(run, "in_scaling"))
    out_scaling = _as_float(_config_value(run, "out_scaling"))
    gamma = _as_float(_config_value(run, "gamma"))
    filtered = _config_value(run, "filtered")
    wld_filtered = _config_value(run, "wld_filtered")
    random_fen_skipping = _as_int(_config_value(run, "random_fen_skipping"))
    early_fen_skipping = _as_int(_config_value(run, "early_fen_skipping"))
    soft_early_fen_skipping = _as_int(_config_value(run, "soft_early_fen_skipping"))
    simple_eval_skipping = _as_int(_config_value(run, "simple_eval_skipping"))
    pc_weights = [
        _as_float(_config_value(run, name))
        for name in ("pc_y0", "pc_y1", "pc_y2", "pc_y3", "pc_y4")
    ]
    ply_filter_points = [
        (
            _as_float(_config_value(run, f"ply_x{index}")),
            _as_float(_config_value(run, f"ply_y{index}")),
        )
        for index in range(1, 5)
    ]

    latest_train_step = _record_step(latest_train)
    latest_train_positions = _record_positions(latest_train, batch_size)
    latest_epoch_index = None if latest_train is None else _as_int(latest_train.get("epoch_index"))
    latest_validation_step = _record_step(latest_validation)
    latest_validation_positions = _record_positions(latest_validation, batch_size)
    latest_lr = _metric_value(latest_train, "lr")
    initial_lr = _metric_value(run.train_records[0] if run.train_records else None, "lr")

    train_log_interval_steps = _infer_interval(run.train_records, "global_step", batch_size=batch_size)
    observed_validation_spacing_positions = _infer_interval(
        run.validation_records,
        "positions_seen",
        batch_size=batch_size,
    )
    latest_train_at_validation = _closest_train_record(
        run.train_records,
        latest_validation_positions,
        batch_size=batch_size,
    )

    best_checkpoint_gap = None
    positions_since_best = None
    best_is_latest_validation = None
    if latest_validation is not None and run.best_checkpoint_metric_value is not None:
        latest_score_mae = _metric_value(latest_validation, "score_mae")
        if latest_score_mae is not None:
            best_checkpoint_gap = latest_score_mae - float(run.best_checkpoint_metric_value)
        if run.best_checkpoint_positions is not None and latest_validation_positions is not None:
            positions_since_best = latest_validation_positions - run.best_checkpoint_positions
            best_is_latest_validation = positions_since_best == 0

    train_validation_gap = _gap(
        _metric_value(latest_train_at_validation, "loss"),
        _metric_value(latest_validation, "validation_loss"),
    )
    wdl_gap = _gap(
        _metric_value(latest_train_at_validation, "wdl_loss"),
        _metric_value(latest_validation, "validation_wdl_loss"),
    )
    result_wdl_gap = _gap(
        _metric_value(latest_train_at_validation, "result_wdl_loss", "wdl_loss"),
        _metric_value(latest_validation, "validation_result_wdl_loss", "validation_wdl_loss"),
    )

    latest_position_fraction = None
    if configured_total_positions is not None and configured_total_positions > 0 and latest_train_positions is not None:
        latest_position_fraction = latest_train_positions / configured_total_positions

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
        latest_score_mae=_metric_value(latest_validation, "score_mae"),
        best_score_mae=run.best_checkpoint_metric_value,
        latest_validation_positions=latest_validation_positions,
        best_positions=run.best_checkpoint_positions,
    )

    summary = {
        "run_dir": str(run.run_dir),
        "status": status,
        "train_records": len(run.train_records),
        "validation_records": len(run.validation_records),
        "latest_train_step": latest_train_step,
        "positions_seen": latest_train_positions,
        "latest_epoch_index": latest_epoch_index,
        "latest_train_loss": _metric_value(latest_train, "loss"),
        "latest_train_wdl_loss": _metric_value(latest_train, "wdl_loss"),
        "latest_train_lambda": _metric_value(latest_train, "lambda"),
        "latest_train_teacher_wdl_loss": _metric_value(latest_train, "teacher_wdl_loss"),
        "latest_train_result_wdl_loss": _metric_value(latest_train, "result_wdl_loss", "wdl_loss"),
        "latest_train_output_reg_loss": _metric_value(latest_train, "output_reg_loss"),
        "latest_lr": latest_lr,
        "latest_validation_step": latest_validation_step,
        "latest_validation_positions": latest_validation_positions,
        "latest_validation_loss": _metric_value(latest_validation, "validation_loss"),
        "latest_validation_wdl_loss": _metric_value(latest_validation, "validation_wdl_loss"),
        "latest_validation_lambda": _metric_value(latest_validation, "lambda"),
        "latest_validation_teacher_wdl_loss": _metric_value(latest_validation, "validation_teacher_wdl_loss"),
        "latest_validation_result_wdl_loss": _metric_value(
            latest_validation,
            "validation_result_wdl_loss",
            "validation_wdl_loss",
        ),
        "latest_validation_output_reg_loss": _metric_value(latest_validation, "validation_output_reg_loss"),
        "latest_score_mae": _metric_value(latest_validation, "score_mae"),
        "latest_validation_score_rmse": _metric_value(latest_validation, "score_rmse"),
        "latest_validation_score_corr": _metric_value(latest_validation, "score_corr", "cp_corr"),
        "latest_validation_cp_mae": _metric_value(latest_validation, "cp_mae"),
        "latest_validation_cp_rmse": _metric_value(latest_validation, "cp_rmse"),
        "latest_validation_cp_corr": _metric_value(latest_validation, "cp_corr"),
        "latest_validation_wdl_accuracy": _metric_value(latest_validation, "wdl_accuracy"),
        "latest_validation_teacher_result_disagreement_rate": _metric_value(
            latest_validation,
            "teacher_result_disagreement_rate",
        ),
        "latest_validation_evaluated_positions": _metric_value(latest_validation, "validation_positions"),
        "best_checkpoint_exists": run.best_checkpoint_exists,
        "best_checkpoint_metric_name": run.best_checkpoint_metric_name,
        "best_checkpoint_metric_value": run.best_checkpoint_metric_value,
        "best_checkpoint_positions": run.best_checkpoint_positions,
        "configured_total_positions": configured_total_positions,
        "latest_position_fraction": latest_position_fraction,
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "epoch_size": epoch_size,
        "train_log_interval_steps": train_log_interval_steps,
        "observed_validation_spacing_positions": observed_validation_spacing_positions,
        "best_checkpoint_gap": best_checkpoint_gap,
        "positions_since_best": positions_since_best,
        "best_is_latest_validation": best_is_latest_validation,
        "resume_recommendation": resume_recommendation,
        "train_validation_gap": train_validation_gap,
        "wdl_gap": wdl_gap,
        "result_wdl_gap": result_wdl_gap,
        "latest_lr_fraction_of_initial": latest_lr_fraction_of_initial,
        "lr_near_zero": lr_near_zero,
        "scheduler_exhausted": scheduler_exhausted,
        "start_lambda": start_lambda,
        "end_lambda": end_lambda,
        "nnue2score": nnue2score,
        "in_scaling": in_scaling,
        "out_scaling": out_scaling,
        "gamma": gamma,
        "filtered": filtered,
        "wld_filtered": wld_filtered,
        "random_fen_skipping": random_fen_skipping,
        "early_fen_skipping": early_fen_skipping,
        "soft_early_fen_skipping": soft_early_fen_skipping,
        "simple_eval_skipping": simple_eval_skipping,
        "pc_weights": pc_weights,
        "ply_filter_points": ply_filter_points,
    }
    return summary


def render_summary_text(summary: dict[str, object]) -> str:
    lines = [
        f"run_dir: {summary['run_dir']}",
        f"status: {summary['status']}  train={summary['train_records']} validation={summary['validation_records']}",
        (
            "progress: "
            f"{_format_optional_int(summary['positions_seen'])}/"
            f"{_format_optional_int(summary['configured_total_positions'])}"
            f" ({_format_optional_fraction(summary['latest_position_fraction'])}), "
            f"epoch {_format_optional_int(summary['latest_epoch_index'])}/"
            f"{_format_optional_int(summary['max_epochs'])}"
        ),
        (
            "budget: "
            f"batch={_format_optional_int(summary['batch_size'])} "
            f"epoch_size={_format_optional_int(summary['epoch_size'])} "
            f"lambda={_format_optional_float(summary['start_lambda'])}->{_format_optional_float(summary['end_lambda'])}"
        ),
    ]
    if summary["latest_train_step"] is not None:
        lines.append(
            "train: "
            f"step={summary['latest_train_step']} "
            f"loss={_format_optional_float(summary['latest_train_loss'])} "
            f"wdl={_format_optional_float(summary['latest_train_wdl_loss'])} "
            f"result={_format_optional_float(summary['latest_train_result_wdl_loss'])} "
            f"lr={_format_optional_float(summary['latest_lr'], precision=8)}"
        )
    if summary["latest_validation_step"] is None:
        lines.append("validation: none")
    else:
        lines.append(
            "validation: "
            f"step={summary['latest_validation_step']} "
            f"loss={_format_optional_float(summary['latest_validation_loss'])} "
            f"wdl={_format_optional_float(summary['latest_validation_wdl_loss'])} "
            f"score_mae={_format_optional_float(summary['latest_score_mae'])} "
            f"cp_mae={_format_optional_float(summary['latest_validation_cp_mae'])} "
            f"cp_corr={_format_optional_float(summary['latest_validation_cp_corr'])} "
            f"wdl_acc={_format_optional_float(summary['latest_validation_wdl_accuracy'])} "
            f"positions={_format_optional_int(summary['latest_validation_evaluated_positions'])}"
        )
    lines.append(
        "best: "
        f"{summary['best_checkpoint_metric_name'] or 'score_mae'}="
        f"{_format_optional_float(summary['best_checkpoint_metric_value'])} "
        f"at={_format_optional_int(summary['best_checkpoint_positions'])} "
        f"gap={_format_optional_float(summary['best_checkpoint_gap'])} "
        f"checkpoint={summary['best_checkpoint_exists']}"
    )
    lines.append(
        "data: "
        f"filtered={summary['filtered']} "
        f"wld_filtered={summary['wld_filtered']} "
        f"random_fen_skipping={_format_optional_int(summary['random_fen_skipping'])} "
        f"early_fen_skipping={_format_optional_int(summary['early_fen_skipping'])} "
        f"soft_early_fen_skipping={_format_optional_int(summary['soft_early_fen_skipping'])} "
        f"simple_eval_skipping={_format_optional_int(summary['simple_eval_skipping'])}"
    )
    lines.append(f"resume: {summary['resume_recommendation']}")
    return "\n".join(lines)


def generate_run_plots(run: MetricsRun) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter, MaxNLocator
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for thrawn-nnue metrics plotting") from exc

    plots_dir = run.run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    for filename in LEGACY_PLOT_FILENAMES:
        (plots_dir / filename).unlink(missing_ok=True)
    outputs: list[Path] = []
    batch_size = _as_int(_config_value(run, "batch_size"))

    if run.train_records or run.validation_records:
        outputs.append(
            _plot_loss_overview(
                plt,
                FuncFormatter,
                MaxNLocator,
                plots_dir / "loss.png",
                run,
                batch_size=batch_size,
            )
        )
    if run.validation_records:
        outputs.append(
            _plot_mae(
                plt,
                FuncFormatter,
                MaxNLocator,
                plots_dir / "mae.png",
                run,
                batch_size=batch_size,
            )
        )

    return outputs


def _plot_loss_overview(plt, formatter_factory, locator_factory, output_path: Path, run: MetricsRun, *, batch_size: int | None) -> Path:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    smoothed_train: list[float] = []
    validation_positions: list[int] = []
    validation_loss: list[float] = []

    if run.train_records:
        train_positions = [_record_axis(record, "positions_seen", batch_size=batch_size) for record in run.train_records]
        train_loss = [_required_metric(record, ("loss",)) for record in run.train_records]
        smoothed_train = _moving_average(train_loss, window=_smoothing_window(len(train_loss)))
        axis.plot(
            train_positions,
            smoothed_train,
            label="train loss (smoothed)",
            linewidth=2.0,
            color="C0",
        )
    if run.validation_records:
        validation_positions = [
            _record_axis(record, "positions_seen", batch_size=batch_size)
            for record in run.validation_records
        ]
        validation_loss = [_required_metric(record, ("validation_loss",)) for record in run.validation_records]
        axis.plot(
            validation_positions,
            validation_loss,
            label="validation loss",
            marker="o",
            markersize=4.0,
            linewidth=2.0,
            color="C3",
        )
    axis.set_title("Loss")
    axis.set_xlabel("Positions Seen (B)")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="upper right")
    axis.xaxis.set_major_formatter(formatter_factory(_positions_billions_formatter))
    axis.xaxis.set_major_locator(locator_factory(nbins=6))
    _set_focus_ylim(axis, smoothed_train, validation_loss)

    metadata = f"validation points: {len(run.validation_records)}"
    axis.text(
        0.015,
        0.97,
        metadata,
        transform=axis.transAxes,
        fontsize=9,
        alpha=0.8,
        va="top",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.85},
    )

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def _plot_mae(
    plt,
    formatter_factory,
    locator_factory,
    output_path: Path,
    run: MetricsRun,
    *,
    batch_size: int | None,
) -> Path:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    positions = [_record_axis(record, "positions_seen", batch_size=batch_size) for record in run.validation_records]
    cp_mae = _complete_finite_series([_metric_value(record, "cp_mae") for record in run.validation_records], len(positions))
    score_mae = _complete_finite_series([_metric_value(record, "score_mae") for record in run.validation_records], len(positions))

    if cp_mae:
        values = cp_mae
        label = "cp_mae"
        axis.set_ylabel("Centipawns")
    elif score_mae:
        values = score_mae
        label = "score_mae"
        axis.set_ylabel("Score")
    else:
        values = []
        label = "mae"
        axis.set_ylabel("Error")

    if values:
        axis.plot(positions, values, marker="o", linewidth=2.0, color="C0", label=label)
        _set_focus_ylim(axis, values)
        best_index = _best_mae_index(run, batch_size=batch_size)
        if best_index is not None:
            best_position = positions[best_index]
            best_value = values[best_index]
            best_mae_step = int(run.validation_records[best_index].get("global_step", 0))
            axis.scatter([best_position], [best_value], s=40, color="C3", zorder=4, label="_nolegend_")
            axis.annotate(
                f"best MAE step {best_mae_step}\n{_format_positions_billions(best_position)}, {label} {best_value:.3f}",
                xy=(best_position, best_value),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.9},
                arrowprops={"arrowstyle": "->", "alpha": 0.6},
            )

    axis.set_title("MAE")
    axis.set_xlabel("Positions Seen (B)")
    axis.grid(True, alpha=0.3)
    axis.xaxis.set_major_formatter(formatter_factory(_positions_billions_formatter))
    axis.xaxis.set_major_locator(locator_factory(nbins=6))
    if values:
        axis.legend(loc="best")
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


def _complete_finite_series(values: list[float | None], expected_length: int) -> list[float]:
    series = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if len(series) != expected_length:
        return []
    return series


def _smoothing_window(length: int) -> int:
    return max(5, min(401, length // 50))


def _positions_billions_formatter(value: float, _position: int) -> str:
    return _format_positions_billions(value)


def _format_positions_billions(value: float) -> str:
    scaled = float(value) / 1_000_000_000.0
    if abs(scaled) >= 10.0 or scaled.is_integer():
        return f"{scaled:.0f}B"
    return f"{scaled:.1f}B"


def _set_focus_ylim(axis, *series_groups: list[float]) -> None:
    values = [value for group in series_groups for value in group if math.isfinite(value)]
    if not values:
        return
    lower = min(values)
    upper = max(values)
    spread = max(upper - lower, 1e-9)
    axis.set_ylim(lower - spread * 0.12, upper + spread * 0.12)


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
    checkpoint_path = _best_checkpoint_path(run_dir)
    if checkpoint_path is None:
        return {
            "best_checkpoint_metric_name": None,
            "best_checkpoint_metric_value": None,
            "best_checkpoint_positions": None,
            "config": None,
            "global_step": None,
            "positions_seen": None,
        }
    try:
        from .checkpoint import load_checkpoint

        payload = load_checkpoint(checkpoint_path, map_location="cpu")
    except Exception:
        return {
            "best_checkpoint_metric_name": None,
            "best_checkpoint_metric_value": None,
            "best_checkpoint_positions": None,
            "config": None,
            "global_step": None,
            "positions_seen": None,
        }

    config = payload.get("config")
    positions_seen = payload.get("positions_seen")

    return {
        "best_checkpoint_metric_name": payload.get("best_checkpoint_metric_name"),
        "best_checkpoint_metric_value": payload.get("best_checkpoint_metric_value"),
        "best_checkpoint_positions": payload.get("best_checkpoint_positions"),
        "config": config,
        "global_step": payload.get("global_step"),
        "positions_seen": positions_seen,
    }


def _best_checkpoint_path(run_dir: Path) -> Path | None:
    checkpoints_dir = run_dir / "checkpoints"
    alias_path = checkpoints_dir / "best.pt"
    if alias_path.exists():
        return alias_path
    stamped_paths = sorted(checkpoints_dir.glob("epoch_*_best.pt"))
    if stamped_paths:
        return stamped_paths[-1]
    return None


def _best_mae_index(run: MetricsRun, *, batch_size: int | None) -> int | None:
    if not run.validation_records:
        return None
    if run.best_checkpoint_positions is not None:
        for index, record in enumerate(run.validation_records):
            if _record_positions(record, batch_size) == run.best_checkpoint_positions:
                return index
    best_mae = run.best_checkpoint_metric_value
    if best_mae is None:
        best_mae = min(_required_metric(record, ("score_mae",)) for record in run.validation_records)
    for index, record in enumerate(run.validation_records):
        if math.isclose(_required_metric(record, ("score_mae",)), best_mae, rel_tol=0.0, abs_tol=1e-12):
            return index
    return None


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


def _config_value(run: MetricsRun, *keys: str) -> object | None:
    for key in keys:
        if run.checkpoint_config is not None and key in run.checkpoint_config:
            return run.checkpoint_config.get(key)
    for records in (run.validation_records, run.train_records):
        if not records:
            continue
        for key in keys:
            if key in records[-1]:
                return records[-1].get(key)
    return None


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


def _gap(train_value: object | None, validation_value: object | None) -> float | None:
    if train_value is None or validation_value is None:
        return None
    return float(validation_value) - float(train_value)


def _resume_recommendation(
    *,
    validation_records: list[dict[str, object]],
    latest_score_mae: float | None,
    best_score_mae: float | None,
    latest_validation_positions: int | None,
    best_positions: int | None,
) -> str:
    if len(validation_records) < 2 or latest_score_mae is None or best_score_mae is None:
        return "insufficient-validation"
    if latest_validation_positions is None or best_positions is None:
        return "insufficient-validation"

    gap = latest_score_mae - best_score_mae
    positions_since_best = latest_validation_positions - best_positions
    if positions_since_best <= 0 or gap <= 2.0:
        return "continue-latest"
    return "export-best"


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
