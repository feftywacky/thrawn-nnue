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

    best_validation_loss, best_validation_step = _best_validation_from_checkpoint(run_path)
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
    )


def summarize_run(run: MetricsRun) -> dict[str, object]:
    latest_train = run.train_records[-1] if run.train_records else None
    latest_validation = run.validation_records[-1] if run.validation_records else None
    status = "validated" if run.validation_records else "train-only"
    if not run.train_records and not run.validation_records:
        status = "missing-metrics"

    return {
        "run_dir": str(run.run_dir),
        "status": status,
        "train_records": len(run.train_records),
        "validation_records": len(run.validation_records),
        "latest_train_step": None if latest_train is None else int(latest_train["global_step"]),
        "latest_train_loss": None if latest_train is None else float(latest_train["loss"]),
        "latest_train_eval_loss": None if latest_train is None else float(latest_train["eval_loss"]),
        "latest_train_result_loss": None if latest_train is None else float(latest_train["result_loss"]),
        "latest_lr": None if latest_train is None else float(latest_train["lr"]),
        "latest_validation_step": None if latest_validation is None else int(latest_validation["global_step"]),
        "latest_validation_loss": None if latest_validation is None else float(latest_validation["validation_loss"]),
        "latest_validation_eval_loss": None if latest_validation is None else float(latest_validation["validation_eval_loss"]),
        "latest_validation_result_loss": None if latest_validation is None else float(latest_validation["validation_result_loss"]),
        "best_validation_loss": run.best_validation_loss,
        "best_validation_step": run.best_validation_step,
        "best_checkpoint_exists": run.best_checkpoint_exists,
    }


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
                [("loss", "blended"), ("eval_loss", "eval"), ("result_loss", "result")],
                "Train Loss",
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
                    ("validation_loss", "blended"),
                    ("validation_eval_loss", "eval"),
                    ("validation_result_loss", "result"),
                ],
                "Validation Loss",
            )
        )

    if run.train_records and run.validation_records:
        outputs.append(_plot_overview(plt, plots_dir / "loss_overview.png", run))

    return outputs


def _plot_losses(plt, output_path: Path, records: list[dict[str, object]], x_key: str, series: list[tuple[str, str]], title: str) -> Path:
    figure, axis = plt.subplots(figsize=(8, 5))
    x_values = [int(record[x_key]) for record in records]
    for field, label in series:
        y_values = [float(record[field]) for record in records]
        axis.plot(x_values, y_values, label=label)
    axis.set_title(title)
    axis.set_xlabel("Global Step")
    axis.set_ylabel("Value")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def _plot_overview(plt, output_path: Path, run: MetricsRun) -> Path:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        [int(record["global_step"]) for record in run.train_records],
        [float(record["loss"]) for record in run.train_records],
        label="train blended",
    )
    axis.plot(
        [int(record["global_step"]) for record in run.validation_records],
        [float(record["validation_loss"]) for record in run.validation_records],
        label="validation blended",
    )
    axis.set_title("Loss Overview")
    axis.set_xlabel("Global Step")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            records.append(json.loads(stripped))
    return records


def _best_validation_from_checkpoint(run_dir: Path) -> tuple[float | None, int | None]:
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        return None, None
    try:
        from .checkpoint import load_checkpoint
        payload = load_checkpoint(checkpoint_path, map_location="cpu")
    except Exception:
        return None, None
    return payload.get("best_validation_loss"), payload.get("best_validation_step")
