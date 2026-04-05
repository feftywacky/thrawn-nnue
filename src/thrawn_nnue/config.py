from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import tomllib


@dataclass(slots=True)
class TrainConfig:
    run_name: str = "default"
    output_dir: str = "runs/default"
    train_datasets: list[str] = field(default_factory=list)
    validation_datasets: list[str] = field(default_factory=list)
    device: str = "auto"
    num_loader_threads: int = 2
    batch_size: int = 256
    steps_per_epoch: int = 1000
    validation_every: int = 500
    validation_steps: int = 0
    max_epochs: int = 10
    checkpoint_every: int = 250
    log_every: int = 25
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    clip_grad_norm: float = 1.0
    amp: bool = True
    feature_set: str = "a768_dual"
    num_features: int = 768
    max_active_features: int = 32
    ft_size: int = 256
    hidden_size: int = 32
    output_perspective: str = "stm"
    score_clip: float = 0.0
    score_scale: float = 1.0
    wdl_scale: float = 410.0
    result_lambda: float = 0.5
    export_ft_scale: float = 127.0
    export_dense_scale: float = 64.0
    export_description: str = "thrawn a768 dual nnue"

    def resolved_output_dir(self, root: Path | None = None) -> Path:
        base = root or Path.cwd()
        return (base / self.output_dir).resolve()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: Path | None = None) -> "TrainConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        extras = sorted(set(data) - known)
        if extras:
            raise ValueError(f"Unknown config keys: {', '.join(extras)}")

        cfg = cls(**data)
        if base_dir is not None:
            cfg.train_datasets = _resolve_path_list(base_dir, cfg.train_datasets)
            cfg.validation_datasets = _resolve_path_list(base_dir, cfg.validation_datasets)
            cfg.output_dir = str((base_dir / cfg.output_dir).resolve())
        cfg.validate()
        return cfg

    @classmethod
    def from_toml(cls, path: str | Path) -> "TrainConfig":
        config_path = Path(path).resolve()
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        return cls.from_dict(raw, base_dir=config_path.parent)

    def validate(self) -> None:
        if self.feature_set != "a768_dual":
            raise ValueError("Only feature_set='a768_dual' is supported in this scaffold")
        if self.num_features != 768:
            raise ValueError("A-768 requires num_features=768")
        if self.max_active_features != 32:
            raise ValueError("Chess A-768 expects max_active_features=32")
        if self.output_perspective != "stm":
            raise ValueError("Only output_perspective='stm' is supported")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive")
        if self.validation_every <= 0:
            raise ValueError("validation_every must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if not 0.0 <= self.result_lambda <= 1.0:
            raise ValueError("result_lambda must be in [0, 1]")
        if self.score_clip < 0.0:
            raise ValueError("score_clip must be >= 0")
        if self.score_scale <= 0.0:
            raise ValueError("score_scale must be > 0")
        if self.num_loader_threads <= 0:
            raise ValueError("num_loader_threads must be positive")
        if self.ft_size <= 0 or self.hidden_size <= 0:
            raise ValueError("network sizes must be positive")
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device must be one of: auto, cuda, mps, cpu")
        if self.validation_datasets and self.validation_steps <= 0:
            raise ValueError("validation_steps must be positive when validation_datasets are configured")


def load_config(path: str | Path) -> TrainConfig:
    return TrainConfig.from_toml(path)


def _resolve_path_list(base_dir: Path, values: list[str]) -> list[str]:
    resolved: list[str] = []
    for value in values:
        p = Path(value)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        resolved.append(str(p))
    return resolved
