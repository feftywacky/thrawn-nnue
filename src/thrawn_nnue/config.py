from __future__ import annotations

from dataclasses import asdict, dataclass, field
from glob import glob
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
    prefetch_batches: int = 0
    batch_size: int = 256
    total_train_positions: int = 0
    superbatch_positions: int = 0
    validation_interval_positions: int = 0
    validation_positions: int = 0
    checkpoint_every: int = 250
    log_every: int = 25
    console_mode: str = "progress"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    output_regularization: float = 0.0
    clip_grad_norm: float = 1.0
    amp: bool = True
    feature_set: str = "a768"
    num_features: int = 768
    max_active_features: int = 32
    ft_size: int = 256
    hidden_size: int = 32
    output_buckets: int = 1
    output_perspective: str = "stm"
    score_clip: float = 0.0
    score_scale: float = 1.0
    wdl_scale: float = 410.0
    eval_lambda: float = 0.7
    lr_drop_fractions: list[float] = field(default_factory=lambda: [0.80, 0.95])
    lr_drop_factor: float = 0.1
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
            cfg.train_datasets = _resolve_dataset_list(base_dir, cfg.train_datasets)
            cfg.validation_datasets = _resolve_dataset_list(base_dir, cfg.validation_datasets)
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
        self.feature_set = _canonical_feature_set(self.feature_set)
        if self.num_features != 768:
            raise ValueError("A-768 requires num_features=768")
        if self.max_active_features != 32:
            raise ValueError("Chess A-768 expects max_active_features=32")
        if self.output_buckets <= 0:
            raise ValueError("output_buckets must be positive")
        if self.output_perspective != "stm":
            raise ValueError("Only output_perspective='stm' is supported")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.total_train_positions <= 0:
            raise ValueError("total_train_positions must be positive")
        if self.superbatch_positions <= 0:
            raise ValueError("superbatch_positions must be positive")
        if self.validation_interval_positions < 0:
            raise ValueError("validation_interval_positions must be >= 0")
        if self.validation_positions < 0:
            raise ValueError("validation_positions must be >= 0")
        if self.score_clip < 0.0:
            raise ValueError("score_clip must be >= 0")
        if self.score_scale <= 0.0:
            raise ValueError("score_scale must be > 0")
        if not 0.0 <= self.eval_lambda <= 1.0:
            raise ValueError("eval_lambda must be in [0, 1]")
        if self.export_ft_scale <= 0.0 or self.export_dense_scale <= 0.0:
            raise ValueError("export scales must be > 0")
        if self.num_loader_threads <= 0:
            raise ValueError("num_loader_threads must be positive")
        if self.prefetch_batches < 0:
            raise ValueError("prefetch_batches must be >= 0")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if self.output_regularization < 0.0:
            raise ValueError("output_regularization must be >= 0")
        if self.clip_grad_norm <= 0.0:
            raise ValueError("clip_grad_norm must be positive")
        if self.ft_size <= 0 or self.hidden_size <= 0:
            raise ValueError("network sizes must be positive")
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device must be one of: auto, cuda, mps, cpu")
        if self.console_mode not in {"progress", "text"}:
            raise ValueError("console_mode must be one of: progress, text")
        if self.lr_drop_factor <= 0.0 or self.lr_drop_factor > 1.0:
            raise ValueError("lr_drop_factor must be in (0, 1]")
        previous_fraction = None
        for fraction in self.lr_drop_fractions:
            if not 0.0 < float(fraction) < 1.0:
                raise ValueError("lr_drop_fractions entries must be in (0, 1)")
            if previous_fraction is not None and float(fraction) <= previous_fraction:
                raise ValueError("lr_drop_fractions must be strictly increasing")
            previous_fraction = float(fraction)

        overlap = _dataset_overlap(self.train_datasets, self.validation_datasets)
        if overlap:
            overlap_list = ", ".join(overlap)
            raise ValueError(f"train_datasets and validation_datasets overlap: {overlap_list}")


def load_config(path: str | Path) -> TrainConfig:
    return TrainConfig.from_toml(path)


def _resolve_dataset_list(base_dir: Path, values: list[str]) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for value in values:
        for path in _expand_dataset_value(base_dir, value):
            if path not in seen:
                resolved.append(path)
                seen.add(path)
    return resolved


def _expand_dataset_value(base_dir: Path, value: str) -> list[str]:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()

    if _looks_like_glob(value):
        matches = sorted(Path(match).resolve() for match in glob(str(candidate), recursive=True))
        files = [str(path) for path in matches if path.is_file()]
        if not files:
            raise ValueError(f"Dataset glob matched no files: {value}")
        return files

    if candidate.is_dir():
        files = sorted(path.resolve() for path in candidate.rglob("*.binpack") if path.is_file())
        if not files:
            raise ValueError(f"Dataset directory contains no .binpack files: {candidate}")
        return [str(path) for path in files]

    return [str(candidate)]


def _looks_like_glob(value: str) -> bool:
    return any(char in value for char in "*?[")


def _canonical_feature_set(value: str) -> str:
    if value == "a768":
        return "a768"
    raise ValueError("Only feature_set='a768' is supported")


def _dataset_overlap(train_datasets: list[str], validation_datasets: list[str]) -> list[str]:
    train_paths = _resolved_dataset_path_set(train_datasets)
    validation_paths = _resolved_dataset_path_set(validation_datasets)
    return sorted(train_paths & validation_paths)


def _resolved_dataset_path_set(values: list[str]) -> set[str]:
    return {str(Path(value).resolve()) for value in values}
