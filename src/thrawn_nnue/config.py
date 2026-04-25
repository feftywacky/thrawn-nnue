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
    epoch_positions: int = 0
    validation_interval_positions: int = 0
    validation_positions: int = 0
    checkpoint_every: int = 250
    log_every: int = 25
    console_mode: str = "progress"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    output_regularization: float = 0.0
    sanity_anchor_weight: float = 0.0
    clip_grad_norm: float = 1.0
    amp: bool = True
    feature_set: str = "halfkp"
    num_features: int = 40960
    num_factor_features: int = 640
    max_active_features: int = 30
    ft_size: int = 1024
    l1_size: int = 256
    l2_size: int = 64
    output_perspective: str = "stm"
    score_clip: float = 4000.0
    wdl_lambda: float = 1.0
    wdl_in_offset: float = 270.0
    wdl_out_offset: float = 270.0
    wdl_in_scaling: float = 340.0
    wdl_out_scaling: float = 380.0
    wdl_loss_power: float = 2.5
    skip_capture_positions: bool = True
    skip_decisive_score_mismatch: bool = True
    decisive_score_mismatch_margin: float = 1000.0
    skip_draw_score_mismatch: bool = True
    draw_score_mismatch_margin: float = 1000.0
    max_abs_score: float = 0.0
    export_ft_scale: float = 127.0
    export_dense_scale: float = 64.0
    export_description: str = "thrawn halfkp nnue"

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
        if self.num_features != 40960:
            raise ValueError("HalfKP requires num_features=40960")
        if self.num_factor_features != 640:
            raise ValueError("HalfKP P factorization requires num_factor_features=640")
        if self.max_active_features != 30:
            raise ValueError("Chess HalfKP expects max_active_features=30")
        if self.output_perspective != "stm":
            raise ValueError("Only output_perspective='stm' is supported")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.total_train_positions <= 0:
            raise ValueError("total_train_positions must be positive")
        if self.epoch_positions <= 0:
            raise ValueError("epoch_positions must be positive")
        if self.validation_interval_positions < 0:
            raise ValueError("validation_interval_positions must be >= 0")
        if self.validation_positions < 0:
            raise ValueError("validation_positions must be >= 0")
        if self.score_clip < 0.0:
            raise ValueError("score_clip must be >= 0")
        if self.wdl_lambda < 0.0 or self.wdl_lambda > 1.0:
            raise ValueError("wdl_lambda must be between 0 and 1")
        if self.wdl_in_scaling <= 0.0 or self.wdl_out_scaling <= 0.0:
            raise ValueError("wdl scaling values must be positive")
        if self.wdl_loss_power <= 0.0:
            raise ValueError("wdl_loss_power must be positive")
        if self.decisive_score_mismatch_margin < 0.0:
            raise ValueError("decisive_score_mismatch_margin must be >= 0")
        if self.draw_score_mismatch_margin < 0.0:
            raise ValueError("draw_score_mismatch_margin must be >= 0")
        if self.max_abs_score < 0.0:
            raise ValueError("max_abs_score must be >= 0")
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
        if self.sanity_anchor_weight < 0.0:
            raise ValueError("sanity_anchor_weight must be >= 0")
        if self.clip_grad_norm <= 0.0:
            raise ValueError("clip_grad_norm must be positive")
        if self.ft_size <= 0 or self.l1_size <= 0 or self.l2_size <= 0:
            raise ValueError("network sizes must be positive")
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device must be one of: auto, cuda, mps, cpu")
        if self.console_mode not in {"progress", "text"}:
            raise ValueError("console_mode must be one of: progress, text")

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
    if value == "halfkp":
        return "halfkp"
    raise ValueError("Only feature_set='halfkp' is supported")


def _dataset_overlap(train_datasets: list[str], validation_datasets: list[str]) -> list[str]:
    train_paths = _resolved_dataset_path_set(train_datasets)
    validation_paths = _resolved_dataset_path_set(validation_datasets)
    return sorted(train_paths & validation_paths)


def _resolved_dataset_path_set(values: list[str]) -> set[str]:
    return {str(Path(value).resolve()) for value in values}
