from __future__ import annotations

from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path
from typing import Any
import math
import tomllib

from .features import FEATURES, canonical_feature_name, feature_shape


@dataclass(slots=True)
class TrainConfig:
    run_name: str = "default"
    default_root_dir: str = "runs/default"
    datasets: list[str] = field(default_factory=list)
    validation_datasets: list[str] = field(default_factory=list)
    validation_size: int = 0
    check_val_every_n_epoch: int = 1

    accelerator: str = "auto"
    num_workers: int = 2
    batch_size: int = 16_384
    pin_memory: bool = True
    data_loader_queue_size: int = 8
    tf32: bool = True
    fused_optimizer: bool = True
    seed: int = 42

    max_epochs: int = 800
    epoch_size: int = 100_000_000
    network_save_period: int = 20
    save_last_network: bool = True
    console_mode: str = "progress"

    features: str = FEATURES
    num_features: int = 22_528
    max_active_features: int = 32
    ft_size: int = 1024
    hidden_size: int = 31
    forward_size: int = 1
    fc1_output_size: int = 32

    filtered: bool = True
    wld_filtered: bool = True
    random_fen_skipping: int = 0
    early_fen_skipping: int = -1
    soft_early_fen_skipping: int = 20
    simple_eval_skipping: int = -1
    pc_y0: float = 0.0
    pc_y1: float = 0.4
    pc_y2: float = 1.0
    pc_y3: float = 1.0
    pc_y4: float = 0.75
    ply_x1: float = 0.0
    ply_y1: float = 0.1
    ply_x2: float = 6.0
    ply_y2: float = 0.15
    ply_x3: float = 10.0
    ply_y3: float = 0.25
    ply_x4: float = 18.0
    ply_y4: float = 0.75

    lr: float = 8.75e-4
    gamma: float = 0.992
    clip_grad_norm: float = 10.0
    amp: bool = True

    nnue2score: float = 600.0
    lambda_: float = 1.0
    start_lambda: float | None = None
    end_lambda: float | None = None
    in_offset: float = 270.0
    in_scaling: float = 340.0
    out_offset: float = 270.0
    out_scaling: float = 380.0
    pow_exp: float = 2.5
    qp_asymmetry: float = 0.0
    w1: float = 0.0
    w2: float = 0.5

    export_ft_scale: float = 255.0
    export_dense_scale: float = 64.0
    export_description: str = "thrawn HalfKAv2_hm 1024x2 -> 31+1 -> 32 -> 1 nnue"

    @property
    def total_positions(self) -> int:
        return self.max_epochs * self.epoch_size

    @property
    def num_batches_per_epoch(self) -> int:
        return max(1, (self.epoch_size + self.batch_size - 1) // self.batch_size)

    def resolved_root_dir(self, root: Path | None = None) -> Path:
        base = root or Path.cwd()
        return (base / self.default_root_dir).resolve()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, base_dir: Path | None = None) -> "TrainConfig":
        data = _normalize_config_keys(data)
        known = {field.name for field in cls.__dataclass_fields__.values()}
        extras = sorted(set(data) - known)
        if extras:
            raise ValueError(f"Unknown config keys: {', '.join(extras)}")

        cfg = cls(**data)
        if base_dir is not None:
            cfg.datasets = _resolve_dataset_list(base_dir, cfg.datasets)
            cfg.validation_datasets = _resolve_dataset_list(base_dir, cfg.validation_datasets)
            cfg.default_root_dir = str((base_dir / cfg.default_root_dir).resolve())
        cfg.validate()
        return cfg

    @classmethod
    def from_toml(cls, path: str | Path) -> "TrainConfig":
        config_path = Path(path).resolve()
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
        return cls.from_dict(raw, base_dir=config_path.parent)

    def validate(self) -> None:
        self.features = canonical_feature_name(self.features)
        shape = feature_shape(self.features)
        self.num_features = shape["num_features"]
        self.max_active_features = shape["max_active_features"]
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.epoch_size <= 0:
            raise ValueError("epoch_size must be positive")
        if self.validation_size < 0:
            raise ValueError("validation_size must be >= 0")
        if self.check_val_every_n_epoch < 1:
            raise ValueError("check_val_every_n_epoch must be >= 1")
        if self.network_save_period < 0:
            raise ValueError("network_save_period must be >= 0")
        if self.num_workers <= 0:
            raise ValueError("num_workers must be positive")
        if self.data_loader_queue_size < 0:
            raise ValueError("data_loader_queue_size must be >= 0")
        if self.random_fen_skipping < 0:
            raise ValueError("random_fen_skipping must be >= 0")
        if self.early_fen_skipping < -1:
            raise ValueError("early_fen_skipping must be >= -1")
        if self.simple_eval_skipping < -1:
            raise ValueError("simple_eval_skipping must be >= -1")
        for name in (
            "pc_y0",
            "pc_y1",
            "pc_y2",
            "pc_y3",
            "pc_y4",
            "ply_x1",
            "ply_y1",
            "ply_x2",
            "ply_y2",
            "ply_x3",
            "ply_y3",
            "ply_x4",
            "ply_y4",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if self.lr <= 0.0:
            raise ValueError("lr must be positive")
        if self.gamma <= 0.0 or self.gamma > 1.0:
            raise ValueError("gamma must be > 0 and <= 1")
        if self.clip_grad_norm <= 0.0:
            raise ValueError("clip_grad_norm must be positive")
        if self.ft_size != 1024:
            raise ValueError("ft_size must be 1024")
        if self.hidden_size != 31:
            raise ValueError("hidden_size must be 31")
        if self.forward_size != 1:
            raise ValueError("forward_size must be 1")
        if self.fc1_output_size != 32:
            raise ValueError("fc1_output_size must be 32")
        if self.accelerator not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("accelerator must be one of: auto, cuda, mps, cpu")
        if self.console_mode not in {"progress", "text"}:
            raise ValueError("console_mode must be one of: progress, text")
        if self.nnue2score <= 0.0:
            raise ValueError("nnue2score must be positive")
        if self.start_lambda is None and self.end_lambda is None:
            self.start_lambda = self.lambda_
            self.end_lambda = self.lambda_
        elif self.start_lambda is None or self.end_lambda is None:
            raise ValueError("Either both or none of start_lambda and end_lambda must be specified")
        for name, value in (
            ("lambda", self.lambda_),
            ("start_lambda", self.start_lambda),
            ("end_lambda", self.end_lambda),
        ):
            if value is None or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.in_scaling <= 0.0 or self.out_scaling <= 0.0:
            raise ValueError("WDL scaling values must be positive")
        if self.pow_exp <= 0.0:
            raise ValueError("pow_exp must be positive")
        if self.qp_asymmetry < 0.0:
            raise ValueError("qp_asymmetry must be >= 0")
        if self.export_ft_scale <= 0.0 or self.export_dense_scale <= 0.0:
            raise ValueError("export scales must be > 0")


def load_config(path: str | Path) -> TrainConfig:
    return TrainConfig.from_toml(path)


def _normalize_config_keys(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    if "lambda" in normalized:
        value = normalized.pop("lambda")
        if "lambda_" in normalized and normalized["lambda_"] != value:
            raise ValueError("Conflicting config keys: lambda and lambda_")
        normalized["lambda_"] = value
    _drop_legacy_fixed_key(normalized, "output_perspective", "stm")
    _drop_legacy_fixed_key(normalized, "optimizer_name", "adamw")
    _drop_legacy_fixed_key(normalized, "weight_decay", 0.0)
    _drop_legacy_fixed_key(normalized, "param_index", 0)
    return normalized


def _drop_legacy_fixed_key(data: dict[str, Any], key: str, expected: object) -> None:
    if key not in data:
        return
    value = data.pop(key)
    if isinstance(expected, float):
        matches = math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
    else:
        matches = value == expected
    if not matches:
        raise ValueError(f"{key} is no longer configurable; expected legacy value {expected!r}")


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
