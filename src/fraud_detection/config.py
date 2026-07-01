"""Typed configuration loaded from a YAML file.

The whole pipeline is driven by a single :class:`Config` object so that runs are reproducible
and every knob (split ratios, cost ratio, model hyper-parameters) lives in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass
class DataConfig:
    raw_csv: str = "data/creditcard.csv"
    openml_name: str = "creditcard"
    openml_version: int = 1


@dataclass
class SplitConfig:
    test_size: float = 0.2
    val_size: float = 0.2


@dataclass
class FeaturesConfig:
    scale_columns: list[str] = field(default_factory=lambda: ["Amount"])


@dataclass
class CostConfig:
    c_fn: float = 100.0
    c_fp: float = 1.0


@dataclass
class ThresholdConfig:
    grid_points: int = 200
    grid_low: float = 0.001
    grid_high: float = 0.999


@dataclass
class BootstrapConfig:
    n_resamples: int = 1000
    ci: float = 0.95


@dataclass
class CalibrationConfig:
    method: str = "isotonic"


@dataclass
class TuningConfig:
    enabled: bool = True
    n_iter: int = 20
    cv_folds: int = 3


@dataclass
class Config:
    seed: int = 42
    artifacts_dir: str = "artifacts"
    data: DataConfig = field(default_factory=DataConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    models: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = yaml.safe_load(path.read_text()) or {}
        cfg = _build(cls, raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        s = self.split
        if not 0 < s.test_size < 1 or not 0 < s.val_size < 1:
            raise ValueError("split.test_size and split.val_size must be in (0, 1)")
        if s.test_size + s.val_size >= 1:
            raise ValueError("split.test_size + split.val_size must leave room for training")
        if self.cost.c_fn <= 0 or self.cost.c_fp <= 0:
            raise ValueError("cost.c_fn and cost.c_fp must be positive")
        if self.calibration.method not in ("isotonic", "sigmoid"):
            raise ValueError("calibration.method must be 'isotonic' or 'sigmoid'")

    @property
    def artifacts_path(self) -> Path:
        return Path(self.artifacts_dir)


def _build(cls: type, raw: dict[str, Any]) -> Any:
    """Recursively map a nested dict onto nested dataclasses, ignoring unknown keys."""
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in raw:
            continue
        value = raw[f.name]
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[f.name] = _build(ftype, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)
