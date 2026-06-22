"""Normalization metadata and array transformations."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class NormalizationStats:
    mu_x: float
    sd_x: float
    mu_y: float
    sd_y: float
    path: Path


def load_normalization_stats(path: str | Path) -> NormalizationStats:
    stats_path = Path(path).expanduser().resolve()
    if not stats_path.is_file():
        raise FileNotFoundError(f"Normalization statistics not found: {stats_path}")
    with np.load(stats_path) as data:
        missing = [key for key in ("mu_X", "sd_X", "mu_Y", "sd_Y") if key not in data]
        if missing:
            raise KeyError(f"Missing normalization values {missing} in {stats_path}")
        stats = NormalizationStats(
            mu_x=float(data["mu_X"]),
            sd_x=float(data["sd_X"]),
            mu_y=float(data["mu_Y"]),
            sd_y=float(data["sd_Y"]),
            path=stats_path,
        )
    if stats.sd_x <= 0 or stats.sd_y <= 0:
        raise ValueError(f"Normalization standard deviations must be positive: {stats_path}")
    return stats


def normalize_lr(values: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (np.asarray(values, dtype=np.float32) - stats.mu_x) / stats.sd_x


def denormalize_hr(values: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return np.asarray(values, dtype=np.float32) * stats.sd_y + stats.mu_y
