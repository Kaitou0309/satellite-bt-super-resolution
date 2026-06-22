"""Configuration loading and validation."""

from pathlib import Path
from typing import Any

import yaml


REQUIRED_GENERATOR_FIELDS = (
    "architecture",
    "rrdb_blocks",
    "rdb_per_rrdb",
    "conv_layers_per_rdb",
    "filters",
    "residual_scaling",
    "upsampling",
)


def load_model_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    if not isinstance(config, dict):
        raise ValueError(f"Model config must contain a YAML mapping: {config_path}")

    missing = [key for key in ("name", "scale", "channels", "generator") if key not in config]
    if missing:
        raise ValueError(f"Missing required config fields {missing}: {config_path}")

    generator = config["generator"]
    if not isinstance(generator, dict):
        raise ValueError("The generator field must be a mapping.")
    missing_generator = [key for key in REQUIRED_GENERATOR_FIELDS if key not in generator]
    if missing_generator:
        raise ValueError(f"Missing generator fields {missing_generator}: {config_path}")
    if str(generator["architecture"]).lower() != "rrdn":
        raise ValueError(f"Unsupported generator architecture: {generator['architecture']!r}")

    return config


def find_repository_root(config_path: str | Path) -> Path:
    config_path = Path(config_path).expanduser().resolve()
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "configs").is_dir() and (candidate / "metadata").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate repository root from {config_path}")
