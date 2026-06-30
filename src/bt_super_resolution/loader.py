"""Config-driven loading for MW-SR and MW-SR-GAN generator releases."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.models import load_model

from .config import find_repository_root, load_model_config
from .models import build_rrdn
from .normalization import NormalizationStats, denormalize_hr, load_normalization_stats, normalize_lr


@dataclass
class ModelBundle:
    """A loaded generator together with the metadata needed for inference."""

    model: Model
    config: dict[str, Any]
    config_path: Path
    weights_path: Path
    normalization: NormalizationStats

    @property
    def name(self) -> str:
        return str(self.config.get("display_name", self.config["name"]))

    @property
    def is_gan_generator(self) -> bool:
        return "gan" in self.config["name"].lower()

    def predict_kelvin(self, lr_bt: np.ndarray, *, batch_size: int = 1, verbose: int = 0) -> np.ndarray:
        del verbose  # Kept for API compatibility with Keras-style inference calls.
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        values = np.asarray(lr_bt, dtype=np.float32)
        original_ndim = values.ndim
        single_channel_image = original_ndim == 3 and values.shape[-1] == 1
        if original_ndim == 2:
            values = values[None, ..., None]
        elif original_ndim == 3:
            values = values[None, ...] if single_channel_image else values[..., None]
        elif original_ndim != 4:
            raise ValueError(f"Expected a 2D, 3D, or 4D LR array, received shape {values.shape}")

        values = np.nan_to_num(values, nan=self.normalization.mu_x)
        normalized = normalize_lr(values, self.normalization)
        batches = []
        for start in range(0, normalized.shape[0], batch_size):
            batch = self.model(normalized[start : start + batch_size], training=False)
            batches.append(np.asarray(batch))
        prediction = np.concatenate(batches, axis=0)
        prediction = denormalize_hr(prediction, self.normalization)
        if original_ndim == 2:
            return prediction[0, ..., 0]
        if original_ndim == 3:
            return prediction[0] if single_channel_image else prediction[..., 0]
        return prediction


def _resolve_weights_path(config: dict[str, Any], repository_root: Path, override: str | Path | None) -> Path:
    if override is not None:
        path = Path(override).expanduser()
        return path.resolve() if path.is_absolute() else (repository_root / path).resolve()

    filename = str(config.get("weights", {}).get("filename", ""))
    if not filename or filename.startswith("TODO"):
        raise FileNotFoundError(
            f"No released weights are configured for {config['name']}. "
            "Pass weights_path explicitly after staging the model artifact."
        )
    return (repository_root / "release_assets" / filename).resolve()


def _resolve_keras_model_path(
    config: dict[str, Any], repository_root: Path, override: str | Path | None
) -> Path:
    if override is not None:
        path = Path(override).expanduser()
        return path.resolve() if path.is_absolute() else (repository_root / path).resolve()

    filename = str(config.get("keras_model", {}).get("filename", ""))
    if not filename or filename.startswith("TODO"):
        raise FileNotFoundError(
            f"No full Keras model is configured for {config['name']}. "
            "Pass model_path explicitly after staging the model artifact."
        )
    return (repository_root / "release_assets" / filename).resolve()


def _verify_sha256(path: Path, expected: str | None) -> None:
    if not expected or expected.startswith("TODO"):
        return
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != expected.lower():
        raise ValueError(f"SHA-256 mismatch for {path.name}: expected {expected}, found {actual}")


def load_generator(
    config_path: str | Path,
    *,
    weights_path: str | Path | None = None,
    repository_root: str | Path | None = None,
    verify_checksum: bool = True,
) -> ModelBundle:
    """Load an MW-SR or MW-SR-GAN generator from its release config and weights.

    MW-SR-GAN uses the same generator topology. Batch normalization is a
    discriminator training option and is therefore not added to the generator.
    """

    config_path = Path(config_path).expanduser().resolve()
    config = load_model_config(config_path)
    root = Path(repository_root).expanduser().resolve() if repository_root else find_repository_root(config_path)
    resolved_weights = _resolve_weights_path(config, root, weights_path)
    if not resolved_weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {resolved_weights}")
    if verify_checksum:
        _verify_sha256(resolved_weights, config.get("weights", {}).get("sha256"))

    generator_config = dict(config["generator"])
    generator_config.pop("architecture")
    generator_config["num_rrdb_blocks"] = generator_config.pop("rrdb_blocks")
    generator_config.setdefault("growth_rate", generator_config["filters"])
    model = build_rrdn(scale=int(config["scale"]), channels=int(config["channels"]), **generator_config)
    model.load_weights(resolved_weights)

    stats_value = config.get("normalization", {}).get("stats_file")
    if not stats_value:
        raise ValueError(f"No normalization.stats_file is configured in {config_path}")
    stats_path = Path(stats_value).expanduser()
    if not stats_path.is_absolute():
        stats_path = root / stats_path

    return ModelBundle(
        model=model,
        config=config,
        config_path=config_path,
        weights_path=resolved_weights,
        normalization=load_normalization_stats(stats_path),
    )


def load_keras_generator(
    config_path: str | Path,
    *,
    model_path: str | Path | None = None,
    repository_root: str | Path | None = None,
    verify_checksum: bool = True,
) -> ModelBundle:
    """Load a complete released ``.keras`` generator with its normalization metadata."""

    config_path = Path(config_path).expanduser().resolve()
    config = load_model_config(config_path)
    root = Path(repository_root).expanduser().resolve() if repository_root else find_repository_root(config_path)
    resolved_model = _resolve_keras_model_path(config, root, model_path)
    if not resolved_model.is_file():
        raise FileNotFoundError(f"Full Keras model not found: {resolved_model}")
    if verify_checksum:
        _verify_sha256(resolved_model, config.get("keras_model", {}).get("sha256"))

    stats_value = config.get("normalization", {}).get("stats_file")
    if not stats_value:
        raise ValueError(f"No normalization.stats_file is configured in {config_path}")
    stats_path = Path(stats_value).expanduser()
    if not stats_path.is_absolute():
        stats_path = root / stats_path

    return ModelBundle(
        model=load_model(resolved_model, compile=False),
        config=config,
        config_path=config_path,
        weights_path=resolved_model,
        normalization=load_normalization_stats(stats_path),
    )
