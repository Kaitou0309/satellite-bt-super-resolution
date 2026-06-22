from pathlib import Path

import numpy as np
import pytest
import yaml

from bt_super_resolution import load_generator, load_keras_generator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    REPOSITORY_ROOT / "configs/rrdn_composite_ssim_alpha_0.8.yaml",
    REPOSITORY_ROOT / "configs/rrdn_gan_batchnorm.yaml",
)


@pytest.mark.release
@pytest.mark.parametrize("config_path", CONFIGS)
def test_staged_release_checkpoint_loads_and_upscales(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    weights_path = REPOSITORY_ROOT / "release_assets" / config["weights"]["filename"]
    if not weights_path.is_file():
        pytest.skip("Release checkpoint is distributed separately from Git source.")

    bundle = load_generator(config_path)
    prediction = bundle.predict_kelvin(np.full((2, 3), 250.0, dtype=np.float32))
    assert prediction.shape == (8, 12)
    assert np.isfinite(prediction).all()


@pytest.mark.release
@pytest.mark.parametrize("config_path", CONFIGS)
def test_staged_keras_model_loads_without_custom_objects(config_path: Path) -> None:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_path = REPOSITORY_ROOT / "release_assets" / config["keras_model"]["filename"]
    if not model_path.is_file():
        pytest.skip("Full Keras model is distributed separately from Git source.")

    bundle = load_keras_generator(config_path)
    prediction = bundle.predict_kelvin(np.full((2, 3), 250.0, dtype=np.float32))
    assert prediction.shape == (8, 12)
    assert np.isfinite(prediction).all()
