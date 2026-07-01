from pathlib import Path
import json

import numpy as np
import yaml

from bt_super_resolution import load_generator
from bt_super_resolution.models import build_rrdn


def test_config_driven_loader_and_kelvin_prediction(tmp_path: Path) -> None:
    weights_path = tmp_path / "tiny.weights.h5"
    model = build_rrdn(
        scale=4,
        channels=1,
        num_rrdb_blocks=1,
        rdb_per_rrdb=1,
        conv_layers_per_rdb=1,
        filters=8,
        growth_rate=8,
    )
    model.save_weights(weights_path)

    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    (metadata_dir / "stats.json").write_text(
        json.dumps({"mu_X": 250.0, "sd_X": 30.0, "mu_Y": 250.0, "sd_Y": 30.0}),
        encoding="utf-8",
    )

    config = {
        "name": "tiny_test_rrdn",
        "display_name": "Tiny test RRDN",
        "scale": 4,
        "channels": 1,
        "generator": {
            "architecture": "rrdn",
            "rrdb_blocks": 1,
            "rdb_per_rrdb": 1,
            "conv_layers_per_rdb": 1,
            "filters": 8,
            "residual_scaling": 0.2,
            "upsampling": "bilinear",
        },
        "normalization": {"stats_file": "metadata/stats.json"},
        "weights": {"filename": weights_path.name, "sha256": "TODO"},
    }
    config_path = tmp_path / "model.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    bundle = load_generator(
        config_path,
        weights_path=weights_path,
        repository_root=tmp_path,
        verify_checksum=False,
    )
    prediction = bundle.predict_kelvin(np.full((4, 5), 250.0, dtype=np.float32))

    assert bundle.name == "Tiny test RRDN"
    assert prediction.shape == (16, 20)
    assert np.isfinite(prediction).all()
