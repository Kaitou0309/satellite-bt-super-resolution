#!/usr/bin/env python3
"""Plot LR context, prediction, ground truth, and residuals for release models."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bt_super_resolution import load_generator


DEFAULT_CONFIGS = (
    REPOSITORY_ROOT / "configs/rrdn_composite_ssim_alpha_0.8.yaml",
    REPOSITORY_ROOT / "configs/rrdn_gan_batchnorm.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, action="append", type=Path)
    parser.add_argument("--lr-key", default="L/bt")
    parser.add_argument("--hr-key", default="H/bt")
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--config", action="append", type=Path)
    parser.add_argument("--weights", action="append", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("selected_model_residual_plots"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--residual-limit", type=float, default=20.0)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def select_scene(values: np.ndarray, index: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 4:
        return values[index, ..., 0]
    if values.ndim == 3:
        return values[..., 0] if values.shape[-1] == 1 else values[index]
    if values.ndim == 2:
        return values
    raise ValueError(f"Unsupported scene array shape: {values.shape}")


def main() -> None:
    args = parse_args()
    configs = args.config or list(DEFAULT_CONFIGS)
    weights = args.weights or []
    if weights and len(weights) != len(configs):
        raise ValueError("Provide one --weights value for every --config, or omit --weights.")
    bundles = []
    for index, config in enumerate(configs):
        try:
            bundles.append(load_generator(config, weights_path=weights[index] if weights else None))
        except (FileNotFoundError, ValueError) as error:
            if args.strict:
                raise
            print(f"Skipping {config}: {error}")
    if not bundles:
        raise RuntimeError("No model could be loaded.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for input_path in args.input:
        with h5py.File(input_path, "r") as handle:
            lr = select_scene(handle[args.lr_key][:], args.scene_index)
            hr = select_scene(handle[args.hr_key][:], args.scene_index)
        lr_up = tf.image.resize(lr[None, ..., None], hr.shape, method="bilinear").numpy()[0, ..., 0]
        predictions = [bundle.predict_kelvin(lr, batch_size=args.batch_size) for bundle in bundles]
        finite_bt = np.concatenate([image[np.isfinite(image)] for image in (hr, lr_up, *predictions)])
        bt_min, bt_max = np.percentile(finite_bt, (1, 99))

        figure, axes = plt.subplots(len(bundles), 4, figsize=(16, 4.2 * len(bundles)), squeeze=False)
        for row, (bundle, prediction) in enumerate(zip(bundles, predictions)):
            panels = (lr_up, prediction, hr, prediction - hr)
            titles = ("Bilinear LR", bundle.name, "Ground Truth", "Residual (SR - HR)")
            for column, (panel, title) in enumerate(zip(panels, titles)):
                residual = column == 3
                image = axes[row, column].imshow(
                    panel,
                    cmap="coolwarm" if residual else "turbo",
                    vmin=-args.residual_limit if residual else bt_min,
                    vmax=args.residual_limit if residual else bt_max,
                )
                axes[row, column].set_title(title, fontsize=11)
                axes[row, column].set_axis_off()
                figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.04, label="K")
        figure.suptitle(f"{input_path.name}, scene {args.scene_index}", fontsize=14, fontweight="bold")
        figure.tight_layout()
        output = args.output_dir / f"{input_path.stem}_scene_{args.scene_index}_residuals.png"
        figure.savefig(output, dpi=600, bbox_inches="tight")
        plt.close(figure)
        print(f"Saved {output}")


if __name__ == "__main__":
    main()
