#!/usr/bin/env python3
"""Evaluate release generators across one HDF5 file or a directory tree."""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bt_super_resolution import load_generator, load_keras_generator


DEFAULT_CONFIGS = (
    REPOSITORY_ROOT / "configs/rrdn_composite_ssim_alpha_0.8.yaml",
    REPOSITORY_ROOT / "configs/rrdn_gan_batchnorm.yaml",
)
METRIC_COLUMNS = ("RMSE", "Global PSNR", "SSIM", "Bias", "Latency (ms/scene)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="An HDF5 file or directory searched recursively.")
    parser.add_argument("--lr-key", default="L/bt")
    parser.add_argument("--hr-key", default="H/bt")
    parser.add_argument("--config", action="append", type=Path)
    parser.add_argument("--weights", action="append", type=Path)
    parser.add_argument("--keras-model", action="append", type=Path)
    parser.add_argument(
        "--artifact-format",
        choices=("weights", "keras"),
        default="weights",
        help="Evaluate reconstructed .weights.h5 checkpoints or complete .keras models.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("metrics.csv"))
    parser.add_argument("--plot-output", type=Path, help="Defaults to <output stem>_matrix.png.")
    parser.add_argument("--strict", action="store_true", help="Stop on an unavailable model or invalid HDF5 file.")
    return parser.parse_args()


def discover_h5_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}:
        return [path]
    if path.is_dir():
        return sorted((*path.rglob("*.h5"), *path.rglob("*.hdf5")))
    raise FileNotFoundError(f"Data path does not contain HDF5 data: {path}")


def as_4d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 2:
        return values[None, ..., None]
    if values.ndim == 3:
        return values[None, ...] if values.shape[-1] == 1 else values[..., None]
    if values.ndim == 4:
        return values
    raise ValueError(f"Expected a 2D, 3D, or 4D array; received {values.shape}")


def read_pair(path: Path, lr_key: str, hr_key: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        if lr_key not in handle or hr_key not in handle:
            raise KeyError(f"Required datasets {lr_key!r} and {hr_key!r} were not both found.")
        return handle[lr_key][:], handle[hr_key][:]


def scan_files(files: list[Path], args: argparse.Namespace) -> tuple[list[Path], float, int]:
    valid_files = []
    hr_min, hr_max = float("inf"), float("-inf")
    scene_count = 0
    for file_index, path in enumerate(files, start=1):
        try:
            _, hr = read_pair(path, args.lr_key, args.hr_key)
            hr = as_4d(hr)
            finite = hr[np.isfinite(hr)]
            if finite.size == 0:
                raise ValueError("HR data contain no finite values.")
            hr_min = min(hr_min, float(np.min(finite)))
            hr_max = max(hr_max, float(np.max(finite)))
            scene_count += hr.shape[0]
            valid_files.append(path)
            print(f"Validated [{file_index}/{len(files)}] {path}")
        except (OSError, KeyError, ValueError) as error:
            if args.strict:
                raise
            print(f"Skipping invalid data file {path}: {error}")
    if not valid_files:
        raise RuntimeError("No valid paired HDF5 files were found.")
    data_range = hr_max - hr_min
    if data_range <= 0:
        raise ValueError("The global HR Kelvin range must be positive.")
    return valid_files, data_range, scene_count


@dataclass
class MetricAccumulator:
    squared_error_sum: float = 0.0
    error_sum: float = 0.0
    valid_pixel_count: int = 0
    ssim_sum: float = 0.0
    ssim_count: int = 0
    inference_seconds: float = 0.0
    scene_count: int = 0
    file_count: int = 0

    def update(self, hr: np.ndarray, sr: np.ndarray, data_range: float, elapsed: float) -> None:
        hr, sr = as_4d(hr), as_4d(sr)
        if hr.shape != sr.shape:
            raise ValueError(f"HR and SR shapes differ: {hr.shape} versus {sr.shape}")
        valid = np.isfinite(hr) & np.isfinite(sr)
        if not np.any(valid):
            raise ValueError("No finite HR/SR pixel pairs are available.")
        error = (sr[valid] - hr[valid]).astype(np.float64)
        self.squared_error_sum += float(np.sum(error * error))
        self.error_sum += float(np.sum(error))
        self.valid_pixel_count += int(error.size)

        for index in range(hr.shape[0]):
            target, prediction = hr[index, ..., 0], sr[index, ..., 0]
            scene_valid = np.isfinite(target) & np.isfinite(prediction)
            if not np.any(scene_valid):
                continue
            fill = float(np.mean(target[scene_valid]))
            target = np.where(scene_valid, target, fill)
            prediction = np.where(scene_valid, prediction, fill)
            self.ssim_sum += float(structural_similarity(target, prediction, data_range=data_range))
            self.ssim_count += 1

        self.inference_seconds += elapsed
        self.scene_count += hr.shape[0]
        self.file_count += 1

    def finalize(self) -> dict[str, float]:
        if self.valid_pixel_count == 0 or self.ssim_count == 0 or self.scene_count == 0:
            raise RuntimeError("No valid predictions were accumulated.")
        mse = self.squared_error_sum / self.valid_pixel_count
        return {
            "RMSE": float(np.sqrt(mse)),
            "Global PSNR": float(10.0 * np.log10(self.data_range**2 / mse)) if mse > 0 else float("inf"),
            "SSIM": self.ssim_sum / self.ssim_count,
            "Bias": self.error_sum / self.valid_pixel_count,
            "Latency (ms/scene)": 1000.0 * self.inference_seconds / self.scene_count,
        }

    data_range: float = 1.0


def save_matrix_plot(rows: list[dict[str, float]], path: Path, file_count: int, scene_count: int) -> None:
    cell_text = [[f"{float(row[column]):.4f}" for column in METRIC_COLUMNS] for row in rows]
    row_labels = [str(row["Model"]) for row in rows]
    figure, axis = plt.subplots(figsize=(14, max(3.2, 2.2 + 0.65 * len(rows))))
    axis.axis("off")
    table = axis.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=METRIC_COLUMNS,
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.05, 1.55)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#dce8f2")
    axis.set_title(
        f"AMSR2 Model Comparison Matrix ({file_count} files, {scene_count} scenes)",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    discovered = discover_h5_files(args.data)
    print(f"Discovered {len(discovered)} HDF5 file(s) under {args.data}")
    files, data_range, total_scenes = scan_files(discovered, args)
    print(f"Evaluating {len(files)} valid files / {total_scenes} scenes; global HR range={data_range:.4f} K")

    configs = args.config or list(DEFAULT_CONFIGS)
    weights = args.weights or []
    keras_models = args.keras_model or []
    if weights and len(weights) != len(configs):
        raise ValueError("Provide one --weights value for every --config, or omit --weights.")
    if keras_models and len(keras_models) != len(configs):
        raise ValueError("Provide one --keras-model value for every --config, or omit --keras-model.")
    if args.artifact_format == "weights" and keras_models:
        raise ValueError("--keras-model requires --artifact-format keras.")
    if args.artifact_format == "keras" and weights:
        raise ValueError("--weights cannot be used with --artifact-format keras.")

    rows = []
    for config_index, config in enumerate(configs):
        try:
            if args.artifact_format == "keras":
                bundle = load_keras_generator(
                    config,
                    model_path=keras_models[config_index] if keras_models else None,
                )
            else:
                bundle = load_generator(config, weights_path=weights[config_index] if weights else None)
        except (FileNotFoundError, ValueError) as error:
            if args.strict:
                raise
            print(f"Skipping {config}: {error}")
            continue

        accumulator = MetricAccumulator(data_range=data_range)
        print(f"\nEvaluating {bundle.name} from {args.artifact_format}")
        for file_index, path in enumerate(files, start=1):
            try:
                lr, hr = read_pair(path, args.lr_key, args.hr_key)
                start = time.perf_counter()
                prediction = bundle.predict_kelvin(lr, batch_size=args.batch_size)
                accumulator.update(hr, prediction, data_range, time.perf_counter() - start)
                print(f"  Processed [{file_index}/{len(files)}] {path.name}")
            except (OSError, KeyError, ValueError) as error:
                if args.strict:
                    raise
                print(f"  Skipping {path}: {error}")

        metrics = accumulator.finalize()
        row = {"Model": bundle.name, **metrics}
        rows.append(row)
        print(f"Completed {bundle.name}: {accumulator.file_count}/{len(files)} files, {accumulator.scene_count} scenes")
        print(row)

    if not rows:
        raise RuntimeError("No model was evaluated.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("Model", *METRIC_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)

    plot_output = args.plot_output or args.output.with_name(f"{args.output.stem}_matrix.png")
    save_matrix_plot(rows, plot_output, len(files), total_scenes)
    print(f"\nSaved CSV: {args.output}")
    print(f"Saved matrix plot: {plot_output}")


if __name__ == "__main__":
    main()
