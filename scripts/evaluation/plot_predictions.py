#!/usr/bin/env python3
"""Plot predictions previously written into HDF5 files by make_prediction.py."""

import argparse
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize

from geo_plotting import SAVED_PREDICTION_NOTE, add_figure_note, load_coordinates, plot_bt_panel


LR_DATASET_CANDIDATES = ("L/bt", "bt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Prediction HDF5 file or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/prediction_plots"))
    parser.add_argument("--scene-index", type=int, default=0)
    parser.add_argument("--lr-key", help="Override LR dataset path; otherwise L/bt and bt are checked.")
    parser.add_argument("--hr-key", default="H/bt")
    parser.add_argument("--prediction-root", default="model_predictions")
    parser.add_argument("--residual-limit", type=float, default=20.0, help="Symmetric residual range in Kelvin.")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--max-files", type=int, help="Plot only the first N sorted files.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Stop when any input file cannot be plotted.")
    return parser.parse_args()


def discover_h5_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}:
        return [path]
    if path.is_dir():
        return sorted((*path.rglob("*.h5"), *path.rglob("*.hdf5")))
    raise FileNotFoundError(f"No HDF5 input found at {path}")


def select_scene(values: np.ndarray, scene_index: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 4:
        return values[scene_index, ..., 0]
    if values.ndim == 3:
        return values[..., 0] if values.shape[-1] == 1 else values[scene_index]
    if values.ndim == 2:
        if scene_index != 0:
            raise IndexError(f"A 2D dataset has only scene index 0, not {scene_index}.")
        return values
    raise ValueError(f"Unsupported dataset shape: {values.shape}")


def find_lr_key(handle: h5py.File, override: str | None) -> str:
    candidates = (override,) if override else LR_DATASET_CANDIDATES
    for candidate in candidates:
        if candidate and candidate in handle:
            return candidate
    raise KeyError(f"No LR dataset found; checked {candidates}.")


def load_plot_data(path: Path, args: argparse.Namespace):
    with h5py.File(path, "r") as handle:
        lr = select_scene(handle[find_lr_key(handle, args.lr_key)][:], args.scene_index)
        hr = select_scene(handle[args.hr_key][:], args.scene_index) if args.hr_key in handle else None
        if args.prediction_root not in handle:
            raise KeyError(f"Missing prediction group {args.prediction_root!r}.")

        predictions = []
        root = handle[args.prediction_root]
        for group_name in sorted(root.keys()):
            group = root[group_name]
            if "bt" not in group:
                continue
            dataset = group["bt"]
            label = dataset.attrs.get("model_name", group_name.replace("_", " "))
            if isinstance(label, bytes):
                label = label.decode("utf-8")
            units = dataset.attrs.get("units", "")
            if isinstance(units, bytes):
                units = units.decode("utf-8")
            if units and str(units).lower() not in {"k", "kelvin"}:
                print(f"Warning: {path}:{dataset.name} reports units={units!r}; plots assume Kelvin.")
            predictions.append((str(label), select_scene(dataset[:], args.scene_index)))

        target_shape = hr.shape if hr is not None else predictions[0][1].shape if predictions else None
        coordinates = (
            load_coordinates(handle, target_shape, args.scene_index, preferred_groups=("H", "L", ""))
            if target_shape is not None
            else None
        )

    if not predictions:
        raise ValueError("No model_predictions/<model>/bt datasets were found.")
    return lr, hr, predictions, coordinates


def resized_lr(lr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    return resize(
        lr,
        target_shape,
        order=1,
        mode="edge",
        preserve_range=True,
        anti_aliasing=False,
    ).astype(np.float32)


def bt_limits(images: list[np.ndarray]) -> tuple[float, float]:
    finite = [image[np.isfinite(image)] for image in images if np.any(np.isfinite(image))]
    if not finite:
        raise ValueError("No finite brightness-temperature values are available.")
    return tuple(float(value) for value in np.percentile(np.concatenate(finite), (1, 99)))


def plot_with_truth(path: Path, output: Path, lr: np.ndarray, hr: np.ndarray, predictions, coordinates, args) -> None:
    lr_up = resized_lr(lr, hr.shape)
    vmin, vmax = bt_limits([lr_up, hr, *(prediction for _, prediction in predictions)])
    figure, axes = plt.subplots(len(predictions), 4, figsize=(17, 4.3 * len(predictions)), squeeze=False)

    for row, (label, prediction) in enumerate(predictions):
        if prediction.shape != hr.shape:
            raise ValueError(f"Prediction {label!r} has shape {prediction.shape}; HR has shape {hr.shape}.")
        panels = (lr_up, prediction, hr, prediction - hr)
        titles = ("Bilinear LR", label, "Ground Truth", "Residual (SR - HR)")
        for column, (panel, title) in enumerate(zip(panels, titles)):
            is_residual = column == 3
            image = plot_bt_panel(
                axes[row, column],
                panel,
                title,
                coordinates=coordinates,
                cmap="coolwarm" if is_residual else "turbo",
                vmin=-args.residual_limit if is_residual else vmin,
                vmax=args.residual_limit if is_residual else vmax,
            )
            figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.04, label="K")

    figure.suptitle(f"{path.name}, scene {args.scene_index}", fontsize=15, fontweight="bold")
    add_figure_note(figure, SAVED_PREDICTION_NOTE)
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def plot_without_truth(path: Path, output: Path, lr: np.ndarray, predictions, coordinates, args) -> None:
    target_shape = predictions[0][1].shape
    lr_up = resized_lr(lr, target_shape)
    vmin, vmax = bt_limits([lr_up, *(prediction for _, prediction in predictions)])
    panels = [("Bilinear LR", lr_up), *predictions]
    figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.7), squeeze=False)
    for axis, (title, panel) in zip(axes[0], panels):
        image = plot_bt_panel(
            axis,
            panel,
            title,
            coordinates=coordinates,
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="K")
    figure.suptitle(f"{path.name}, scene {args.scene_index}", fontsize=15, fontweight="bold")
    add_figure_note(figure, SAVED_PREDICTION_NOTE)
    figure.tight_layout(rect=(0, 0.04, 1, 0.96))
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    files = discover_h5_files(args.input)
    if args.max_files is not None:
        if args.max_files < 1:
            raise ValueError("--max-files must be at least 1.")
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0
    for index, path in enumerate(files, start=1):
        output = args.output_dir / f"{path.stem}_scene_{args.scene_index}_predictions.png"
        if output.exists() and not args.overwrite:
            print(f"Keeping existing [{index}/{len(files)}] {output}")
            continue
        try:
            lr, hr, predictions, coordinates = load_plot_data(path, args)
            if hr is None:
                plot_without_truth(path, output, lr, predictions, coordinates, args)
            else:
                plot_with_truth(path, output, lr, hr, predictions, coordinates, args)
            plotted += 1
            print(f"Saved [{index}/{len(files)}] {output}")
        except (OSError, KeyError, IndexError, ValueError) as error:
            if args.strict:
                raise
            print(f"Skipping {path}: {error}")
    print(f"Completed: created {plotted} plot(s) from {len(files)} HDF5 file(s).")


if __name__ == "__main__":
    main()
