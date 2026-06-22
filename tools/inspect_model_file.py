"""Inspect a Keras HDF5 model or weights checkpoint without loading TensorFlow."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py


DEFAULT_PATH = Path(
    "release_assets/"
    "bt-sr-rrdn-gan-9rrdb-bn-generator-v0.1.0.weights.h5"
)


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} GiB"


def inspect_h5(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    dataset_count = 0
    stored_values = 0
    stored_bytes = 0
    sample_datasets: list[tuple[str, tuple[int, ...], str]] = []

    with h5py.File(path, "r") as handle:
        root_attributes = sorted(handle.attrs.keys())
        root_groups = sorted(handle.keys())

        def collect(name: str, obj: h5py.Dataset | h5py.Group) -> None:
            nonlocal dataset_count, stored_values, stored_bytes
            if not isinstance(obj, h5py.Dataset):
                return
            dataset_count += 1
            stored_values += int(obj.size)
            stored_bytes += int(obj.size * obj.dtype.itemsize)
            if len(sample_datasets) < 20:
                sample_datasets.append((name, tuple(obj.shape), str(obj.dtype)))

        handle.visititems(collect)
        has_model_config = "model_config" in handle.attrs
        is_weights_only = path.name.endswith(".weights.h5") or not has_model_config

    print(f"Path: {path}")
    print(f"File size: {format_bytes(path.stat().st_size)}")
    print(f"Detected type: {'weights-only checkpoint' if is_weights_only else 'legacy full Keras model'}")
    print(f"Root attributes: {root_attributes or '[none]'}")
    print(f"Root groups: {root_groups or '[none]'}")
    print(f"Stored datasets: {dataset_count}")
    print(f"Stored numeric values: {stored_values:,}")
    print(f"Approximate tensor storage: {format_bytes(stored_bytes)}")

    if sample_datasets:
        print("\nFirst stored tensors:")
        for name, shape, dtype in sample_datasets:
            print(f"  {name}: shape={shape}, dtype={dtype}")

    print("\nRecommended loading method:")
    if is_weights_only:
        print("  1. Rebuild the exact generator architecture.")
        print(f"  2. Run: model.load_weights({str(path)!r})")
        print("  custom_objects is not needed for load_weights().")
        print("  A model summary is unavailable until the architecture is rebuilt.")
    else:
        print("  Use tf.keras.models.load_model(path, compile=False).")
        print("  Supply custom_objects only if the model contains unregistered custom layers.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect whether an HDF5 checkpoint stores a full model or weights only."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_PATH,
        help=f"HDF5 file to inspect (default: {DEFAULT_PATH})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    inspect_h5(parse_args().path)
