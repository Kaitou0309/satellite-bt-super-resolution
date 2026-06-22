#!/usr/bin/env python3
"""Inspect HDF5 structure and verify saved prediction dimensions."""

import argparse
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py


LR_DATASET_CANDIDATES = ("L/bt", "bt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="An HDF5 file or directory searched recursively.")
    parser.add_argument("--log", type=Path, default=Path("inspect_h5_files.log"))
    parser.add_argument("--max-files", type=int, help="Inspect only the first N sorted files.")
    parser.add_argument("--lr-key", help="Override LR dataset path; otherwise L/bt and bt are checked.")
    parser.add_argument("--hr-key", default="H/bt")
    parser.add_argument("--prediction-root", default="model_predictions")
    parser.add_argument("--no-attrs", action="store_true")
    parser.add_argument("--summary-only", action="store_true", help="Skip the full HDF5 tree.")
    parser.add_argument("--strict", action="store_true", help="Exit with an error on unreadable files or shape mismatches.")
    return parser.parse_args()


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> None:
        for stream in self.streams:
            stream.write(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@dataclass
class InspectionTotals:
    inspected_files: int = 0
    failed_files: int = 0
    prediction_datasets: int = 0
    shape_mismatches: int = 0


def collect_h5_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in {".h5", ".hdf5"}:
        return [path]
    if path.is_dir():
        return sorted((*path.rglob("*.h5"), *path.rglob("*.hdf5")))
    raise FileNotFoundError(f"No HDF5 input found at {path}")


def print_attrs(obj, *, indent: str, enabled: bool) -> None:
    if not enabled or not obj.attrs:
        return
    print(f"{indent}attributes:")
    for key, value in obj.attrs.items():
        print(f"{indent}  - {key}: {value}")


def print_dataset_variables(handle: h5py.File) -> None:
    variables = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            variables.append((name, obj.shape, obj.dtype))

    handle.visititems(visitor)
    print("\nDataset variables:")
    if not variables:
        print("  No datasets found.")
    for name, shape, dtype in variables:
        print(f"  - {name} | shape={shape} | dtype={dtype}")


def print_full_tree(handle: h5py.File, *, print_attributes: bool) -> None:
    print("\nFull HDF5 tree:")

    def visitor(name, obj):
        indent = "  " * name.count("/")
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}- DATASET: {name} | shape={obj.shape} | dtype={obj.dtype}")
        else:
            print(f"{indent}- GROUP:   {name}")
        print_attrs(obj, indent=indent + "  ", enabled=print_attributes)

    handle.visititems(visitor)


def spatial_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 2:
        return shape
    if len(shape) == 3:
        return shape[:2] if shape[-1] == 1 else shape[-2:]
    if len(shape) == 4:
        return shape[-3:-1]
    raise ValueError(f"Cannot infer spatial dimensions from shape {shape}")


def scene_count(shape: tuple[int, ...]) -> int:
    if len(shape) == 2:
        return 1
    if len(shape) == 3:
        return 1 if shape[-1] == 1 else shape[0]
    if len(shape) == 4:
        return shape[0]
    raise ValueError(f"Cannot infer scene count from shape {shape}")


def find_lr_key(handle: h5py.File, override: str | None) -> str | None:
    candidates = (override,) if override else LR_DATASET_CANDIDATES
    return next((candidate for candidate in candidates if candidate and candidate in handle), None)


def print_prediction_verification(handle: h5py.File, args: argparse.Namespace, totals: InspectionTotals) -> None:
    print("\nPrediction shape verification:")
    lr_key = find_lr_key(handle, args.lr_key)
    hr_dataset = handle[args.hr_key] if args.hr_key in handle else None

    if lr_key:
        lr_shape = handle[lr_key].shape
        print(f"  LR: {lr_key} shape={lr_shape}, spatial={spatial_shape(lr_shape)}")
    else:
        lr_shape = None
        print("  LR: not found")

    if hr_dataset is not None:
        hr_shape = hr_dataset.shape
        hr_spatial = spatial_shape(hr_shape)
        hr_scenes = scene_count(hr_shape)
        print(f"  HR: {args.hr_key} shape={hr_shape}, spatial={hr_spatial}, scenes={hr_scenes}")
        if lr_shape is not None:
            lr_spatial = spatial_shape(lr_shape)
            scale = (hr_spatial[0] / lr_spatial[0], hr_spatial[1] / lr_spatial[1])
            print(f"  HR/LR spatial scale: {scale[0]:g} x {scale[1]:g}")
    else:
        hr_shape = None
        hr_spatial = None
        hr_scenes = None
        print(f"  HR: {args.hr_key} not found; predictions cannot be compared with target dimensions.")

    if args.prediction_root not in handle:
        print(f"  Predictions: group {args.prediction_root!r} not found.")
        return

    prediction_root = handle[args.prediction_root]
    found = False
    for model_group_name in sorted(prediction_root.keys()):
        model_group = prediction_root[model_group_name]
        if "bt" not in model_group:
            print(f"  - {model_group_name}: missing bt dataset")
            continue
        found = True
        dataset = model_group["bt"]
        totals.prediction_datasets += 1
        prediction_spatial = spatial_shape(dataset.shape)
        prediction_scenes = scene_count(dataset.shape)
        model_name = dataset.attrs.get("model_name", model_group_name)
        if isinstance(model_name, bytes):
            model_name = model_name.decode("utf-8")

        reasons = []
        if hr_spatial is not None and prediction_spatial != hr_spatial:
            reasons.append(f"spatial {prediction_spatial} != HR {hr_spatial}")
        if hr_scenes is not None and prediction_scenes != hr_scenes:
            reasons.append(f"scenes {prediction_scenes} != HR {hr_scenes}")
        if reasons:
            totals.shape_mismatches += 1
            status = "MISMATCH: " + "; ".join(reasons)
        elif hr_spatial is not None:
            status = "MATCH: full HR spatial dimensions preserved"
        else:
            status = "UNVERIFIED: no HR target"
        print(f"  - {model_name}: shape={dataset.shape}, spatial={prediction_spatial} | {status}")

    if not found:
        print("  Predictions: no model_predictions/<model>/bt datasets found.")


def inspect_file(path: Path, args: argparse.Namespace, totals: InspectionTotals) -> None:
    print("\n" + "=" * 100)
    print(f"FILE: {path}")
    print("=" * 100)
    try:
        with h5py.File(path, "r") as handle:
            totals.inspected_files += 1
            print("\nTop-level keys:")
            for key in handle.keys():
                print(f"  - {key}")
            print_attrs(handle, indent="  ", enabled=not args.no_attrs)
            print_dataset_variables(handle)
            if not args.summary_only:
                print_full_tree(handle, print_attributes=not args.no_attrs)
            print_prediction_verification(handle, args, totals)
    except (OSError, KeyError, ValueError) as error:
        totals.failed_files += 1
        print(f"FAILED to inspect {path}: {error}")
        if args.strict:
            raise


def main() -> None:
    args = parse_args()
    files = collect_h5_files(args.input)
    if args.max_files is not None:
        if args.max_files < 1:
            raise ValueError("--max-files must be at least 1.")
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No .h5/.hdf5 files found under {args.input}")

    args.log.parent.mkdir(parents=True, exist_ok=True)
    totals = InspectionTotals()
    with args.log.open("w", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)):
            print("=" * 100)
            print("HDF5 INSPECTION")
            print("=" * 100)
            print(f"Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Input:         {args.input}")
            print(f"Log:           {args.log}")
            print(f"Files selected: {len(files)}")
            for index, path in enumerate(files, start=1):
                print(f"  {index:03d}. {path}")
            for path in files:
                inspect_file(path, args, totals)

            print("\n" + "=" * 100)
            print("SUMMARY")
            print("=" * 100)
            print(f"Files inspected:      {totals.inspected_files}")
            print(f"Files failed:         {totals.failed_files}")
            print(f"Prediction datasets:  {totals.prediction_datasets}")
            print(f"Shape mismatches:     {totals.shape_mismatches}")
            print(f"Log written to:       {args.log}")

    if args.strict and (totals.failed_files or totals.shape_mismatches):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
