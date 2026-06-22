#!/usr/bin/env python3
"""Copy HDF5 inputs and append predictions from configured release models."""

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import h5py

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from bt_super_resolution import load_generator, load_keras_generator


DEFAULT_CONFIGS = (
    REPOSITORY_ROOT / "configs/rrdn_composite_ssim_alpha_0.8.yaml",
    REPOSITORY_ROOT / "configs/rrdn_gan_batchnorm.yaml",
)
LR_DATASET_CANDIDATES = ("L/bt", "bt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input HDF5 file or directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output HDF5 file or directory.")
    parser.add_argument("--config", action="append", type=Path, help="Model YAML; repeat for multiple models.")
    parser.add_argument("--weights", action="append", type=Path, help="Checkpoint matching each --config.")
    parser.add_argument("--keras-model", action="append", type=Path, help="Full model matching each --config.")
    parser.add_argument(
        "--artifact-format",
        choices=("weights", "keras"),
        default="weights",
        help="Use reconstructed .weights.h5 checkpoints or complete .keras models.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Stop if a model artifact is unavailable.")
    return parser.parse_args()


def discover_h5_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted((*path.rglob("*.h5"), *path.rglob("*.hdf5")))


def output_path(source: Path, input_root: Path, output_root: Path) -> Path:
    if input_root.is_file():
        return output_root if output_root.suffix.lower() in {".h5", ".hdf5"} else output_root / source.name
    return output_root / source.relative_to(input_root)


def lr_dataset(handle: h5py.File) -> str:
    for candidate in LR_DATASET_CANDIDATES:
        if candidate in handle:
            return candidate
    raise KeyError(f"None of {LR_DATASET_CANDIDATES} is present in the HDF5 file.")


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def load_bundles(args: argparse.Namespace):
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
    bundles = []
    for index, config in enumerate(configs):
        try:
            if args.artifact_format == "keras":
                bundles.append(
                    load_keras_generator(
                        config,
                        model_path=keras_models[index] if keras_models else None,
                    )
                )
            else:
                bundles.append(load_generator(config, weights_path=weights[index] if weights else None))
        except (FileNotFoundError, ValueError) as error:
            if args.strict:
                raise
            print(f"Skipping {config}: {error}")
    if not bundles:
        raise RuntimeError("No model could be loaded.")
    return bundles


def main() -> None:
    args = parse_args()
    sources = discover_h5_files(args.input)
    if not sources:
        raise FileNotFoundError(f"No HDF5 files found under {args.input}")
    bundles = load_bundles(args)

    for source in sources:
        destination = output_path(source, args.input, args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve() and not destination.exists():
            shutil.copy2(source, destination)
        with h5py.File(source, "r") as source_h5:
            lr = source_h5[lr_dataset(source_h5)][:]

        with h5py.File(destination, "a") as output_h5:
            root = output_h5.require_group("model_predictions")
            for bundle in bundles:
                group = root.require_group(safe_name(bundle.name))
                if "bt" in group:
                    if not args.overwrite:
                        print(f"Keeping existing {destination}:{group.name}/bt")
                        continue
                    del group["bt"]
                prediction = bundle.predict_kelvin(lr, batch_size=args.batch_size)
                dataset = group.create_dataset("bt", data=prediction, compression="gzip")
                dataset.attrs["model_config"] = str(bundle.config_path)
                dataset.attrs["model_name"] = bundle.name
                dataset.attrs["artifact_file"] = bundle.weights_path.name
                dataset.attrs["artifact_format"] = args.artifact_format
                if args.artifact_format == "weights":
                    dataset.attrs["weights_file"] = bundle.weights_path.name
                else:
                    dataset.attrs["keras_model_file"] = bundle.weights_path.name
                dataset.attrs["units"] = "K"
                print(f"Wrote {destination}:{group.name}/bt {prediction.shape}")


if __name__ == "__main__":
    main()
