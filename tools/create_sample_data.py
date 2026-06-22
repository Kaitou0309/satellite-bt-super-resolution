#!/usr/bin/env python3
"""Create a compact paired LR/HR example from one project HDF5 scene."""

import argparse
from pathlib import Path

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("sample_data/amsr2_example.h5"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.input, "r") as source, h5py.File(args.output, "w") as target:
        target.attrs["title"] = "AMSR2-derived brightness-temperature super-resolution example"
        target.attrs["description"] = "One paired 89 GHz LR/HR scene for software validation, not model training."
        target.attrs["source_file"] = args.input.name
        target.attrs["spatial_scale"] = 4
        for key in ("L/bt", "H/bt"):
            if key not in source:
                raise KeyError(f"Required dataset {key!r} not found in {args.input}")
            source_dataset = source[key]
            output_dataset = target.create_dataset(
                key,
                data=source_dataset[:],
                compression="gzip",
                compression_opts=9,
                shuffle=True,
            )
            for attribute, value in source_dataset.attrs.items():
                output_dataset.attrs[attribute] = value
        target["L"].attrs["role"] = "low-resolution model input"
        target["H"].attrs["role"] = "high-resolution evaluation target"
    print(f"Created {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
