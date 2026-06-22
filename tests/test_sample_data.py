from pathlib import Path

import h5py


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_public_example_has_expected_schema_and_scale() -> None:
    with h5py.File(REPOSITORY_ROOT / "sample_data/amsr2_example.h5", "r") as handle:
        assert handle["L/bt"].shape == (96, 100)
        assert handle["H/bt"].shape == (384, 400)
        assert handle["L/bt"].attrs["units"] == "K"
        assert handle["H/bt"].attrs["units"] == "K"
