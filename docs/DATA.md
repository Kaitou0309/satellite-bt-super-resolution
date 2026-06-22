# Dataset Documentation

## Scientific purpose

The project learns a four-times spatial super-resolution mapping for 89 GHz microwave brightness-temperature imagery. Paired data are derived from Advanced Microwave Scanning Radiometer 2 (AMSR2) observations so that the low-resolution (LR) input and high-resolution (HR) target describe the same underlying scene.

## Pair construction

AMSR2 provides the high-resolution source observations used by the simulator. The research workflow convolves those observations with normalized Gaussian antenna response functions to represent two microwave-sounder configurations:

- **LR input:** nominal 2.2 degree field of view, 1.1 degree scan step, 96 scan samples, and an 8/3 second scan period.
- **HR target:** nominal 0.55 degree field of view with four-times denser scan sampling and adjusted along-track sampling.

The LR patches have shape `96 x 100`; paired HR patches have shape `384 x 400`. This is a four-times enhancement in both spatial dimensions.

The HR fields are physically based simulations of how the same scene could appear with a smaller antenna footprint and denser sampling. They are not observations from an existing high-resolution microwave sounder. Consequently, model output should be interpreted as a learned reconstruction against an idealized simulated target, not as a direct measurement.

## Coverage

The research dataset was generated from global AMSR2 observations collected between July 2025 and May 2026. It covers ocean, land, coastlines, precipitation systems, and varied atmospheric conditions. Tropical cyclones and other intense tropical-weather systems were deliberately represented to improve coverage of eyewalls, organized convection, and rainbands.

Exact train/evaluation split counts and the permanent dataset DOI will be added when the complete dataset archive is released.

## HDF5 schema

```text
L/bt    float32 low-resolution brightness temperature, Kelvin
H/bt    float32 high-resolution target brightness temperature, Kelvin
```

Research files may additionally contain latitude and longitude fields. The public software requires only `L/bt` for inference and both `L/bt` and `H/bt` for quantitative evaluation.

## Normalization

The selected checkpoints require the statistics in `metadata/unified_global_stats.npz`:

```text
mu_X, sd_X    LR mean and standard deviation
mu_Y, sd_Y    HR mean and standard deviation
```

Inputs are normalized before inference and predictions are denormalized back to Kelvin. Statistics from a different dataset must not be substituted without retraining or validating the model.

## Public example

`sample_data/amsr2_example.h5` contains one complete paired scene with the production `96 x 100` and `384 x 400` dimensions. It is intended for installation checks, prediction examples, plotting, and tests. It is not large enough or sufficiently diverse for model training.

The entire local `AMSR2/` directory is intentionally excluded from Git. Publishing one compressed scene keeps the source repository small while preserving a realistic end-to-end example. The full dataset should be distributed through a DOI-backed scientific repository or other external object storage.

## Provenance and credit

Original AMSR2 observations are available through the JAXA G-Portal. In accordance with the G-Portal data-credit requirement:

> Original data for this value added data product was provided by Japan Aerospace Exploration Agency.

The paired LR/HR fields and public example are value-added simulated products created by the project workflow. Users should retain the JAXA acknowledgement when redistributing derived data or publishing results.

## Related observations

ATMS observations used for qualitative applications are available through NOAA CLASS. Tomorrow.io observations were supplied through a research collaboration and remain subject to the applicable data-access policy; they are not distributed in this repository.

## Manuscript source

The construction details above are summarized from the project manuscript, *Machine Learning-Based Super-Resolution of Microwave Sounder Window Channel Imagery* (draft updated June 17, 2026). Formal dataset and manuscript citations will replace this note when public identifiers are assigned.
