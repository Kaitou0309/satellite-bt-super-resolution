# Beginner Guide

This folder is for users who want to run the released microwave super-resolution models without needing to understand the full machine-learning architecture first.

The intended audience includes satellite scientists, operational meteorology researchers, and collaborators from institutions such as NOAA, NASA, universities, or private weather-data teams who mainly need a practical workflow:

1. Load a released `.keras` model through the repository helper.
2. Open an HDF5 satellite brightness-temperature file.
3. Run prediction in Kelvin.
4. Save or plot the output.

## Start Here

Open:

[notebooks/01_use_pretrained_keras_models.ipynb](notebooks/01_use_pretrained_keras_models.ipynb)

The notebook demonstrates the same workflow for:

- [sample_data/amsr2_example.h5](../sample_data/amsr2_example.h5), the small public paired example included in Git.
- `AMSR2/`, if you have the larger local AMSR2 collection.
- `ATMS/`, if you have ATMS HDF5 files prepared with a compatible `bt` dataset.

Only the small AMSR2 example is intended to be committed to Git. Larger local sensor collections should remain outside Git history and can be referenced by path when running the notebook locally.

## Model Format Used Here

The beginner workflow uses the `.keras` model artifacts because they are easiest to load for prediction. The repository loader still applies the required Kelvin normalization and denormalization metadata from [configs/](../configs/) and [metadata/](../metadata/).

## When To Use Other Folders

- Use [scripts/evaluation/](../scripts/evaluation/) when you want command-line metrics or batch plotting.
- Use [docs/](../docs/) when you want architecture and data documentation.
- Use [scripts/training/](../scripts/training/) only when you are ready to train, fine-tune, or continue training models.

## Notes

- The model expects one-channel brightness-temperature input in Kelvin.
- Prediction output is also returned in Kelvin.
- If latitude and longitude grids are present in the input HDF5 file, the updated plotting scripts can show geographic axes.
- If no geolocation variables exist, plots fall back to pixel-index display.
