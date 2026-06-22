# Runnable Workflows

These scripts use the canonical `bt_super_resolution` package under `src/`.

## Contents

- `inference/make_prediction.py`: copy HDF5 inputs and append one or more model predictions.
- `evaluation/metrics.py`: evaluate one file or a recursive directory and export CSV plus a comparison matrix.
- `evaluation/plot_predictions.py`: plot predictions already stored in HDF5 without loading models.
- `evaluation/plot_selected_model_residuals.py`: generate selected-scene prediction and residual panels.
- `training/train_rrdn.py`: train RRDN with MAE, composite SSIM, or physics-inspired objectives.
- `training/train_gan.py`: continue training a pretrained RRDN generator with the PatchGAN discriminator.

Model YAML files are the source of truth for released architecture, checkpoint, and normalization metadata. RRDN-GAN inference loads only its RRDN generator; discriminator BatchNorm is a training configuration.

`make_prediction.py`, `metrics.py`, and `plot_selected_model_residuals.py` use weights-only artifacts by default. Pass `--artifact-format keras` to run the corresponding complete `.keras` generators. Both paths use the same YAML normalization metadata and return predictions in Kelvin.

Both training scripts call the preserved original `bt_super_resolution.models.build_RRDN` architecture implementation used by the research training workflow.

The scripts can run directly from a repository checkout. Installing the package with `python -m pip install -e .` is still recommended for development.
