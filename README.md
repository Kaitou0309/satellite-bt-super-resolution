# Satellite Brightness Temperature Super-Resolution

TensorFlow models and reproducible workflows for reconstructing four-times higher-resolution 89 GHz satellite microwave brightness-temperature fields from low-resolution observations.

The project provides two selected microwave super-resolution generators:

| Model | Generator | Training objective | Status |
| --- | --- | --- | --- |
| MW-SR | RRDN generator: 9 RRDB, 3 RDB/RRDB, 5 conv/RDB, 64 filters | Composite SSIM, alpha 0.8 | Config and checkpoint validated |
| MW-SR-GAN | Same RRDN generator | Composite reconstruction plus adversarial refinement | Config and checkpoint validated |

The GAN discriminator uses Batch Normalization during training. BatchNorm is not part of the released generator and is not needed for inference.

## Choose your path

This repository is organized for three levels of users.

### 1. Beginner-friendly model use

Start here if you mainly want to run the released model on satellite brightness-temperature HDF5 files and inspect the output. This path is intended for scientists and collaborators who may not need the full RRDN/GAN theory to use the model productively.

- Folder: `beginner_guide/`
- Notebook: `beginner_guide/notebooks/01_use_pretrained_keras_models.ipynb`
- Raw ATMS example: `github_case/`
- Recommended artifact: `.keras`
- Typical users: applied satellite scientists, NOAA/NASA/research collaborators, meteorology teams, and first-time repository users.
- Covered workflow: load model, read AMSR2/ATMS/Tomorrow.io-style HDF5 inputs, run prediction, inspect LR and predicted HR output.
- Use `github_case/` when starting from raw ATMS SDR/GEO swath files rather than prepared model-input HDF5 files.

### 2. ML-aware model understanding

Use this path if you are comfortable with machine-learning concepts and want to understand the model architecture, losses, validation metrics, and qualitative behavior.

- Folders: `docs/`, `configs/`, `scripts/evaluation/`, `src/bt_super_resolution/models/`
- Covered topics: RDN/RRDN/RRDB generator design, PatchGAN-style discriminator, Composite SSIM, GAN refinement, residual plots, PSD behavior, and quantitative metrics.
- Best starting points: the architecture figures below, `MODEL_CARD.md`, `docs/DATA.md`, and the evaluation scripts.

### 3. Training and fine-tuning

Use this path if you want to reproduce training, fine-tune the released generators, or continue training on compatible data. This path assumes you can manage Python environments, HDF5 datasets, GPU/HPC execution, and experiment tracking.

- Folders: `scripts/training/`, `legacy/`, `configs/`, `metadata/`
- Recommended artifact: `.weights.h5` for architecture reconstruction and continued training; `.keras` for convenient inference.
- Covered workflow: model creation, normalization, loss selection, RRDN training, GAN refinement, and checkpoint export.
- Important note: full research datasets and experiment logs are not included in Git. Users must provide compatible training/evaluation HDF5 files and validate their own data rights.

## Model architecture

### MW-SR generator

![RRDN generator architecture showing the shallow feature extractor, RRDB trunk, global residual connection, and four-times upsampling head](docs/figures/RRDN_Model_Structure.png)

The generator maps a single-channel low-resolution BT field of shape `H x W x 1` to a field of shape `4H x 4W x 1`. A shallow convolution first maps BT values into a feature representation. The deep trunk then applies residual-in-residual dense blocks (RRDBs), each composed of densely connected residual dense blocks (RDBs).

Residual learning occurs at three levels: inside each RDB, across each RRDB, and across the complete feature trunk. The RRDB residual is scaled by 0.2 before it is added to the identity path, which stabilizes deeper training. The final head uses bilinear 4x upsampling followed by convolutional refinement. Batch Normalization is omitted from the generator to avoid unnecessarily transforming the radiometric feature distribution.

### MW-SR-GAN refinement

![MW-SR-GAN training architecture showing the pretrained generator, reconstruction objective, patch discriminator, and adversarial objectives](docs/figures/GAN_Model_Architecture.png)

GAN training begins from the pretrained MW-SR generator rather than an untrained network. The generator continues to optimize a reconstruction objective while receiving adversarial feedback from a deep PatchGAN-style discriminator. Instead of assigning one real/fake score to an entire scene, the discriminator produces a spatial probability map and evaluates local BT patches around features such as cloud edges, storm boundaries, and sharp thermal transitions.

The released MW-SR-GAN generator was trained with the BatchNorm discriminator variant. The discriminator is required only during adversarial training; prediction uses the generator by itself.

## Repository layout

```text
beginner_guide/     Notebook-first guide for users who just want model inference
configs/             Release model architecture and training metadata
docs/                Dataset documentation and selected research figures
legacy/              Historical model-definition references
metadata/            Required normalization statistics
sample_data/         One compact, full-size LR/HR example
scripts/evaluation/  Metrics and visualization workflows
scripts/inference/   HDF5 prediction workflow
scripts/training/    RRDN training and MW-SR-GAN fine-tuning
github_case/          Raw ATMS SDR/GEO example for map generation
src/                  Canonical importable Python package
tests/                Fast tests and optional release-checkpoint tests
tools/                HDF5, checkpoint, and sample-generation utilities
```

Raw datasets, generated outputs, scheduler logs, and checkpoint binaries are intentionally excluded from Git history.

The original Chen architecture implementations are preserved as `src/bt_super_resolution/models/rdn_chen.py` and `rrdn_chen.py`. Training calls the original `build_RRDN` function directly. The public `build_rrdn` API is only a config-name adapter around that same implementation, ensuring training and inference reconstruct an identical layer topology.

## Installation

Using Conda:

```bash
conda env create -f environment.yml
conda activate bt-super-resolution
```

Using an existing Python 3.10 environment:

```bash
python -m pip install -e .
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
python -m pytest -m "not release"
```

## Model files

Each selected generator is distributed in two formats through GitHub Releases. Training was performed on an HPC environment that did not support saving the newer `.keras` format, so the original selected checkpoints were saved as weights-only `.weights.h5` files. Those validated checkpoints are preserved for architecture reconstruction, fine-tuning, and HPC compatibility. Equivalent `.keras` files are exported afterward for easier loading and prediction.

The expected filenames and SHA-256 checksums are recorded in `configs/` and `release_assets/SHA256SUMS.txt`.

Place downloaded files under `release_assets/`:

```text
release_assets/bt-sr-rrdn-9rrdb-composite-ssim-a08-v0.1.0.weights.h5
release_assets/bt-sr-rrdn-9rrdb-composite-ssim-a08-v0.1.0.keras
release_assets/bt-sr-rrdn-gan-9rrdb-bn-generator-v0.1.0.weights.h5
release_assets/bt-sr-rrdn-gan-9rrdb-bn-generator-v0.1.0.keras
```

<!-- TODO: Replace with links to the first versioned GitHub Release. -->

### What the `.keras` files contain

Both converted artifacts were inspected as Keras v3 archives and reloaded without custom objects. Each contains:

- The complete Functional generator architecture: 484 serialized standard Keras layers.
- All generator weights in the archive's internal `model.weights.h5`.
- Input/output graph configuration, layer names, shapes, and activations.
- Keras serialization metadata, including the Keras version and save date.

The exports have an empty compile configuration because they are inference-focused generator releases. Compared with a compiled training-oriented `.keras` checkpoint, they do not contain an optimizer or optimizer state, training loss, compiled metrics, or custom training objectives. They also do not contain the external Kelvin normalization statistics, YAML experiment metadata, GAN discriminator, dataset, training history, model card, or evaluation outputs. Those companion artifacts remain in the repository. A `.keras` file does not automatically package external preprocessing unless that preprocessing is built into the model graph.

## Python inference

```python
from bt_super_resolution import load_generator

bundle = load_generator("configs/rrdn_composite_ssim_alpha_0.8.yaml")
prediction_kelvin = bundle.predict_kelvin(lr_bt, batch_size=8)
```

The loader reconstructs the exact architecture, verifies the checkpoint checksum, loads the required normalization statistics, and returns predictions in Kelvin.

For Kelvin-aware inference from a complete Keras artifact, use the repository loader:

```python
from bt_super_resolution import load_keras_generator

bundle = load_keras_generator("configs/rrdn_composite_ssim_alpha_0.8.yaml")
prediction_kelvin = bundle.predict_kelvin(lr_bt, batch_size=8)
```

For direct normalized-input Keras loading:

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "release_assets/bt-sr-rrdn-9rrdb-composite-ssim-a08-v0.1.0.keras",
    compile=False,
)
```

The full model expects normalized LR input and returns normalized HR output. Use the repository loader when Kelvin-aware normalization and denormalization are required.

## HDF5 prediction

Run both selected models on the public example:

```bash
python scripts/inference/make_prediction.py \
    --input sample_data/amsr2_example.h5 \
    --output outputs/example_with_predictions.h5 \
    --batch-size 1 \
    --overwrite \
    --strict
```

The same command accepts a directory and recursively processes every `.h5` or `.hdf5` file.

`make_prediction.py` copies each source HDF5 file before appending predictions, so latitude/longitude variables are preserved when they exist in the source file. Prediction datasets record whether matching geolocation fields were available for plotting.

Use the complete `.keras` models instead of reconstructed weights:

```bash
python scripts/inference/make_prediction.py \
    --input sample_data/amsr2_example.h5 \
    --output outputs/example_keras_predictions.h5 \
    --artifact-format keras \
    --batch-size 1 \
    --overwrite \
    --strict
```

## Raw ATMS GitHub case

The `github_case/` folder is a beginner-facing raw-data example. It is separate from `scripts/inference/make_prediction.py` because it starts one step earlier in the workflow: raw ATMS SDR and geolocation files instead of model-ready HDF5 input.

Use this example when you have paired raw files such as:

```text
github_case/data/SATMS_j01_d20260415*.h5
github_case/data/GATMO_j01_d20260415*.h5
```

Run the example from the repository root:

```bash
python github_case/atms_example.py
```

The script reads the SDR brightness-temperature array, applies the ATMS scale factors, extracts channel 16, loads the matching latitude/longitude swath, and writes an original-resolution map such as:

```text
github_case/orig.png
```

The current example writes the output figure into the `github_case/` folder so it stays next to the raw-data demonstration script.

This raw ATMS example is useful for visual inspection and for understanding how ATMS swath data are organized. It is not the same as the model prediction pipeline. To run the released MW-SR or MW-SR-GAN model, first prepare a model-ready HDF5 file containing a compatible brightness-temperature dataset, then use `scripts/inference/make_prediction.py`.

## Evaluation and plots

```bash
python scripts/evaluation/metrics.py \
    --data sample_data/amsr2_example.h5 \
    --output outputs/example_metrics.csv \
    --plot-output outputs/example_metrics_matrix.png \
    --strict

python scripts/evaluation/plot_predictions.py \
    --input outputs/example_with_predictions.h5 \
    --output-dir outputs/prediction_plots \
    --overwrite \
    --strict
```

Generate prediction and residual panels directly from the `.keras` models:

```bash
python scripts/evaluation/plot_selected_model_residuals.py \
    --input sample_data/amsr2_example.h5 \
    --artifact-format keras \
    --output-dir outputs/keras_residual_plots \
    --strict
```

Evaluate the complete `.keras` release artifacts with the same normalization and metrics:

```bash
python scripts/evaluation/metrics.py \
    --data sample_data/amsr2_example.h5 \
    --artifact-format keras \
    --output outputs/example_keras_metrics.csv \
    --strict
```

Metrics include RMSE, global PSNR, mean per-scene SSIM, bias, and inference latency. Plotting preserves the complete spatial field; percentile and residual limits affect only color display.

The plotting scripts automatically use longitude and latitude axes when compatible coordinate datasets are present in the HDF5 file. If geolocation variables are missing, plotting falls back to the original pixel-index display.

## Results and visual evaluation

The following figures summarize the current research experiments. They are included as supporting analysis rather than as a claim that fine-scale atmospheric structure can be uniquely recovered from one microwave channel.

### Quantitative model comparison

#### AMSR2 hurricane-specific data performance

| Model | RMSE | Bias | SSIM | PSNR |
| --- | ---: | ---: | ---: | ---: |
| Bilinear Interpolation | 4.1501 | -0.0566 | 0.8599 | 33.4296 |
| RRDN 9rrdb (MAE, 5 conv) | 3.6802 | 0.1512 | 0.8818 | 34.4566 |
| RRDN 9rrdb (Composite SSIM alpha=0.2) | 3.8439 | 0.1288 | 0.8767 | 34.0881 |
| RRDN 9rrdb (Composite SSIM alpha=0.5) | 3.8327 | -0.0221 | 0.8773 | 34.0962 |
| MW-SR | 3.3685 | 0.0222 | 0.8991 | 35.2978 |
| RRDN 3rrdb (Total Physics) | 3.8226 | 0.0303 | 0.8750 | 34.1189 |
| RRDN 5rrdb (Total Physics) | 3.7809 | 0.0888 | 0.8773 | 34.2399 |
| RRDN 7rrdb (Total Physics) | 3.8796 | 0.1015 | 0.8739 | 33.9972 |
| RRDN 9rrdb (Total Physics) | 3.6519 | 0.1077 | 0.8815 | 34.5343 |
| RRDN 11rrdb (Total Physics) | 3.8114 | 0.0100 | 0.8752 | 34.1412 |
| RRDN 20rrdb (Total Physics) | 3.8642 | 0.0575 | 0.8736 | 34.0288 |
| MW-SR-GAN (No BatchNorm ablation) | 3.1521 | **0.0087** | 0.9050 | 35.8073 |
| MW-SR-GAN | **3.1057** | 0.0733 | **0.9072** | **35.9081** |

#### Standard HDF5 evaluation set performance

| Model | RMSE | Bias | SSIM | PSNR |
| --- | ---: | ---: | ---: | ---: |
| Bilinear Interpolation | 4.2570 | 0.0238 | 0.8396 | 33.3977 |
| RRDN 9rrdb (MAE, 5 conv) | 3.7622 | 0.0533 | 0.8638 | 34.4185 |
| RRDN 9rrdb (Composite SSIM alpha=0.2) | 3.9255 | 0.1083 | 0.8583 | 34.0570 |
| RRDN 9rrdb (Composite SSIM alpha=0.5) | 3.9351 | -0.0773 | 0.8586 | 34.0281 |
| MW-SR | 3.4219 | -0.0230 | 0.8835 | 35.2625 |
| RRDN 3rrdb (Total Physics) | 3.9155 | -0.0444 | 0.8564 | 34.0669 |
| RRDN 5rrdb (Total Physics) | 3.9136 | 0.0122 | 0.8578 | 34.1296 |
| RRDN 7rrdb (Total Physics) | 4.0036 | 0.0211 | 0.8543 | 33.9050 |
| RRDN 9rrdb (Total Physics) | 3.7732 | 0.0542 | 0.8626 | 34.4249 |
| RRDN 11rrdb (Total Physics) | 3.9279 | -0.0343 | 0.8555 | 34.0494 |
| RRDN 20rrdb (Total Physics) | 3.9616 | -0.0457 | 0.8549 | 33.9725 |
| MW-SR-GAN (No BatchNorm ablation) | 3.2596 | **0.0506** | 0.8902 | 35.6568 |
| MW-SR-GAN | **3.2213** | 0.1183 | **0.8914** | **35.7417** |

RMSE is computed globally by pooling all pixels from all scenes into one error array and calculating one RMSE value. Bias is also computed globally by pooling every pixel from every scene and averaging the signed error. SSIM is computed per scene and then averaged across scenes. PSNR is computed globally from the same global MSE value.

Across both evaluation sets, MW-SR-GAN variants produce the strongest RMSE, SSIM, and PSNR results. MW-SR is the strongest non-GAN model, while increasing RRDB depth or adding the current total-physics loss does not consistently improve reconstruction.

### Reconstruction and residuals

![Low-resolution input, bilinear interpolation, MW-SR-GAN prediction, and high-resolution truth for a hurricane scene](docs/figures/LR_Prediction_Truth.png)

This paired hurricane example compares the original LR input, bilinear interpolation, the MW-SR-GAN reconstruction, and the aligned HR target. The enlarged region shows where the learned model restores sharper eyewall and rainband organization than interpolation alone.

![MW-SR and MW-SR-GAN reconstructions with residual maps and pixel-error distributions](docs/figures/Image_Reconstruction_Residual_of_Models.png)

Residual maps show `truth - prediction` in Kelvin and make spatially organized errors visible. In this scene, adversarial refinement reduces the displayed error spread relative to the Composite SSIM model while preserving sharper storm structure. A strong visual result should still be read together with RMSE, bias, PSNR, and SSIM because sharper output is not automatically more physically correct.

### Spatial-frequency behavior

![Radially averaged power spectral density comparison of headline models and ground truth](docs/figures/psd_comparison_headline.png)

Radially averaged power spectral density (PSD) measures how spatial variability is distributed across frequency scales. Lower frequency indices correspond to broad thermal patterns; higher indices correspond to finer changes such as cloud edges and storm boundaries. The GAN curves remain closer to the ground-truth spectrum through more of the middle- and high-frequency range than the reconstruction-only models, supporting the interpretation that patch-based adversarial refinement preserves additional local structure.

All models still under-represent the highest-frequency power. This remaining spectral gap is an important limitation: the LR input does not uniquely encode every missing HR feature, and neither MW-SR nor the current physics-inspired loss fully resolves that information bottleneck.

### Operational storm examples

<p align="center">
  <img src="docs/figures/Super_Typhoon_Sinlaku.png" width="47%" alt="Original ATMS brightness-temperature field and MW-SR-GAN prediction for Super Typhoon Sinlaku">
  <img src="docs/figures/Tropical_Cyclone_Develop_Case.png" width="37%" alt="VIIRS context, original ATMS field, and MW-SR-GAN prediction for a developing tropical cyclone">
</p>

These ATMS examples test inference on meteorologically important scenes outside the paired AMSR2 evaluation set. The Super Typhoon Sinlaku comparison shows the original ATMS field and its MW-SR-GAN output; the developing-cyclone example adds VIIRS imagery as contextual reference. Because these cases do not have aligned HR BT targets, they support qualitative inspection only and are not used to claim quantitative reconstruction accuracy.

## Training from scratch

Training data must contain batched `L/bt` and `H/bt` datasets. Server-specific paths are deliberately not embedded in the scripts.

```bash
python scripts/training/train_rrdn.py \
    --train_path /path/to/training_data.h5 \
    --eval_path /path/to/evaluation_data.h5 \
    --stats_path metadata/unified_global_stats.npz \
    --loss_fn composite_ssim \
    --ssim_alpha 0.8
```

If `--eval_path` is omitted, 20 percent of the training data is reserved for validation.

## Continue training with GAN refinement

```bash
python scripts/training/train_gan.py \
    --train_path /path/to/training_data.h5 \
    --stats_path metadata/unified_global_stats.npz \
    --pretrained_generator_path /path/to/pretrained.weights.h5 \
    --use_batchnorm_d
```

Users may train from scratch, fine-tune the released generators, or continue training on their own compatible data. New datasets require carefully validated normalization statistics and sensor-specific evaluation.

## Data

The included `sample_data/amsr2_example.h5` is a 388 KB software-validation example with the production shapes:

```text
L/bt    (96, 100), float32, Kelvin
H/bt    (384, 400), float32, Kelvin
```

It is not a training dataset. The complete local `AMSR2/` collection remains outside Git and should eventually be hosted in a DOI-backed scientific repository. See [docs/DATA.md](docs/DATA.md) for construction, provenance, limitations, and attribution.

## Limitations

- Fine-scale information is not uniquely determined by one low-resolution microwave channel.
- Output behavior may vary with sensor geometry, season, surface type, storm structure, and training coverage.
- Sharper structures do not guarantee physical correctness.
- The HR training targets are physically based simulations, not direct observations from an existing high-resolution sounder.
- Operational ATMS and Tomorrow.io examples without aligned HR targets support qualitative assessment only.

## License

Source code and official released generator weights are licensed under Apache-2.0. This permits model use, modification, fine-tuning, continued training, and redistribution subject to the license and notices. See `LICENSE`, `NOTICE`, and `MODEL_LICENSE.md`.

Third-party and full research datasets retain their respective terms and are not relicensed by this repository.

## Citation

If this software supports your work, please cite the repository:

```bibtex
@software{wang2026satellite_bt_super_resolution,
  author = {Wang, Likun and Chen, Kongkun},
  title = {Satellite Brightness Temperature Super-Resolution},
  year = {2026},
  url = {https://github.com/Kaitou0309/satellite-bt-super-resolution}
}
```

The same software metadata is available in `CITATION.cff`. Formal manuscript and dataset citations will be added when public identifiers are assigned.
