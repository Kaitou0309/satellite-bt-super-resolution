# Satellite Brightness Temperature Super-Resolution

TensorFlow models and reproducible workflows for reconstructing four-times higher-resolution 89 GHz satellite microwave brightness-temperature fields from low-resolution observations.

The project provides two selected RRDN generators:

| Model | Generator | Training objective | Status |
| --- | --- | --- | --- |
| RRDN Composite SSIM | 9 RRDB, 3 RDB/RRDB, 5 conv/RDB, 64 filters | Composite SSIM, alpha 0.8 | Config and checkpoint validated |
| RRDN-GAN | Same RRDN generator | Composite reconstruction plus adversarial refinement | Config and checkpoint validated |

The GAN discriminator uses Batch Normalization during training. BatchNorm is not part of the released generator and is not needed for inference.

## Repository layout

```text
configs/             Release model architecture and training metadata
docs/                Dataset documentation and selected research figures
legacy/              Historical model-definition references
metadata/            Required normalization statistics
sample_data/         One compact, full-size LR/HR example
scripts/evaluation/  Metrics and visualization workflows
scripts/inference/   HDF5 prediction workflow
scripts/training/    RRDN training and RRDN-GAN fine-tuning
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

## Model checkpoints

The two weights-only HDF5 checkpoints are distributed separately from Git source. Their expected filenames and SHA-256 checksums are recorded in `configs/` and `release_assets/SHA256SUMS.txt`.

Place downloaded files under `release_assets/`:

```text
release_assets/RRDN_9RRDB_3RDB_5convlayer_g64_UNI_COMPOSITE_SSIM_ALPHA0.8_bs8_lr5e-05_loss_fncomposite_ssim_alpha0.8_BEST.h5
release_assets/RRDN_GAN_9RRDB_3RDB_5conv_g64_bs8_glr1e-05_dlr1e-05_adv1e-04_GENERATOR_BEST.weights.h5
```

<!-- TODO: Replace with links to the first versioned GitHub Release. -->

## Python inference

```python
from bt_super_resolution import load_generator

bundle = load_generator("configs/rrdn_composite_ssim_alpha_0.8.yaml")
prediction_kelvin = bundle.predict_kelvin(lr_bt, batch_size=8)
```

The loader reconstructs the exact architecture, verifies the checkpoint checksum, loads the required normalization statistics, and returns predictions in Kelvin.

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

Metrics include RMSE, global PSNR, mean per-scene SSIM, bias, and inference latency. Plotting preserves the complete spatial field; percentile and residual limits affect only color display.

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

Formal manuscript and dataset citations will be added when public identifiers are assigned. The provisional software metadata is available in `CITATION.cff`.
