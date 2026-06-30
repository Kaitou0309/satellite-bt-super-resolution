# Model Card

## Intended use

The selected MW-SR generators reconstruct 4x high-resolution, single-channel satellite microwave brightness temperature fields from normalized low-resolution inputs.

## Released models

1. MW-SR: an RRDN generator trained with composite SSIM alpha 0.8.
2. MW-SR-GAN: the MW-SR generator refined with adversarial training using a BatchNorm discriminator.

Each generator is released as both a weights-only `.weights.h5` checkpoint and a complete `.keras` model. The HPC training environment did not support the newer Keras model format, so `.weights.h5` is the original training artifact. The equivalent `.keras` export is provided for convenient architecture-plus-weights loading and prediction.

The `.keras` files contain the Functional generator graph and weights, but not normalization statistics, optimizer state, compiled losses or metrics, training history, datasets, or the GAN discriminator. Kelvin-space inference requires the companion normalization metadata and repository loader.

## Inputs and outputs

- Input: normalized one-channel low-resolution brightness temperature field.
- Output: normalized one-channel 4x super-resolved field.
- Scientific unit after denormalization: Kelvin.

## Limitations

- Fine-scale information is not uniquely determined by a single low-resolution channel.
- Outputs may depend on sensor geometry, season, storm structure, and training-data coverage.
- Sharper visual structure does not guarantee physical correctness.
- Operational sensor examples without aligned high-resolution targets are qualitative.

## Required companion artifacts

- Exact RRDN architecture configuration.
- `metadata/unified_global_stats.npz`.
- TensorFlow/Keras environment information.
- Checkpoint checksum.
- A tested inference script.

## Validation

The research workflow evaluates RMSE, global PSNR, SSIM, bias, latency, residual maps, and radially averaged power spectral density. Both selected checkpoints have been loaded with the public config-driven loader and evaluated across 42 local AMSR2 hurricane scenes. All 84 saved prediction fields matched the complete HR dimensions.

The public example is intended for software validation only. Full scientific benchmark tables and permanent dataset identifiers will accompany the manuscript and dataset release.

## Training and modification

The official weights may be used directly, fine-tuned, or continued on compatible user data under `MODEL_LICENSE.md`. Users must recompute or deliberately validate normalization statistics when changing sensors or datasets.
