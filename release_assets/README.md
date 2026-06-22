# Release Asset Staging

This directory is for local review of model binaries before they are uploaded to a GitHub Release. Model files are ignored by Git and must not be committed to repository history.

## Selected release models

- RRDN composite-SSIM alpha 0.8, 9 RRDB blocks, 3 RDBs per RRDB, 5 convolutional layers per RDB, 64 filters.
- RRDN-GAN generator trained with the BatchNorm discriminator.

The HPC training environment did not support the newer `.keras` format, so both selected models were originally stored as weights-only `.weights.h5` checkpoints. Keep those original checkpoints for reproducible architecture reconstruction, fine-tuning, and HPC compatibility. The corresponding `.keras` exports package each generator architecture with its weights for easier loading and prediction. Discriminator weights are not required for inference.

The `.keras` exports contain the complete Functional generator architecture, generator weights, and Keras serialization metadata. They are inference-focused and do not contain compile configuration, optimizer state, training losses or metrics, external normalization statistics, YAML metadata, the GAN discriminator, datasets, or training history. Keep the config and `metadata/unified_global_stats.npz` with the model release workflow.

Upload all four ignored model files and `SHA256SUMS.txt` to a versioned GitHub Release. Do not force-add model binaries to Git history.

## Release checklist

1. Commit and push the tracked source, configs, documentation, and checksum manifest.
2. From the repository root, run `(cd release_assets && shasum -a 256 -c SHA256SUMS.txt)`.
3. Create release tag `v0.1.0`, or edit that release if it already exists.
4. Upload the four model files listed in `SHA256SUMS.txt` plus the checksum manifest.
5. Confirm that the release page shows all five downloadable assets.

For a new release with GitHub CLI:

```bash
gh release create v0.1.0 \
    release_assets/*.weights.h5 \
    release_assets/*.keras \
    release_assets/SHA256SUMS.txt \
    --title "v0.1.0 - Initial model release" \
    --notes "Pretrained RRDN Composite SSIM and RRDN-GAN generators in weights-only and complete Keras formats."
```

If `v0.1.0` already exists, upload or replace its model assets:

```bash
gh release upload v0.1.0 \
    release_assets/*.weights.h5 \
    release_assets/*.keras \
    release_assets/SHA256SUMS.txt \
    --clobber
```
