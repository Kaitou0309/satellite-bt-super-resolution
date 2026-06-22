# Release Asset Staging

This directory is for local review of model binaries before they are uploaded to a GitHub Release. Model files are ignored by Git and must not be committed to repository history.

## Selected release models

- RRDN composite-SSIM alpha 0.8, 9 RRDB blocks, 3 RDBs per RRDB, 5 convolutional layers per RDB, 64 filters. Present locally and verified as weights-only HDF5.
- RRDN-GAN generator trained with the BatchNorm discriminator. Present locally with its checksum recorded in `SHA256SUMS.txt`.

Only the two selected generator checkpoints should be published. Discriminator weights are not required for inference.

Upload both ignored checkpoint files and `SHA256SUMS.txt` to a versioned GitHub Release. Do not force-add the model binaries to Git history.
