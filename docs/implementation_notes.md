# Implementation notes

## Boundary targets
`mask_to_boundary` computes `maxpool(M) - (-maxpool(-M))`, i.e. dilation minus erosion with
a square structuring element of size `2 * (width // 2) + 1`. It runs on the same device as
the batch, so no pre-computed boundary maps are stored on disk.

## Class balancing
Boundary pixels typically cover 2-5 % of an image. The boundary BCE uses
`pos_weight = (1 - p) / p` (clipped at 50), where `p` is the fraction of boundary pixels in
the batch.

## Mixed precision
The forward pass runs under `torch.autocast`; outputs are cast back to `float32` before the
loss so the Dice denominator and `log` terms are computed in full precision.

## Metrics
* Dice / IoU are computed **per image** and averaged (the convention used by most polyp
  segmentation benchmarks), with empty-prediction/empty-target pairs counted as 1.
* HD95 is the symmetric 95th percentile of surface distances in pixels; images where
  exactly one mask is empty are excluded (`nan`) from the mean.

## Reproducibility
* `seed` controls Python, NumPy and PyTorch RNGs; set `deterministic: true` to force
  deterministic cuDNN kernels (slower).
* Split files are generated with a fixed seed (2026) by `scripts/make_splits.py`.
* The resolved config is stored in every checkpoint and in `runs/<name>/config.json`.
