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

## v2 components

### Encoders
`TorchvisionEncoder` turns ResNet/ConvNeXt into a 6-level pyramid (strides 1, 2, 4, 8, 16,
32). A light full-resolution `ConvBlock` stem is added so that the last decoder stage has
pixel-accurate features for the boundary/SDF heads and the point head. ConvNeXt, whose
first stage is stride 4, also gets a stride-2 conv stem. Channel widths are discovered with
a dummy forward pass, so new backbones only need a stage list.

### Global Context Transformer
Tokens from the stride-32 map are projected to `transformer_dim` and given fixed 2-D
sine-cosine position encodings, so any input resolution works without interpolating learned
embeddings. The output projection is zero-initialised, which makes the block an identity at
initialisation and protects the pretrained encoder features early in training.

### Signed distance targets
`geometry.signed_distance` uses SciPy's exact Euclidean distance transform on CPU. Each
side of the contour is normalised separately so small and large objects contribute
comparably. The consistency term compares `sigmoid(mask)` with `sigmoid(-kappa * sdf)`;
`kappa = 50` turns the normalised SDF into a sharp soft mask.

### Uncertainty-guided Point Refinement
* Point features: bilinear samples of the full-resolution decoder features and of the coarse
  mask logit, boundary logit and SDF (`point_sample`, `align_corners=False`, so pixel centres
  are sampled exactly).
* The point head predicts a *residual* to the coarse logit and is zero-initialised: the
  refined mask equals the coarse mask at initialisation.
* Training: oversample 3x, keep 75 % by score, 25 % uniform. Inference: refine the top
  `refine_eval_points` pixels (scattered back in place). `mask_coarse` is also returned.

### Uncertainty metrics
* **UEO**: maximum Dice between the thresholded (min-max normalised) uncertainty map and the
  error map over 49 thresholds.
* **Uncertainty AUROC**: rank-based AUROC (with tie handling) of uncertainty for separating
  erroneous from correct pixels.
* **ECE**: 15 equal-width bins over pixel probabilities.
