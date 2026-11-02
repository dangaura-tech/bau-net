<div align="center">

# BAU-Net

### Boundary-Aware U-Net with Monte-Carlo Uncertainty for Medical Image Segmentation

[Dang Truong Minh Le](https://github.com/dangaura-tech)

Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)

[![CI](https://github.com/dangaura-tech/bau-net/actions/workflows/ci.yml/badge.svg)](https://github.com/dangaura-tech/bau-net/actions/workflows/ci.yml)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

---

## Abstract

Accurate delineation of lesion boundaries is critical in clinical workflows such as
polyp resection and melanoma excision, yet encoder-decoder networks trained with
region-based losses tend to produce over-smoothed contours, and they give no indication
of *where* the prediction should not be trusted. **BAU-Net** addresses both issues within
a single, lightweight architecture:

1. an **auxiliary boundary head** supervised by a morphological boundary band extracted
   on-the-fly from the ground-truth mask, whose prediction is **fused back** into the
   mask branch;
2. **attention-gated skip connections** that suppress irrelevant encoder activations
   (specular highlights, hair, mucosal folds);
3. **Monte-Carlo dropout** in the decoder, which yields pixel-wise *predictive entropy*
   and *mutual information* maps at test time without any extra training.

The codebase provides a reproducible pipeline (configs, seeds, split files, ablation
scripts) for **Kvasir-SEG**, **CVC-ClinicDB** and **ISIC 2018**, and works with any dataset
stored as `images/` + `masks/`.

## Method

```
                ┌────────────── attention-gated skips ──────────────┐
                │                                                   ▼
 image ─► Res-Encoder (5 levels) ─► bottleneck ─► Res-Decoder (MC-dropout) ─► F
                                                                             │
                                          ┌──────────────────────────────────┤
                                          ▼                                  │
                                   Boundary head ──► B̂ ──► σ(B̂) ⊕ F ─► fuse ─► Mask head ─► M̂
                                          │                                              │
                       L_boundary(B̂, ∂M) ◄┘              L_dice(M̂, M) + L_bce(M̂, M) ◄───┘
```

**Training objective**

$$\mathcal{L} = \mathcal{L}_{\text{Dice}}(\hat M, M) + \mathcal{L}_{\text{BCE}}(\hat M, M) + \lambda_b\,\mathcal{L}_{\text{wBCE}}(\hat B, \partial M) + \lambda_{\text{aux}} \sum_k \mathcal{L}_{\text{seg}}(\hat M_k, M)$$

where $\partial M = \text{dilate}(M) - \text{erode}(M)$ is computed with max-pooling on the
GPU, the boundary BCE is class-balanced, and $\hat M_k$ are deep-supervision outputs.

**Uncertainty.** With $T$ stochastic forward passes $p_t$, BAU-Net reports the predictive
entropy $\mathcal{H}[\bar p]$ and the mutual information
$\mathcal{H}[\bar p] - \frac{1}{T}\sum_t \mathcal{H}[p_t]$ (epistemic uncertainty).

## Installation

```bash
git clone https://github.com/dangaura-tech/bau-net.git
cd bau-net
conda create -n baunet python=3.10 -y && conda activate baunet
pip install -r requirements.txt     # or: pip install -e ".[dev]"
```

## Data preparation

No data is shipped with this repository. Each script downloads or reorganises the
official release and writes reproducible split files to `splits/` (seed 2026).

| Dataset | Modality | Images | Command |
|---|---|---|---|
| [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) | colonoscopy | 1,000 | `bash scripts/prepare_kvasir.sh data/kvasir` |
| [CVC-ClinicDB](https://polyp.grand-challenge.org/CVCClinicDB/) | colonoscopy | 612 | place in `data/cvc_clinicdb/{images,masks}` then `python scripts/make_splits.py --images data/cvc_clinicdb/images --out splits/cvc_clinicdb` |
| [ISIC 2018 Task 1](https://challenge.isic-archive.com/data/#2018) | dermoscopy | 2,594 | `python scripts/prepare_isic2018.py --src <download dir>` |

Expected layout:

```
data/kvasir/
├── images/  cju0qkwl35piu0993l0dewei2.jpg ...
└── masks/   cju0qkwl35piu0993l0dewei2.jpg ...
splits/kvasir/{train,val,test}.txt
```

## Usage

**Training**

```bash
python tools/train.py --config configs/kvasir_seg.yaml
# any config key can be overridden from the command line
python tools/train.py --config configs/isic2018.yaml train.lr=5e-4 train.epochs=150 seed=1
```

Checkpoints (`best.pth`, `last.pth`), `train.log` and `history.jsonl` are written to
`runs/<config name>/`. Resume with `--resume runs/kvasir_seg/last.pth`.

**Evaluation** (Dice, IoU, precision, recall, HD95)

```bash
python tools/evaluate.py --checkpoint runs/kvasir_seg/best.pth \
    data.val_split=splits/kvasir/test.txt --out results/kvasir_seg_s42.json
```

**Inference with uncertainty maps**

```bash
python tools/predict.py --checkpoint runs/kvasir_seg/best.pth \
    --input path/to/images --output predictions --mc-samples 20
# writes <name>_mask.png, <name>_overlay.png, <name>_uncertainty.png
```

**Ablations** (full model vs. no boundary head / no attention / plain U-Net, 3 seeds)

```bash
bash scripts/run_ablations.sh          # aggregates mean ± std into a Markdown table
```

**Complexity**

```bash
python tools/benchmark.py --config configs/kvasir_seg.yaml
```

**Smoke test on synthetic data (CPU, ~1 minute)**

```bash
python tools/train.py --config configs/synthetic.yaml --device cpu
```

## Repository structure

```
bau-net/
├── baunet/
│   ├── models/        # BAU-Net, residual blocks, attention gates
│   ├── data/          # folder & synthetic datasets, joint augmentations
│   ├── losses.py      # Dice, focal, boundary-aware objective
│   ├── metrics.py     # Dice, IoU, precision, recall, HD95
│   ├── engine.py      # Trainer (AMP, warmup + cosine, early stopping), evaluate()
│   └── utils/         # YAML configs with inheritance, seeding, logging
├── configs/           # dataset configs + ablations/
├── tools/             # train / evaluate / predict / benchmark
├── scripts/           # data preparation, splits, ablation runner
├── tests/             # unit + smoke tests (pytest)
└── docs/              # implementation notes
```

## Configuration

Configs are YAML files that inherit from `configs/_base_.yaml` via the `_base_` key.
The most relevant options:

| Key | Default | Description |
|---|---|---|
| `model.widths` | `[32, 64, 128, 256, 512]` | channels per resolution level |
| `model.boundary_head` | `true` | auxiliary boundary branch + fusion |
| `model.attention` | `true` | attention gates on skips |
| `model.dropout` | `0.1` | decoder spatial dropout (MC-dropout) |
| `loss.boundary_weight` | `0.5` | $\lambda_b$ |
| `loss.boundary_width` | `3` | width of the boundary band in pixels |
| `train.amp` | `true` | mixed precision on CUDA |

## Citation

If you find this code useful, please cite the repository:

```bibtex
@misc{le2026baunet,
  title        = {{BAU-Net}: Boundary-Aware {U-Net} with Monte-Carlo Uncertainty for Medical Image Segmentation},
  author       = {Le, Dang Truong Minh},
  year         = {2026},
  howpublished = {\url{https://github.com/dangaura-tech/bau-net}}
}
```

## Acknowledgements

This work builds on ideas from U-Net (Ronneberger et al., 2015), Attention U-Net
(Oktay et al., 2018) and MC-dropout (Gal & Ghahramani, 2016). We thank the creators of
Kvasir-SEG, CVC-ClinicDB and the ISIC Archive for making their data publicly available.

## License

Code is released under the [MIT License](LICENSE). The datasets are subject to their own
licenses and are intended for research use only. **This software is not a medical device
and must not be used for clinical decision making.**
