"""Uncertainty-driven active learning: which unlabelled images should an expert annotate next?

Pixel-wise annotation of medical images is expensive, so the labelling budget should go to the
images the model would learn most from. Every image is scored from BAU-Net's Monte-Carlo dropout
predictions, and a batch is then chosen to be both *informative* and *diverse*.

Image-level acquisition scores (higher = more useful to label):

* ``entropy``: mean predictive entropy (total uncertainty);
* ``bald``: mean mutual information between prediction and weights, i.e. the epistemic part
  that more labels can reduce (Houlsby et al., 2011; Gal et al., 2017);
* ``boundary``: mutual information averaged over a band around the predicted boundary. Most
  errors of a segmentation model sit on object contours, while interior and background pixels
  are mostly confident; averaging over the whole image dilutes exactly the signal we want;
* ``region``: the mean of the most uncertain ``q`` fraction of pixels, which reacts to a single
  badly segmented structure in an otherwise easy image.

Batch selection: picking the top-k scores tends to return near-duplicates (consecutive video
frames, the same lesion type). ``diverse_select`` first keeps a candidate pool of the most
uncertain images, then runs greedy k-centre (core-set, Sener and Savarese, 2018) on bottleneck
embeddings, seeded with the already labelled images, so each pick covers a new region of
feature space.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .geometry import mask_to_boundary

STRATEGIES = ("entropy", "bald", "boundary", "region", "random")


def acquisition_scores(
    res: Dict[str, torch.Tensor],
    band_width: int = 7,
    region_quantile: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """Per-image scores ``(B,)`` from :meth:`BAUNet.predict_with_uncertainty` outputs."""
    prob, ent, mi = res["prob"], res["entropy"], res["mutual_info"]
    b = prob.shape[0]
    band = mask_to_boundary((prob > 0.5).float(), band_width)
    band_mi = (mi * band).flatten(1).sum(1) / band.flatten(1).sum(1).clamp_min(1.0)
    # images without any predicted foreground have no boundary: fall back to the global epistemic score
    has_band = band.flatten(1).sum(1) > 0
    k = max(1, int(round(region_quantile * mi[0].numel())))
    return {
        "entropy": ent.flatten(1).mean(1),
        "bald": mi.flatten(1).mean(1),
        "boundary": torch.where(has_band, band_mi, mi.flatten(1).mean(1)),
        "region": mi.flatten(1).topk(k, dim=1).values.mean(1),
        "random": torch.rand(b, device=prob.device),
    }


@torch.no_grad()
def bottleneck_embedding(model, x: torch.Tensor) -> torch.Tensor:
    """L2-normalised global-average-pooled deepest encoder features ``(B, C)`` for diversity."""
    was = model.training
    model.eval()
    f = model.encoder(x)[-1]
    model.train(was)
    return F.normalize(f.mean(dim=(2, 3)), dim=1)


def k_center_greedy(embeddings: np.ndarray, budget: int, labelled: Optional[np.ndarray] = None) -> np.ndarray:
    """Greedy 2-approximation of k-centre: repeatedly take the point farthest from all centres."""
    e = np.asarray(embeddings, float)
    n = len(e)
    budget = min(budget, n)
    if budget <= 0:
        return np.zeros(0, int)
    if labelled is not None and len(labelled):
        lab = np.asarray(labelled, float)
        dist = np.min(((e[:, None] - lab[None]) ** 2).sum(-1), axis=1)
        chosen = []
    else:
        first = int(np.argmax(((e - e.mean(0)) ** 2).sum(1)))  # deterministic start: the most atypical point
        chosen = [first]
        dist = ((e - e[first]) ** 2).sum(1)
    while len(chosen) < budget:
        i = int(np.argmax(dist))
        chosen.append(i)
        dist = np.minimum(dist, ((e - e[i]) ** 2).sum(1))
    return np.array(chosen, int)


def diverse_select(
    scores: np.ndarray,
    embeddings: np.ndarray,
    budget: int,
    pool_factor: float = 3.0,
    labelled_embeddings: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Indices of ``budget`` images: k-centre over the ``pool_factor * budget`` most uncertain ones."""
    scores = np.asarray(scores, float)
    pool = np.argsort(-scores)[: max(budget, int(np.ceil(pool_factor * budget)))]
    picked = k_center_greedy(np.asarray(embeddings)[pool], budget, labelled_embeddings)
    return pool[picked]


def topk_select(scores: Sequence[float], budget: int) -> np.ndarray:
    return np.argsort(-np.asarray(scores, float))[:budget]
