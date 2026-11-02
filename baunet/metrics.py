"""Evaluation metrics.

Region:      Dice, IoU, precision, recall
Contour:     HD95, Boundary IoU (Cheng et al., CVPR 2021)
Uncertainty: pixel ECE, uncertainty-error overlap (UEO; Jungo & Reyes, MICCAI 2019) and the
             AUROC of the uncertainty map for detecting segmentation errors.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
from scipy import ndimage

from .geometry import erode


def _confusion(pred: torch.Tensor, target: torch.Tensor):
    pred = pred.flatten(1).float()
    target = target.flatten(1).float()
    tp = (pred * target).sum(1)
    fp = (pred * (1 - target)).sum(1)
    fn = ((1 - pred) * target).sum(1)
    return tp, fp, fn


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    tp, fp, fn = _confusion(pred, target)
    empty = (tp + fp + fn) == 0
    return torch.where(empty, torch.ones_like(tp), (2 * tp) / (2 * tp + fp + fn + eps))


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    tp, fp, fn = _confusion(pred, target)
    empty = (tp + fp + fn) == 0
    return torch.where(empty, torch.ones_like(tp), tp / (tp + fp + fn + eps))


def boundary_iou(pred: torch.Tensor, target: torch.Tensor, dilation_ratio: float = 0.02) -> torch.Tensor:
    """Boundary IoU: IoU restricted to the inner band of width ``d = ratio * diagonal``."""
    h, w = pred.shape[-2:]
    d = max(1, int(round(dilation_ratio * (h * h + w * w) ** 0.5)))
    pb = pred - erode(pred, d)
    tb = target - erode(target, d)
    return iou_score(pb, tb)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask ^ ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)


def hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """Symmetric 95th-percentile Hausdorff distance in pixels (``nan`` if exactly one mask is empty)."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float("nan")
    sp, st = _surface(pred), _surface(target)
    dt_t = ndimage.distance_transform_edt(~st)
    dt_p = ndimage.distance_transform_edt(~sp)
    d = np.concatenate([dt_t[sp], dt_p[st]])
    return float(np.percentile(d, 95))


def expected_calibration_error(prob: np.ndarray, target: np.ndarray, n_bins: int = 15) -> float:
    prob = prob.ravel()
    target = target.ravel()
    idx = np.clip((prob * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.any():
            ece += sel.mean() * abs(prob[sel].mean() - target[sel].mean())
    return float(ece)


def uncertainty_error_overlap(unc: np.ndarray, error: np.ndarray, n_thresholds: int = 50) -> float:
    """Max Dice between the thresholded uncertainty map and the error map."""
    unc = unc.ravel()
    error = error.ravel().astype(bool)
    if not error.any():
        return float("nan")
    u = (unc - unc.min()) / (unc.max() - unc.min() + 1e-12)
    best = 0.0
    for t in np.linspace(0, 1, n_thresholds, endpoint=False)[1:]:
        m = u > t
        inter = (m & error).sum()
        best = max(best, 2 * inter / (m.sum() + error.sum()))
    return float(best)


def auroc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based (Mann-Whitney) AUROC, ``nan`` when only one class is present."""
    score = score.ravel()
    label = label.ravel().astype(bool)
    n_pos, n_neg = label.sum(), (~label).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = score.argsort(kind="mergesort")
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks of ties
    s_sorted = score[order]
    _, first, counts = np.unique(s_sorted, return_index=True, return_counts=True)
    for f, c in zip(first[counts > 1], counts[counts > 1], strict=True):
        ranks[order[f : f + c]] = ranks[order[f : f + c]].mean()
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


class SegmentationMeter:
    """Accumulates per-image metrics over an evaluation run."""

    def __init__(self, threshold: float = 0.5, compute_hd95: bool = True) -> None:
        self.threshold = threshold
        self.compute_hd95 = compute_hd95
        self.values: Dict[str, list] = defaultdict(list)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor, uncertainty: Optional[torch.Tensor] = None) -> None:
        prob = torch.sigmoid(logits)
        self.update_prob(prob, target, uncertainty)

    @torch.no_grad()
    def update_prob(self, prob: torch.Tensor, target: torch.Tensor, uncertainty: Optional[torch.Tensor] = None) -> None:
        pred = (prob > self.threshold).float()
        tp, fp, fn = _confusion(pred, target)
        self.values["dice"] += dice_score(pred, target).tolist()
        self.values["iou"] += iou_score(pred, target).tolist()
        self.values["boundary_iou"] += boundary_iou(pred, target).tolist()
        self.values["precision"] += (tp / (tp + fp).clamp_min(1)).tolist()
        self.values["recall"] += (tp / (tp + fn).clamp_min(1)).tolist()
        p_np, t_np = pred.cpu().numpy(), target.cpu().numpy()
        prob_np = prob.cpu().numpy()
        unc_np = uncertainty.cpu().numpy() if uncertainty is not None else None
        for i in range(p_np.shape[0]):
            if self.compute_hd95:
                self.values["hd95"].append(hd95(p_np[i, 0], t_np[i, 0]))
            self.values["ece"].append(expected_calibration_error(prob_np[i], t_np[i]))
            if unc_np is not None:
                err = p_np[i, 0] != t_np[i, 0]
                self.values["ueo"].append(uncertainty_error_overlap(unc_np[i, 0], err))
                self.values["unc_auroc"].append(auroc(unc_np[i, 0], err))

    def compute(self) -> Dict[str, float]:
        out = {}
        for k, v in self.values.items():
            arr = np.asarray(v, dtype=float)
            out[k] = float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")
        return out

    def reset(self) -> None:
        self.values.clear()
