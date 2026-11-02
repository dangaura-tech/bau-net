"""Segmentation metrics: Dice, IoU, precision, recall and 95% Hausdorff distance."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import numpy as np
import torch
from scipy import ndimage


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


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask ^ ndimage.binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)


def hd95(pred: np.ndarray, target: np.ndarray) -> float:
    """Symmetric 95th-percentile Hausdorff distance in pixels for 2-D binary masks.

    Returns 0 when both masks are empty and ``nan`` when exactly one is empty.
    """
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


class SegmentationMeter:
    """Accumulates per-image metrics over an evaluation run."""

    def __init__(self, threshold: float = 0.5, compute_hd95: bool = True) -> None:
        self.threshold = threshold
        self.compute_hd95 = compute_hd95
        self.values: Dict[str, list] = defaultdict(list)

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = (torch.sigmoid(logits) > self.threshold).float()
        tp, fp, fn = _confusion(pred, target)
        self.values["dice"] += dice_score(pred, target).tolist()
        self.values["iou"] += iou_score(pred, target).tolist()
        self.values["precision"] += (tp / (tp + fp).clamp_min(1)).tolist()
        self.values["recall"] += (tp / (tp + fn).clamp_min(1)).tolist()
        if self.compute_hd95:
            p = pred.cpu().numpy()
            t = target.cpu().numpy()
            for i in range(p.shape[0]):
                self.values["hd95"].append(hd95(p[i, 0], t[i, 0]))

    def compute(self) -> Dict[str, float]:
        return {k: float(np.nanmean(v)) if len(v) else float("nan") for k, v in self.values.items()}

    def reset(self) -> None:
        self.values.clear()
