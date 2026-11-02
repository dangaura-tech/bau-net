"""Training objective of BAU-Net.

L = L_dice + L_bce                                (mask)
  + λ_b   · L_wbce(B̂, ∂M)                         (class-balanced boundary)
  + λ_sdf · ||Ŝ - SDF(M)||_1                      (shape prior)
  + λ_c   · ||σ(M̂) - σ(-κ·Ŝ)||_1                  (mask / SDF consistency)
  + λ_pt  · BCE(point logits, M(p))               (uncertainty-guided points)
  + λ_aux · mean_k L_seg(M̂_k, M)                  (deep supervision)
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import mask_to_boundary, signed_distance
from .models.point_refine import point_sample

__all__ = ["mask_to_boundary", "DiceLoss", "FocalLoss", "BoundaryAwareLoss", "build_loss"]


class DiceLoss(nn.Module):
    """Soft Dice loss computed per image and averaged over the batch."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(logits).flatten(1)
        target = target.flatten(1)
        inter = (prob * target).sum(1)
        denom = prob.sum(1) + target.sum(1)
        return (1 - (2 * inter + self.smooth) / (denom + self.smooth)).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = torch.exp(-bce)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (alpha_t * (1 - p_t) ** self.gamma * bce).mean()


class BoundaryAwareLoss(nn.Module):
    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        boundary_weight: float = 0.5,
        boundary_width: int = 3,
        sdf_weight: float = 0.5,
        consistency_weight: float = 0.2,
        consistency_kappa: float = 50.0,
        point_weight: float = 1.0,
        aux_weight: float = 0.4,
        focal: bool = False,
    ) -> None:
        super().__init__()
        self.dice = DiceLoss()
        self.pixel = FocalLoss() if focal else nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.boundary_weight = boundary_weight
        self.boundary_width = boundary_width
        self.sdf_weight = sdf_weight
        self.consistency_weight = consistency_weight
        self.kappa = consistency_kappa
        self.point_weight = point_weight
        self.aux_weight = aux_weight

    def _seg(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.dice_weight * self.dice(logits, target) + self.bce_weight * self.pixel(logits, target)

    def forward(self, outputs: Dict[str, torch.Tensor], target: torch.Tensor) -> Dict[str, torch.Tensor]:
        losses = {"seg": self._seg(outputs["mask"], target)}
        if "boundary" in outputs and self.boundary_weight > 0:
            b_target = mask_to_boundary(target, self.boundary_width)
            pos = b_target.mean().clamp_min(1e-4)
            pos_weight = ((1 - pos) / pos).clamp(max=50.0)
            losses["boundary"] = self.boundary_weight * F.binary_cross_entropy_with_logits(
                outputs["boundary"], b_target, pos_weight=pos_weight
            )
        if "sdf" in outputs:
            if self.sdf_weight > 0:
                losses["sdf"] = self.sdf_weight * F.l1_loss(outputs["sdf"], signed_distance(target))
            if self.consistency_weight > 0:
                from_sdf = torch.sigmoid(-self.kappa * outputs["sdf"])
                losses["consistency"] = self.consistency_weight * F.l1_loss(torch.sigmoid(outputs["mask"]), from_sdf)
        if "point_logits" in outputs and self.point_weight > 0:
            labels = point_sample(target, outputs["point_coords"], mode="nearest")
            losses["point"] = self.point_weight * F.binary_cross_entropy_with_logits(outputs["point_logits"], labels)
        if "aux" in outputs and self.aux_weight > 0:
            losses["aux"] = self.aux_weight * torch.stack([self._seg(a, target) for a in outputs["aux"]]).mean()
        losses["total"] = sum(losses.values())
        return losses


def build_loss(cfg: Dict) -> BoundaryAwareLoss:
    return BoundaryAwareLoss(**cfg.get("loss", {}))
