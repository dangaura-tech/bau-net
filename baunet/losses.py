"""Loss functions for boundary-aware segmentation."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def mask_to_boundary(mask: torch.Tensor, width: int = 3) -> torch.Tensor:
    """Extract a boundary band of roughly ``width`` pixels from a binary mask.

    Uses the morphological gradient (dilation - erosion) implemented with max-pooling,
    so it runs on the GPU and is differentiable-free (used only for targets).

    Args:
        mask: tensor of shape (B, 1, H, W) with values in {0, 1}.
    """
    k = 2 * (width // 2) + 1
    pad = k // 2
    dilated = F.max_pool2d(mask, k, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, k, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1)


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
    """Total objective  L = L_dice + L_bce + lambda_b * L_boundary (+ deep supervision).

    The boundary term is a class-balanced BCE between the boundary head and a boundary
    band extracted from the ground-truth mask; positive pixels are re-weighted because
    boundaries occupy only a small fraction of the image.
    """

    def __init__(
        self,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        boundary_weight: float = 0.5,
        boundary_width: int = 3,
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
        if "aux" in outputs:
            aux = torch.stack([self._seg(a, target) for a in outputs["aux"]]).mean()
            losses["aux"] = self.aux_weight * aux
        losses["total"] = sum(losses.values())
        return losses


def build_loss(cfg: Dict) -> BoundaryAwareLoss:
    return BoundaryAwareLoss(**cfg.get("loss", {}))
