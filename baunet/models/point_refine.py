"""Uncertainty-guided Point Refinement (UPR).

Dense decoders spend the same capacity on every pixel although errors concentrate on a
thin band around object contours. UPR re-predicts only the pixels where the coarse
prediction is least certain:

* **Selection score** ``s = -|z| + λ_b · σ(b)`` combines the margin of the coarse mask
  logit ``z`` (small margin = uncertain) with the predicted boundary probability ``σ(b)``,
  so selection is driven by both aleatoric ambiguity and geometric prior.
* **Training** uses importance sampling (Kirillov et al., PointRend): ``k·N`` random
  points are drawn, the ``β·N`` highest-scoring ones are kept and ``(1-β)·N`` uniformly
  random points are added for coverage.
* **Point head** - a shared MLP over the concatenation of fine decoder features and the
  coarse mask/boundary/SDF predictions at each point, predicting a *residual* to the
  coarse logit.
* **Inference** refines the top-``N`` scoring pixels of the full-resolution map in place.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def point_sample(inp: torch.Tensor, coords: torch.Tensor, **kwargs) -> torch.Tensor:
    """Sample ``inp`` (B, C, H, W) at normalised ``coords`` (B, P, 2) in [0, 1] (x, y) -> (B, C, P)."""
    grid = (2.0 * coords - 1.0).unsqueeze(2)
    return F.grid_sample(inp, grid, align_corners=False, **kwargs).squeeze(3)


def selection_score(mask_logits: torch.Tensor, boundary_logits: Optional[torch.Tensor], boundary_weight: float):
    score = -mask_logits.abs()
    if boundary_logits is not None and boundary_weight > 0:
        score = score + boundary_weight * torch.sigmoid(boundary_logits)
    return score


@torch.no_grad()
def sample_training_points(
    score_map: torch.Tensor, num_points: int, oversample: float = 3.0, importance: float = 0.75
) -> torch.Tensor:
    """Importance-sample ``num_points`` coordinates (B, N, 2) biased towards high scores."""
    b = score_map.shape[0]
    k = max(int(num_points * oversample), num_points)
    cand = torch.rand(b, k, 2, device=score_map.device)
    s = point_sample(score_map, cand)[:, 0]  # (B, k)
    n_imp = int(importance * num_points)
    idx = s.topk(n_imp, dim=1).indices
    picked = torch.gather(cand, 1, idx[..., None].expand(-1, -1, 2))
    rand = torch.rand(b, num_points - n_imp, 2, device=score_map.device)
    return torch.cat([picked, rand], dim=1)


@torch.no_grad()
def top_uncertain_pixels(score_map: torch.Tensor, num_points: int):
    """Return flat indices (B, N) and centre coordinates (B, N, 2) of the top-``N`` pixels."""
    b, _, h, w = score_map.shape
    n = min(num_points, h * w)
    idx = score_map.view(b, -1).topk(n, dim=1).indices
    xs = (idx % w).float().add_(0.5).div_(w)
    ys = torch.div(idx, w, rounding_mode="floor").float().add_(0.5).div_(h)
    return idx, torch.stack([xs, ys], dim=-1)


class PointHead(nn.Module):
    """Shared MLP (1x1 Conv1d) predicting a residual logit for each sampled point."""

    def __init__(self, fine_channels: int, coarse_channels: int = 3, hidden: int = 128, layers: int = 3) -> None:
        super().__init__()
        mlp = []
        c = fine_channels + coarse_channels
        for _ in range(layers):
            mlp += [nn.Conv1d(c, hidden, 1), nn.ReLU(inplace=True)]
            c = hidden
        self.mlp = nn.Sequential(*mlp)
        self.out = nn.Conv1d(hidden, 1, 1)
        nn.init.zeros_(self.out.weight)  # refinement starts as identity
        nn.init.zeros_(self.out.bias)

    def forward(self, fine: torch.Tensor, coarse: torch.Tensor, coarse_logit: torch.Tensor) -> torch.Tensor:
        return coarse_logit + self.out(self.mlp(torch.cat([fine, coarse], dim=1)))


class UncertaintyPointRefiner(nn.Module):
    def __init__(
        self,
        fine_channels: int,
        train_points: int = 2048,
        eval_points: int = 8192,
        oversample: float = 3.0,
        importance: float = 0.75,
        boundary_weight: float = 1.0,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.head = PointHead(fine_channels, coarse_channels=3, hidden=hidden)
        self.train_points = train_points
        self.eval_points = eval_points
        self.oversample = oversample
        self.importance = importance
        self.boundary_weight = boundary_weight

    @staticmethod
    def _coarse_stack(
        mask: torch.Tensor, boundary: Optional[torch.Tensor], sdf: Optional[torch.Tensor]
    ) -> torch.Tensor:
        zeros = torch.zeros_like(mask)
        return torch.cat([mask, boundary if boundary is not None else zeros, sdf if sdf is not None else zeros], 1)

    def forward(
        self,
        fine: torch.Tensor,
        mask: torch.Tensor,
        boundary: Optional[torch.Tensor] = None,
        sdf: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        coarse = self._coarse_stack(mask, boundary, sdf)
        score = selection_score(mask.detach(), None if boundary is None else boundary.detach(), self.boundary_weight)
        if self.training:
            coords = sample_training_points(score, self.train_points, self.oversample, self.importance)
            logits = self.head(point_sample(fine, coords), point_sample(coarse, coords), point_sample(mask, coords))
            return {"mask": mask, "point_coords": coords, "point_logits": logits}
        idx, coords = top_uncertain_pixels(score, self.eval_points)
        logits = self.head(point_sample(fine, coords), point_sample(coarse, coords), point_sample(mask, coords))
        b, c, h, w = mask.shape
        refined = mask.reshape(b, -1).scatter(1, idx, logits[:, 0]).view(b, c, h, w)
        return {"mask": refined, "mask_coarse": mask, "point_coords": coords}
