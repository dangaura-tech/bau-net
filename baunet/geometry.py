"""Geometric targets derived from binary masks."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage


def mask_to_boundary(mask: torch.Tensor, width: int = 3) -> torch.Tensor:
    """Boundary band of roughly ``width`` pixels: dilation minus erosion (max-pooling, on device)."""
    k = 2 * (width // 2) + 1
    pad = k // 2
    dilated = F.max_pool2d(mask, k, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, k, stride=1, padding=pad)
    return (dilated - eroded).clamp(0, 1)


def erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    k = 2 * radius + 1
    return -F.max_pool2d(-mask, k, stride=1, padding=radius)


def signed_distance_np(mask: np.ndarray) -> np.ndarray:
    """Normalised signed distance field in [-1, 1]: negative inside, positive outside.

    Each side is normalised by its own maximum distance so that small and large objects
    contribute comparably. Empty / full masks map to +1 / -1 everywhere.
    """
    m = mask.astype(bool)
    if not m.any():
        return np.ones(m.shape, dtype=np.float32)
    if m.all():
        return -np.ones(m.shape, dtype=np.float32)
    outside = ndimage.distance_transform_edt(~m)
    inside = ndimage.distance_transform_edt(m)
    sdf = outside / outside.max() - inside / inside.max()
    return sdf.astype(np.float32)


@torch.no_grad()
def signed_distance(mask: torch.Tensor) -> torch.Tensor:
    """Batched :func:`signed_distance_np` for tensors of shape (B, 1, H, W)."""
    arr = mask.detach().cpu().numpy() > 0.5
    out = np.stack([signed_distance_np(a[0])[None] for a in arr])
    return torch.from_numpy(out).to(mask.device, mask.dtype)
