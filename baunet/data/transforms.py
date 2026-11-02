"""Joint image/mask augmentations implemented with plain torch ops.

Every transform receives and returns ``(image, mask)`` where ``image`` is a float tensor
(C, H, W) in [0, 1] and ``mask`` is a float tensor (1, H, W) in {0, 1}.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F

Pair = Tuple[torch.Tensor, torch.Tensor]


class Compose:
    def __init__(self, transforms: List[Callable[[torch.Tensor, torch.Tensor], Pair]]) -> None:
        self.transforms = transforms

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Pair:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


class Resize:
    def __init__(self, size: Sequence[int]) -> None:
        self.size = tuple(size)

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Pair:
        image = F.interpolate(image[None], size=self.size, mode="bilinear", align_corners=False)[0]
        mask = F.interpolate(mask[None], size=self.size, mode="nearest")[0]
        return image, mask


class RandomFlip:
    def __init__(self, p_h: float = 0.5, p_v: float = 0.5) -> None:
        self.p_h, self.p_v = p_h, p_v

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Pair:
        if random.random() < self.p_h:
            image, mask = image.flip(-1), mask.flip(-1)
        if random.random() < self.p_v:
            image, mask = image.flip(-2), mask.flip(-2)
        return image, mask


class RandomAffine:
    """Random rotation + isotropic scaling applied identically to image and mask."""

    def __init__(self, degrees: float = 30.0, scale: Sequence[float] = (0.85, 1.15), p: float = 0.7) -> None:
        self.degrees = degrees
        self.scale = scale
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Pair:
        if random.random() > self.p:
            return image, mask
        angle = math.radians(random.uniform(-self.degrees, self.degrees))
        s = random.uniform(*self.scale)
        cos, sin = math.cos(angle) / s, math.sin(angle) / s
        theta = torch.tensor([[cos, -sin, 0.0], [sin, cos, 0.0]], dtype=image.dtype)[None]
        grid = F.affine_grid(theta, (1, *image.shape), align_corners=False)
        image = F.grid_sample(image[None], grid, mode="bilinear", padding_mode="reflection", align_corners=False)[0]
        mask = F.grid_sample(mask[None], grid, mode="nearest", padding_mode="zeros", align_corners=False)[0]
        return image, mask


class ColorJitter:
    def __init__(self, brightness: float = 0.2, contrast: float = 0.2, p: float = 0.8) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.p = p

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Pair:
        if random.random() < self.p:
            b = 1 + random.uniform(-self.brightness, self.brightness)
            c = 1 + random.uniform(-self.contrast, self.contrast)
            mean = image.mean(dim=(-2, -1), keepdim=True)
            image = ((image - mean) * c + mean) * b
        return image.clamp(0, 1), mask


class Normalize:
    def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
        self.mean = torch.tensor(mean)[:, None, None]
        self.std = torch.tensor(std)[:, None, None]

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> Pair:
        return (image - self.mean) / self.std, mask


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(cfg: Dict, train: bool) -> Compose:
    size = cfg.get("image_size", [352, 352])
    ts: List[Callable] = [Resize(size)]
    if train and cfg.get("augment", True):
        ts += [RandomFlip(), RandomAffine(), ColorJitter()]
    if cfg.get("normalize", True):
        ts.append(Normalize(cfg.get("mean", IMAGENET_MEAN), cfg.get("std", IMAGENET_STD)))
    return Compose(ts)
