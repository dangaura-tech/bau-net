"""Datasets.

``FolderSegmentationDataset`` reads any dataset organised as::

    root/
      images/  xxx.jpg | xxx.png ...
      masks/   xxx.png            (same stem, foreground > 127)

which covers Kvasir-SEG, CVC-ClinicDB, ISIC 2018 Task 1 and BUSI after running the
preparation scripts in ``scripts/``. Splits are defined by plain-text files listing stems.

``SyntheticShapesDataset`` procedurally generates noisy blob images for smoke tests and CI.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import build_transforms

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


class FolderSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str,
        split_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        image_dir: str = "images",
        mask_dir: str = "masks",
        mask_suffix: str = "",
    ) -> None:
        self.root = Path(root)
        self.image_dir = self.root / image_dir
        self.mask_dir = self.root / mask_dir
        self.mask_suffix = mask_suffix
        self.transform = transform
        images = {p.stem: p for p in sorted(self.image_dir.iterdir()) if p.suffix.lower() in IMG_EXTS}
        if split_file:
            stems = [s.strip() for s in Path(split_file).read_text().splitlines() if s.strip()]
            missing = [s for s in stems if s not in images]
            if missing:
                raise FileNotFoundError(f"{len(missing)} images listed in {split_file} not found, e.g. {missing[0]}")
        else:
            stems = list(images)
        self.items: List[Tuple[Path, Path]] = [(images[s], self._mask_path(s)) for s in stems]
        if not self.items:
            raise RuntimeError(f"no images found in {self.image_dir}")

    def _mask_path(self, stem: str) -> Path:
        for ext in (".png", ".jpg", ".bmp", ".tif"):
            p = self.mask_dir / f"{stem}{self.mask_suffix}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"mask for '{stem}' not found in {self.mask_dir}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.items[idx]
        image = torch.from_numpy(np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0)
        image = image.permute(2, 0, 1).contiguous()
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 127
        mask_t = torch.from_numpy(mask.astype(np.float32))[None]
        if self.transform is not None:
            image, mask_t = self.transform(image, mask_t)
        return {"image": image, "mask": mask_t, "name": img_path.stem}


class SyntheticShapesDataset(Dataset):
    """Random ellipses / irregular blobs on textured noise, with exact masks."""

    def __init__(self, n: int = 256, size: int = 128, seed: int = 0, transform: Optional[Callable] = None) -> None:
        self.n = n
        self.size = size
        self.seed = seed
        self.transform = transform

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.seed * 100_003 + idx)
        s = self.size
        yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
        cy, cx = rng.uniform(0.3 * s, 0.7 * s, size=2)
        ry, rx = rng.uniform(0.1 * s, 0.25 * s, size=2)
        angle = rng.uniform(0, 2 * math.pi)
        theta = np.arctan2(yy - cy, xx - cx)
        wobble = 1 + 0.15 * np.sin(rng.integers(3, 7) * theta + angle)
        r = np.sqrt(((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2)
        mask = (r < wobble).astype(np.float32)
        base = rng.uniform(0.2, 0.5, size=3).astype(np.float32)
        fg = rng.uniform(0.5, 0.9, size=3).astype(np.float32)
        image = base[:, None, None] + mask[None] * (fg - base)[:, None, None]
        image += rng.normal(0, 0.08, size=image.shape).astype(np.float32)
        image_t = torch.from_numpy(np.clip(image, 0, 1))
        mask_t = torch.from_numpy(mask)[None]
        if self.transform is not None:
            image_t, mask_t = self.transform(image_t, mask_t)
        return {"image": image_t, "mask": mask_t, "name": f"synthetic_{idx:05d}"}


def build_datasets(cfg: Dict) -> Tuple[Dataset, Dataset]:
    data = cfg["data"]
    train_tf = build_transforms(data, train=True)
    val_tf = build_transforms(data, train=False)
    if data["name"] == "synthetic":
        size = data.get("image_size", [128, 128])[0]
        return (
            SyntheticShapesDataset(data.get("train_size", 256), size, seed=0, transform=train_tf),
            SyntheticShapesDataset(data.get("val_size", 64), size, seed=1, transform=val_tf),
        )
    root = data["root"]
    return (
        FolderSegmentationDataset(root, data.get("train_split"), train_tf),
        FolderSegmentationDataset(root, data.get("val_split"), val_tf),
    )


def build_dataloaders(cfg: Dict) -> Tuple[DataLoader, DataLoader]:
    train_ds, val_ds = build_datasets(cfg)
    t = cfg.get("train", {})
    workers = t.get("num_workers", 4)
    train = DataLoader(
        train_ds,
        batch_size=t.get("batch_size", 8),
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val = DataLoader(val_ds, batch_size=t.get("eval_batch_size", 8), shuffle=False, num_workers=workers)
    return train, val
