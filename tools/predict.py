#!/usr/bin/env python
"""Run inference on images and save masks, overlays and MC-dropout uncertainty maps.

Example:
    python tools/predict.py --checkpoint runs/kvasir_seg/best.pth --input demo/ --output predictions/ --mc-samples 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baunet.data.datasets import IMG_EXTS  # noqa: E402
from baunet.data.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from baunet.models import build_model  # noqa: E402


def to_heatmap(x: np.ndarray) -> np.ndarray:
    """Map values in [0, 1] to an RGB 'inferno-like' colour ramp without matplotlib."""
    x = np.clip(x, 0, 1)
    r = np.clip(1.5 * x, 0, 1)
    g = np.clip(1.5 * x - 0.5, 0, 1)
    b = np.clip(np.sin(np.pi * x) * 0.8, 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="image file or directory")
    p.add_argument("--output", default="predictions")
    p.add_argument("--mc-samples", type=int, default=0, help=">0 enables Monte-Carlo dropout uncertainty")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    device = torch.device(args.device)
    model.to(device).eval()
    size = tuple(cfg["data"].get("image_size", [352, 352]))
    mean = torch.tensor(cfg["data"].get("mean", IMAGENET_MEAN))[:, None, None]
    std = torch.tensor(cfg["data"].get("std", IMAGENET_STD))[:, None, None]

    src = Path(args.input)
    files = [src] if src.is_file() else sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXTS)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    for f in files:
        rgb = np.array(Image.open(f).convert("RGB"))
        h, w = rgb.shape[:2]
        x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        x = F.interpolate(x[None], size=size, mode="bilinear", align_corners=False)
        if cfg["data"].get("normalize", True):
            x = (x - mean) / std
        x = x.to(device)
        with torch.no_grad():
            if args.mc_samples > 0:
                res = model.predict_with_uncertainty(x, n_samples=args.mc_samples)
                prob, unc = res["prob"], res["mutual_info"]
            else:
                prob, unc = torch.sigmoid(model(x)["mask"]), None
        prob = F.interpolate(prob, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        mask = prob > args.threshold
        Image.fromarray((mask * 255).astype(np.uint8)).save(out / f"{f.stem}_mask.png")
        overlay = rgb.copy()
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
        Image.fromarray(overlay).save(out / f"{f.stem}_overlay.png")
        if unc is not None:
            u = F.interpolate(unc, size=(h, w), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
            u = u / (u.max() + 1e-8)
            Image.fromarray(to_heatmap(u)).save(out / f"{f.stem}_uncertainty.png")
        print(f"{f.name}: foreground {mask.mean() * 100:.1f}%")


if __name__ == "__main__":
    main()
