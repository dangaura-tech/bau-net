#!/usr/bin/env python
"""Train BAU-Net briefly on synthetic shapes and render a qualitative figure
(input | ground truth | prediction | uncertainty) - a CPU-only sanity demo, no data needed.

    python tools/demo_figure.py --epochs 25 --out assets/demo_synthetic.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baunet.data import SyntheticShapesDataset  # noqa: E402
from baunet.data.datasets import build_dataloaders  # noqa: E402
from baunet.engine import Trainer  # noqa: E402
from baunet.losses import build_loss  # noqa: E402
from baunet.models import build_model  # noqa: E402
from baunet.utils import load_config, merge_overrides, seed_everything  # noqa: E402


def heat(x: np.ndarray) -> np.ndarray:
    """Map [0, 1] to an inferno-like RGB ramp without matplotlib."""
    x = np.clip(x, 0, 1)
    rgb = np.stack([np.clip(1.5 * x, 0, 1), np.clip(1.5 * x - 0.5, 0, 1), np.clip(np.sin(np.pi * x) * 0.8, 0, 1)], -1)
    return (rgb * 255).astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--size", type=int, default=96)
    p.add_argument("--samples", type=int, default=4)
    p.add_argument("--out", default="assets/demo_synthetic.png")
    p.add_argument("--work-dir", default="runs/demo_figure")
    args = p.parse_args()

    seed_everything(0)
    cfg = merge_overrides(
        load_config("configs/synthetic.yaml"),
        [f"train.epochs={args.epochs}", f"data.image_size=[{args.size},{args.size}]", "data.train_size=256",
         "train.log_interval=1000"],
    )  # fmt: skip
    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg)
    Trainer(model, build_loss(cfg), train_loader, val_loader, cfg, args.work_dir, torch.device("cpu")).fit()

    ds = SyntheticShapesDataset(n=args.samples, size=args.size, seed=123)
    s = args.size * 2
    pad, head = 6, 22
    cols = ["input", "ground truth", "prediction", "uncertainty"]
    canvas = Image.new("RGB", (4 * (s + pad) + pad, head + args.samples * (s + pad) + pad), (13, 17, 23))
    draw = ImageDraw.Draw(canvas)
    for c, name in enumerate(cols):
        draw.text((pad + c * (s + pad) + 4, 5), name, fill=(220, 220, 220))
    model.eval()
    for r in range(args.samples):
        item = ds[r]
        x = item["image"][None]
        res = model.predict_with_uncertainty(x, n_samples=8, tta=True)
        img = (item["image"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        gt = (item["mask"][0].numpy() * 255).astype(np.uint8)
        pred = ((res["prob"][0, 0] > 0.5).numpy() * 255).astype(np.uint8)
        unc = res["entropy"][0, 0].numpy()
        unc = heat(unc / (unc.max() + 1e-8))
        tiles = [img, np.stack([gt] * 3, -1), np.stack([pred] * 3, -1), unc]
        for c, t in enumerate(tiles):
            tile = Image.fromarray(t).resize((s, s), Image.NEAREST)
            canvas.paste(tile, (pad + c * (s + pad), head + r * (s + pad)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"figure written to {args.out}")


if __name__ == "__main__":
    main()
