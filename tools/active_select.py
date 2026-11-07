#!/usr/bin/env python
"""Rank unlabelled images for annotation (uncertainty-driven active learning).

Writes a CSV with every image's acquisition scores and marks the ``--budget`` images selected
for labelling. With ``--diverse`` the batch is chosen by k-centre over bottleneck embeddings
within the most uncertain candidates, seeded with ``--labelled`` images if given.

Example:
    python tools/active_select.py --checkpoint runs/kvasir_seg/best.pth --unlabelled pool/images \\
        --labelled data/Kvasir-SEG/train/images --budget 50 --strategy boundary --diverse --out next_batch.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baunet.active import (  # noqa: E402
    STRATEGIES,
    acquisition_scores,
    bottleneck_embedding,
    diverse_select,
    topk_select,
)
from baunet.data.datasets import IMG_EXTS  # noqa: E402
from baunet.data.transforms import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from baunet.models import build_model  # noqa: E402


def list_images(folder: str | None) -> list[Path]:
    if not folder:
        return []
    return sorted(f for f in Path(folder).iterdir() if f.suffix.lower() in IMG_EXTS)


def load_batch(files, size, cfg, device) -> torch.Tensor:
    mean = torch.tensor(cfg["data"].get("mean", IMAGENET_MEAN))[:, None, None]
    std = torch.tensor(cfg["data"].get("std", IMAGENET_STD))[:, None, None]
    xs = []
    for f in files:
        x = torch.from_numpy(np.array(Image.open(f).convert("RGB"))).permute(2, 0, 1).float() / 255.0
        x = F.interpolate(x[None], size=size, mode="bilinear", align_corners=False)[0]
        xs.append((x - mean) / std if cfg["data"].get("normalize", True) else x)
    return torch.stack(xs).to(device)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--unlabelled", required=True, help="folder of candidate images")
    p.add_argument("--labelled", default=None, help="folder of already labelled images (diversity seed)")
    p.add_argument("--budget", type=int, default=20)
    p.add_argument("--strategy", choices=STRATEGIES, default="boundary")
    p.add_argument("--diverse", action="store_true", help="k-centre selection within the uncertain pool")
    p.add_argument("--pool-factor", type=float, default=3.0)
    p.add_argument("--mc-samples", type=int, default=8)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", default="active_selection.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt["config"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    device = torch.device(args.device)
    model.to(device).eval()
    size = tuple(cfg["data"].get("image_size", [352, 352]))

    files = list_images(args.unlabelled)
    if not files:
        raise SystemExit(f"no images in {args.unlabelled}")
    scores = {s: [] for s in STRATEGIES}
    emb = []
    for s in range(0, len(files), args.batch_size):
        x = load_batch(files[s : s + args.batch_size], size, cfg, device)
        res = model.predict_with_uncertainty(x, n_samples=args.mc_samples, tta=args.tta)
        for k, v in acquisition_scores(res).items():
            scores[k].extend(v.cpu().tolist())
        if args.diverse:
            emb.append(bottleneck_embedding(model, x).cpu().numpy())
        print(f"\r{min(s + args.batch_size, len(files))}/{len(files)}", end="", file=sys.stderr)
    print(file=sys.stderr)

    primary = np.array(scores[args.strategy])
    if args.diverse:
        labelled = None
        lab_files = list_images(args.labelled)
        if lab_files:
            chunks = [lab_files[s : s + args.batch_size] for s in range(0, len(lab_files), args.batch_size)]
            labelled = np.concatenate(
                [bottleneck_embedding(model, load_batch(c, size, cfg, device)).cpu().numpy() for c in chunks]
            )
        picked = diverse_select(primary, np.concatenate(emb), args.budget, args.pool_factor, labelled)
    else:
        picked = topk_select(primary, args.budget)
    order = {int(i): r + 1 for r, i in enumerate(picked)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["image", "selected_rank", *STRATEGIES])
        for i in np.argsort(-primary):
            w.writerow([files[i].name, order.get(int(i), ""), *(f"{scores[k][i]:.6f}" for k in STRATEGIES)])
    mode = args.strategy + (", diverse" if args.diverse else "")
    print(f"selected {len(picked)} / {len(files)} images ({mode}) -> {args.out}")


if __name__ == "__main__":
    main()
