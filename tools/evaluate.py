#!/usr/bin/env python
"""Evaluate a trained checkpoint on a dataset split and write per-metric results.

Example:
    python tools/evaluate.py --checkpoint runs/kvasir_seg/best.pth \
        --config configs/kvasir_seg.yaml data.val_split=splits/kvasir/test.txt --out results/kvasir.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baunet.data.datasets import build_datasets  # noqa: E402
from baunet.engine import evaluate  # noqa: E402
from baunet.models import build_model  # noqa: E402
from baunet.utils import load_config, merge_overrides  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", default=None, help="defaults to the config stored in the checkpoint")
    p.add_argument("--out", default=None, help="optional JSON file for the metrics")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = load_config(args.config) if args.config else ckpt["config"]
    cfg = merge_overrides(cfg, args.overrides)
    _, val_ds = build_datasets(cfg)
    loader = DataLoader(val_ds, batch_size=cfg["train"].get("eval_batch_size", 8), num_workers=4)

    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    device = torch.device(args.device)
    metrics = evaluate(model.to(device), loader, device, compute_hd95=True, threshold=args.threshold)

    width = max(map(len, metrics))
    for k, v in metrics.items():
        print(f"{k:<{width}} : {v:.4f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"checkpoint": args.checkpoint, "n": len(val_ds), **metrics}, indent=2))


if __name__ == "__main__":
    main()
