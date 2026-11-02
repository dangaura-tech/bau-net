#!/usr/bin/env python
"""Train BAU-Net.

Examples:
    python tools/train.py --config configs/kvasir_seg.yaml
    python tools/train.py --config configs/isic2018.yaml --work-dir runs/isic train.lr=5e-4
    python tools/train.py --config configs/synthetic.yaml --device cpu        # smoke test
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baunet.data import build_dataloaders  # noqa: E402
from baunet.engine import Trainer  # noqa: E402
from baunet.losses import build_loss  # noqa: E402
from baunet.models import build_model  # noqa: E402
from baunet.utils import get_logger, load_config, merge_overrides, seed_everything  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--work-dir", default=None, help="output directory (default: runs/<config name>)")
    p.add_argument("--resume", default=None, help="checkpoint to resume from")
    p.add_argument("--device", default=None)
    p.add_argument("overrides", nargs="*", help="config overrides, e.g. train.epochs=50")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = merge_overrides(load_config(args.config), args.overrides)
    seed_everything(cfg.get("seed", 42), deterministic=cfg.get("deterministic", False))
    work_dir = args.work_dir or str(Path("runs") / Path(args.config).stem)
    device = torch.device(args.device) if args.device else None

    train_loader, val_loader = build_dataloaders(cfg)
    model = build_model(cfg)
    logger = get_logger()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(
        f"model: BAU-Net ({n_params:.2f} M params) | train {len(train_loader.dataset)} | val {len(val_loader.dataset)}"
    )

    trainer = Trainer(model, build_loss(cfg), train_loader, val_loader, cfg, work_dir, device)
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit()


if __name__ == "__main__":
    main()
