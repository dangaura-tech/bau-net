#!/usr/bin/env python
"""Report parameter count, FLOPs (via torch.utils.flop_counter) and inference throughput."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baunet.models import build_model  # noqa: E402
from baunet.utils import load_config  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    cfg = load_config(args.config)
    device = torch.device(args.device)
    model = build_model(cfg).to(device).eval()
    h, w = cfg["data"].get("image_size", [352, 352])
    x = torch.randn(args.batch_size, cfg["model"].get("in_channels", 3), h, w, device=device)

    params = sum(p.numel() for p in model.parameters())
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as fc, torch.no_grad():
            model(x[:1])
        gflops = fc.get_total_flops() / 1e9
    except Exception:  # pragma: no cover - older torch
        gflops = float("nan")

    with torch.no_grad():
        for _ in range(5):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        tic = time.perf_counter()
        for _ in range(args.iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    fps = args.iters * args.batch_size / (time.perf_counter() - tic)
    print(f"params: {params / 1e6:.2f} M | GFLOPs: {gflops:.2f} @ {h}x{w} | throughput: {fps:.1f} img/s on {device}")


if __name__ == "__main__":
    main()
