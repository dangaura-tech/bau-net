#!/usr/bin/env python
"""Export a trained BAU-Net checkpoint to ONNX and verify it against PyTorch.

Examples:
    python tools/export_onnx.py --checkpoint runs/kvasir_seg/best.pth --out exports/baunet_kvasir.onnx
    python tools/export_onnx.py --config configs/synthetic.yaml --out exports/tiny.onnx   # random weights
    python tools/export_onnx.py --checkpoint ... --no-refine     # coarse mask only (smaller, faster graph)

Requires: pip install -e ".[onnx]"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baunet.export import calibrate_batchnorm, export_onnx, verify_onnx  # noqa: E402
from baunet.models import build_model  # noqa: E402
from baunet.utils import load_config, merge_overrides  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="checkpoint written by tools/train.py (contains its config)")
    src.add_argument("--config", help="build an untrained model from a config (for testing the pipeline)")
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, nargs=2, default=None, metavar=("H", "W"),
                   help="input size (default: data.image_size of the config)")  # fmt: skip
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--static-batch", action="store_true", help="fix the batch dimension to 1")
    p.add_argument("--no-refine", action="store_true", help="export without uncertainty-guided point refinement")
    p.add_argument("--no-verify", action="store_true", help="skip the ONNX Runtime parity check")
    p.add_argument("overrides", nargs="*")
    args = p.parse_args()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        cfg = merge_overrides(ckpt["config"], args.overrides)
        cfg["model"]["pretrained"] = False
        model = build_model(cfg)
        model.load_state_dict(ckpt["model"])
    else:
        cfg = merge_overrides(load_config(args.config), args.overrides)
        cfg["model"]["pretrained"] = False
        model = build_model(cfg)
    size = args.size or cfg["data"].get("image_size", [352, 352])
    if args.config:  # untrained weights: give BatchNorm realistic statistics first
        calibrate_batchnorm(model, size)
    path = export_onnx(model, args.out, size, args.opset, not args.static_batch, refine=not args.no_refine)
    print(f"exported {path} ({path.stat().st_size / 1e6:.1f} MB) for input 1x3x{size[0]}x{size[1]}")
    if not args.no_verify:
        x = torch.randn(2 if not args.static_batch else 1, 3, size[0], size[1])
        stats = verify_onnx(str(path), model, x, refine=not args.no_refine)
        print(
            f"ONNX Runtime vs PyTorch: max abs difference {stats['max_abs_diff']:.2e}, "
            f"pixels differing > 1e-3: {stats['mismatch_fraction']:.4%}"
        )
        if stats["mismatch_fraction"] > 1e-3:
            raise SystemExit("parity check failed")


if __name__ == "__main__":
    main()
