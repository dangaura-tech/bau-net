#!/usr/bin/env python
"""Create reproducible train/val/test split files (one image stem per line).

Example:
    python scripts/make_splits.py --images data/kvasir/images --out splits/kvasir --ratios 0.8 0.1 0.1
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ratios", nargs=3, type=float, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-6:
        raise SystemExit("ratios must sum to 1")
    stems = sorted(f.stem for f in Path(args.images).iterdir() if f.suffix.lower() in IMG_EXTS)
    random.Random(args.seed).shuffle(stems)
    n = len(stems)
    n_train = int(n * args.ratios[0])
    n_val = int(n * args.ratios[1])
    splits = {
        "train": stems[:n_train],
        "val": stems[n_train : n_train + n_val],
        "test": stems[n_train + n_val :],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, items in splits.items():
        (out / f"{name}.txt").write_text("\n".join(sorted(items)) + "\n")
        print(f"{name:5s}: {len(items)}")


if __name__ == "__main__":
    main()
