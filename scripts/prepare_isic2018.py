#!/usr/bin/env python
"""Re-organise the official ISIC 2018 Task 1 download into images/ + masks/.

Download from https://challenge.isic-archive.com/data/#2018 :
    ISIC2018_Task1-2_Training_Input.zip
    ISIC2018_Task1_Training_GroundTruth.zip

Example:
    python scripts/prepare_isic2018.py --src ~/Downloads/isic2018 --dst data/isic2018
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder with the extracted official archives")
    p.add_argument("--dst", default="data/isic2018")
    args = p.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    img_dir = next(src.glob("*Training_Input*"), None)
    gt_dir = next(src.glob("*Training_GroundTruth*"), None)
    if img_dir is None or gt_dir is None:
        raise SystemExit("could not find *Training_Input* and *Training_GroundTruth* folders in --src")

    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "masks").mkdir(parents=True, exist_ok=True)
    n = 0
    for img in sorted(img_dir.glob("ISIC_*.jpg")):
        mask = gt_dir / f"{img.stem}_segmentation.png"
        if not mask.exists():
            print(f"warning: no mask for {img.name}", file=sys.stderr)
            continue
        shutil.copy2(img, dst / "images" / img.name)
        shutil.copy2(mask, dst / "masks" / f"{img.stem}.png")
        n += 1
    print(f"copied {n} image/mask pairs to {dst}")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("make_splits.py")),
            "--images",
            str(dst / "images"),
            "--out",
            "splits/isic2018",
            "--ratios",
            "0.7",
            "0.1",
            "0.2",
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
