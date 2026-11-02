#!/usr/bin/env python
"""Aggregate ``results/*_s<seed>.json`` into a mean +- std Markdown table."""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ["dice", "iou", "precision", "recall", "hd95"]


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    groups = defaultdict(list)
    for f in sorted(root.glob("*.json")):
        name = re.sub(r"_s\d+$", "", f.stem)
        groups[name].append(json.loads(f.read_text()))
    if not groups:
        raise SystemExit(f"no result files in {root}")
    print("| Model | " + " | ".join(m.upper() for m in METRICS) + " |")
    print("|---" * (len(METRICS) + 1) + "|")
    for name, runs in groups.items():
        cells = []
        for m in METRICS:
            v = np.array([r[m] for r in runs if m in r], dtype=float)
            cells.append(f"{np.nanmean(v):.4f} ± {np.nanstd(v):.4f}" if len(v) else "-")
        print(f"| {name} (n={len(runs)}) | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
