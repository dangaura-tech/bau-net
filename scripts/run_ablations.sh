#!/usr/bin/env bash
# Train the full model, all ablations and backbone variants on Kvasir-SEG with 3 seeds,
# then evaluate on the test split with MC-dropout + TTA uncertainty metrics.
set -euo pipefail

SEEDS=${SEEDS:-"0 1 2"}
CONFIGS=(configs/kvasir_seg.yaml configs/ablations/*.yaml configs/backbones/*.yaml)

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "$cfg" .yaml)
  for seed in $SEEDS; do
    run="runs/ablations/${name}_s${seed}"
    python tools/train.py --config "$cfg" --work-dir "$run" seed="$seed"
    python tools/evaluate.py --checkpoint "$run/best.pth" --out "results/${name}_s${seed}.json" \
      --mc-samples 8 --tta data.val_split=splits/kvasir/test.txt
  done
done

python scripts/summarize_results.py results/
