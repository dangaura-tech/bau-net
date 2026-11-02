#!/usr/bin/env bash
# Download Kvasir-SEG and create the standard 80/10/10 split.
# Dataset: Jha et al., "Kvasir-SEG: A Segmented Polyp Dataset", MMM 2020.
# License: research and educational use only - see https://datasets.simula.no/kvasir-seg/
set -euo pipefail

DST=${1:-data/kvasir}
URL="https://datasets.simula.no/downloads/kvasir-seg.zip"

mkdir -p "$(dirname "$DST")"
if [ ! -d "$DST/images" ]; then
  tmp=$(mktemp -d)
  echo "Downloading Kvasir-SEG ..."
  curl -L --fail -o "$tmp/kvasir-seg.zip" "$URL"
  unzip -q "$tmp/kvasir-seg.zip" -d "$tmp"
  mkdir -p "$DST"
  mv "$tmp/Kvasir-SEG/images" "$DST/images"
  mv "$tmp/Kvasir-SEG/masks" "$DST/masks"
  rm -rf "$tmp"
fi

python scripts/make_splits.py --images "$DST/images" --out splits/kvasir
echo "Done. Train with: python tools/train.py --config configs/kvasir_seg.yaml"
