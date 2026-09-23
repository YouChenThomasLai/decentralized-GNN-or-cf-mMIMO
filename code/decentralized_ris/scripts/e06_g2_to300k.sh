#!/usr/bin/env bash
# E06: continue the retained G2 trajectory from 150k to 300k.
set -euo pipefail

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

ris_python=${RIS_PYTHON:-python}
device=${DEVICE:-cuda:0}
artifact_root=${ARTIFACT_ROOT:-../../artifacts/decentralized_ris}
out="$artifact_root/e06_graph_energy_training/g2_long_training/g2"
ckpt150=${G2_150K_CHECKPOINT:-"$out/iter150000/checkpoints/iter150000.pt"}
run300="$out/iter300000"
staged300="$out/graph-energy-g2-total300k-resume150k_iter150000_seed0"
log="$out/graph-energy-g2-total300k-resume150k.log"

test -f "$ckpt150"
test ! -e "$run300"
test ! -e "$staged300"
test ! -e "$log"
mkdir -p "$out"

PYTHONUNBUFFERED=1 "$ris_python" train.py \
  --arch g2 --n_iter 150000 --seed 0 \
  --device "$device" --tag graph-energy-g2-total300k-resume150k \
  --resume "$ckpt150" --out_dir "$out" \
  2>&1 | tee "$log"

mv "$staged300" "$run300"
test -f "$run300/checkpoints/last.pt"
cp "$run300/checkpoints/last.pt" "$run300/checkpoints/iter300000.pt"
