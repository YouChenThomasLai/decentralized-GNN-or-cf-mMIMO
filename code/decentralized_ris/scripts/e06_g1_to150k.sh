#!/usr/bin/env bash
# E06: continue the G1 trajectory from 40k to 100k and 150k, matching the G2 schedule.
set -euo pipefail

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

ris_python=${RIS_PYTHON:-python}
device=${DEVICE:-cuda:0}
artifact_root=${ARTIFACT_ROOT:-../../artifacts/decentralized_ris}
out="$artifact_root/e06_graph_energy_training/g1_long_training/g1"
ckpt40=${G1_40K_CHECKPOINT:-"$out/iter040000/checkpoints/last.pt"}
run100="$out/iter100000"
run150="$out/iter150000"
staged100="$out/graph-energy-g1-total100k-resume40k_iter60000_seed0"
staged150="$out/graph-energy-g1-total150k-resume100k_iter50000_seed0"

test -f "$ckpt40"
test ! -e "$run100"
test ! -e "$run150"
test ! -e "$staged100"
test ! -e "$staged150"
mkdir -p "$out"

PYTHONUNBUFFERED=1 "$ris_python" train.py \
  --arch g1 --n_iter 60000 --seed 0 \
  --device "$device" --tag graph-energy-g1-total100k-resume40k \
  --resume "$ckpt40" --out_dir "$out" \
  2>&1 | tee "$out/graph-energy-g1-total100k-resume40k.log"

mv "$staged100" "$run100"
test -f "$run100/checkpoints/last.pt"
cp "$run100/checkpoints/last.pt" "$run100/checkpoints/iter100000.pt"

PYTHONUNBUFFERED=1 "$ris_python" train.py \
  --arch g1 --n_iter 50000 --seed 0 \
  --device "$device" --tag graph-energy-g1-total150k-resume100k \
  --resume "$run100/checkpoints/iter100000.pt" --out_dir "$out" \
  2>&1 | tee "$out/graph-energy-g1-total150k-resume100k.log"

mv "$staged150" "$run150"
test -f "$run150/checkpoints/last.pt"
cp "$run150/checkpoints/last.pt" "$run150/checkpoints/iter150000.pt"
