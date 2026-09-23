#!/usr/bin/env bash
# E10: train the parameter-matched conventional DNN control family.
#
#   d0  centralized-input MLP that emits every action at once
#   d1  AP-shared MLP behind G2's energy-consensus AP-to-CPU interface
#
# Budget, optimizer, batch size, loss, seed, and validation cadence match the
# E06 G2-150k finalist; only the representation changes.  ARMS selects one arm
# so the two trainings can occupy separate tmux windows on the same host.
set -euo pipefail

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

ris_python=${RIS_PYTHON:-python}
device=${DEVICE:-cuda:0}
artifact_root=${ARTIFACT_ROOT:-../../artifacts/decentralized_ris}
out="$artifact_root/e10_dnn_benchmark"
arms=${ARMS:-"d0 d1"}
iters=${N_ITER:-150000}
seed=${SEED:-0}
width=${DNN_WIDTH:-40}
depth=${DNN_DEPTH:-3}
eval_seed=${EVAL_SEED:-20260915}
final_samples=${FINAL_SAMPLES:-400}

for arch in $arms; do
  test ! -e "$out/dnn-$arch-${iters}_iter${iters}_seed${seed}"
done
mkdir -p "$out"

for arch in $arms; do
  tag="dnn-$arch-${iters}"
  PYTHONUNBUFFERED=1 "$ris_python" train.py \
    --arch "$arch" --n_iter "$iters" --seed "$seed" \
    --dnn_width "$width" --dnn_depth "$depth" \
    --eval_seed "$eval_seed" --test_sample_final "$final_samples" \
    --device "$device" --tag "$tag" --out_dir "$out" \
    2>&1 | tee "$out/$tag.log"
  test -f "$out/${tag}_iter${iters}_seed${seed}/checkpoints/best.pt"
done
