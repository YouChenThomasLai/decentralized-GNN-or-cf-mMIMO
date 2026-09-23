#!/usr/bin/env bash
# Sum-rate versus the number of AP antennas M: matched-budget R0 and G2 training.
# One training seed per operating point; runs launch concurrently because the
# training step is CPU-bound, so several processes share one GPU efficiently.
set -euo pipefail

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

ris_python=${RIS_PYTHON:-python}
device=${DEVICE:-cuda:0}
artifact_root=${ARTIFACT_ROOT:-../../artifacts/decentralized_ris}
out=${OUT_DIR:-$artifact_root/scratch/runs/antenna_sweep}
n_iter=${N_ITER:-150000}
antennas=${ANTENNAS:-"1 2 3 4 5"}
archs=${ARCHS:-"r0 g2"}
seed=${SEED:-0}

export OMP_NUM_THREADS=${THREADS:-2}
export MKL_NUM_THREADS=${THREADS:-2}
export PYTHONUNBUFFERED=1

mkdir -p "$out"
for arch in $archs; do
    for antenna in $antennas; do
        test ! -e "$out/msweep-$arch-M${antenna}_iter${n_iter}_seed${seed}"
    done
done

pids=()
tags=()
for arch in $archs; do
    for antenna in $antennas; do
        tag="msweep-$arch-M$antenna"
        "$ris_python" train.py \
            --arch "$arch" --M "$antenna" --N 30 --L 4 --K 8 --AP 5 \
            --pmax_dbm 15 --batch_size 8 --n_iter "$n_iter" --seed "$seed" \
            --device "$device" --tag "$tag" --out_dir "$out" \
            > "$out/$tag.log" 2>&1 &
        pids+=("$!")
        tags+=("$tag")
        echo "[launch] $tag pid=$! log=$out/$tag.log"
    done
done

status=0
for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
        echo "[ok] ${tags[$index]}"
    else
        echo "[fail] ${tags[$index]}"
        status=1
    fi
done
echo "[sweep-done] status=$status"
exit "$status"
