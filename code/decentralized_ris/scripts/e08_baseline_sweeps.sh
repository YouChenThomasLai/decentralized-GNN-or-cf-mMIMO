#!/usr/bin/env bash
# E08: reproduce the short-budget R0 antenna and power sweeps.
set -euo pipefail

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

ris_python=${RIS_PYTHON:-python}
device=${DEVICE:-cuda:0}
artifact_root=${ARTIFACT_ROOT:-../../artifacts/decentralized_ris}
common_args=(
    --N 30 --L 4 --K 8 --batch_size 8
    --n_iter 2000 --test_sample_final 3200 --device "$device"
)

for antennas in 1 2 3 4 5; do
    "$ris_python" train.py "${common_args[@]}" --M "$antennas" --pmax_dbm 15 \
        --out_dir "$artifact_root/e08_baseline_sweeps/vary_m"
done

for power in 5 10 15 20 25 30 35; do
    "$ris_python" train.py "${common_args[@]}" --M 2 --pmax_dbm "$power" \
        --out_dir "$artifact_root/e08_baseline_sweeps/vary_pmax"
done
