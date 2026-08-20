#!/bin/bash

set -euo pipefail

phase=${1:-smoke}
python_bin=${PYTHON_BIN:-python}
device=${DEVICE:-cuda:0}

run_scale() {
    local scale=$1 ap=$2 ue=$3 side=$4 seed=$5 runs=$6 iterations=$7
    local batch_size=$8 evaluation_samples=$9 result_group=${10}
    local out_dir="../results_snapshot_scaling/${result_group}/stage1_no_ris/scale${scale}_A${ap}_K${ue}"
    mkdir -p "${out_dir}"
    "${python_bin}" trainer_2.py \
        --scale "${scale}" --num_ap "${ap}" --K "${ue}" \
        --square_side "${side}" --M 2 --pmax_dbm 15 \
        --noise_power 1e-12 --batch_size "${batch_size}" \
        --seed "${seed}" --runs "${runs}" --n_iter "${iterations}" \
        --test_sample_val "${evaluation_samples}" \
        --test_sample_final "${evaluation_samples}" \
        --device "${device}" --out_dir "${out_dir}" \
        2>&1 | tee "${out_dir}/seeds${seed}-$((seed + runs - 1)).log"
}

case "${phase}" in
    smoke)
        run_scale 1 5 8 177.2453850905516 0 1 2 1 2 smoke
        run_scale 2 10 16 250.66282746310003 0 1 2 1 2 smoke
        ;;
    trend)
        run_scale 1 5 8 177.2453850905516 0 3 2000 8 3200 .
        run_scale 2 10 16 250.66282746310003 0 3 2000 8 3200 .
        run_scale 4 20 32 354.4907701811032 0 1 2000 8 3200 .
        run_scale 4 20 32 354.4907701811032 1 2 2000 8 3200 .
        ;;
    *)
        echo "Usage: $0 [smoke|trend]" >&2
        exit 2
        ;;
esac
