#!/bin/bash

set -euo pipefail

phase=${1:-smoke}
python_bin=${PYTHON_BIN:-python}
device=${DEVICE:-cuda:0}

run_scale() {
    local scale=$1 ap=$2 ue=$3 ris=$4 spatial=$5 seed=$6 runs=$7
    local iterations=$8 batch_size=$9 evaluation_samples=${10} result_group=${11}
    local out_dir="../results_snapshot_scaling/${result_group}/stage0_ris/scale${scale}_A${ap}_K${ue}_L${ris}"
    mkdir -p "${out_dir}"
    "${python_bin}" trainer_2.py \
        --scale "${scale}" --num_ap "${ap}" --K "${ue}" --L "${ris}" \
        --spatial_scale "${spatial}" --M 2 --N 30 --pmax_dbm 15 \
        --batch_size "${batch_size}" --seed "${seed}" --runs "${runs}" \
        --n_iter "${iterations}" --test_sample_val "${evaluation_samples}" \
        --test_sample_final "${evaluation_samples}" --device "${device}" \
        --out_dir "${out_dir}" \
        2>&1 | tee "${out_dir}/seeds${seed}-$((seed + runs - 1)).log"
}

case "${phase}" in
    smoke)
        run_scale 1 5 8 4 100 0 1 2 1 2 smoke
        run_scale 2 10 16 8 141.4213562373095 0 1 2 1 2 smoke
        ;;
    trend)
        run_scale 1 5 8 4 100 0 3 2000 8 3200 .
        run_scale 2 10 16 8 141.4213562373095 0 3 2000 8 3200 .
        run_scale 4 20 32 16 200 0 1 2000 8 3200 .
        run_scale 4 20 32 16 200 1 2 2000 8 3200 .
        ;;
    *)
        echo "Usage: $0 [smoke|trend]" >&2
        exit 2
        ;;
esac
