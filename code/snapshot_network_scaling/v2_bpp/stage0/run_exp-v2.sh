#!/bin/bash

set -euo pipefail

phase=${1:-implementation}
python_bin=${PYTHON_BIN:-python}
device=${DEVICE:-cuda:0}
results_root=${RESULTS_ROOT:-../../results_snapshot_scaling_v2_bpp}

scale_name() {
    case "$1" in
        1) echo "scale1_A5_K8_L4" ;;
        2) echo "scale2_A10_K16_L8" ;;
        4) echo "scale4_A20_K32_L16" ;;
    esac
}

run_scale() {
    local scale=$1 seed=$2 runs=$3 iterations=$4 batch_size=$5
    local evaluation_samples=$6 group=$7
    local name out_dir run_seed log_path
    name=$(scale_name "${scale}")
    if [[ "${group}" == "smoke" ]]; then
        out_dir="${results_root}/stage0_ris/smoke/${name}"
    else
        out_dir="${results_root}/stage0_ris/${name}"
    fi
    mkdir -p "${out_dir}"
    for ((offset = 0; offset < runs; offset++)); do
        run_seed=$((seed + offset))
        log_path="${out_dir}/launcher_seed${run_seed}.log"
        if [[ -e "${log_path}" ]]; then
            echo "Refusing to overwrite launcher log: ${log_path}" >&2
            exit 1
        fi
        "${python_bin}" trainer_2.py \
            --scale "${scale}" --M 2 --N 30 --pmax_dbm 15 \
            --batch_size "${batch_size}" --seed "${run_seed}" --runs 1 \
            --seed_extension_reason "${SEED_EXTENSION_REASON:-}" \
            --n_iter "${iterations}" --test_sample_val "${evaluation_samples}" \
            --test_sample_final "${evaluation_samples}" --device "${device}" \
            --out_dir "${out_dir}" \
            2>&1 | tee "${log_path}"
    done
}

require_phase() {
    "${python_bin}" gatekeeper.py "$1" --results-root "${results_root}"
}

case "${phase}" in
    implementation)
        "${python_bin}" test_stage0.py
        "${python_bin}" test_discrete_mapping.py
        ;;
    topology)
        mkdir -p "${results_root}/topology_gate"
        topology_log="${results_root}/topology_gate/launcher.log"
        if [[ -e "${topology_log}" ]]; then
            echo "Refusing to overwrite launcher log: ${topology_log}" >&2
            exit 1
        fi
        "${python_bin}" topology_gate.py \
            --output-root "${results_root}/topology_gate" \
            2>&1 | tee "${topology_log}"
        ;;
    approve-topology-red-flag)
        if [[ -z "${TOPOLOGY_REVIEW_REASON:-}" ]]; then
            echo "Set TOPOLOGY_REVIEW_REASON after manually reviewing the power drift." >&2
            exit 1
        fi
        "${python_bin}" topology_gate.py \
            --approve-red-flag "${results_root}/topology_gate/summary.json" \
            --review-reason "${TOPOLOGY_REVIEW_REASON}"
        ;;
    smoke)
        require_phase smoke
        run_scale 1 0 1 "${SMOKE_ITERATIONS:-100}" 8 "${SMOKE_SAMPLES:-128}" smoke
        run_scale 2 0 1 "${SMOKE_ITERATIONS:-100}" 8 "${SMOKE_SAMPLES:-128}" smoke
        ;;
    trend)
        require_phase trend
        run_scale 1 0 3 2000 8 3200 trend
        run_scale 2 0 3 2000 8 3200 trend
        ;;
    scale4-seed0)
        require_phase scale4_seed0
        run_scale 4 0 1 2000 8 3200 trend
        ;;
    scale4-seeds12)
        require_phase scale4_seeds12
        run_scale 4 1 2 2000 8 3200 trend
        ;;
    extend-scales12)
        if [[ "${ALLOW_SEED_EXTENSION:-0}" != 1 || -z "${SEED_EXTENSION_REASON:-}" ]]; then
            echo "Set ALLOW_SEED_EXTENSION=1 and SEED_EXTENSION_REASON after a Phase 5 condition." >&2
            exit 1
        fi
        require_phase extend_scales12
        run_scale 1 3 2 2000 8 3200 trend
        run_scale 2 3 2 2000 8 3200 trend
        ;;
    extend-scale4)
        if [[ "${ALLOW_SEED_EXTENSION:-0}" != 1 || -z "${SEED_EXTENSION_REASON:-}" ]]; then
            echo "Set ALLOW_SEED_EXTENSION=1 and SEED_EXTENSION_REASON after a Phase 5 condition." >&2
            exit 1
        fi
        require_phase extend_scale4
        run_scale 4 3 2 2000 8 3200 trend
        ;;
    *)
        echo "Usage: $0 {implementation|topology|approve-topology-red-flag|smoke|trend|scale4-seed0|scale4-seeds12|extend-scales12|extend-scale4}" >&2
        exit 2
        ;;
esac
