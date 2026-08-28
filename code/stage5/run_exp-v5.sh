#!/bin/bash

set -euo pipefail

stage1_run="${STAGE1C_RUN_DIR:-../results_stage1c_bpp_noise_1e-12/M2_K8_P15.0/run0}"
ap_coordinates="${STAGE1_AP_COORDINATES:-${stage1_run}/arrays/BS_0.txt}"
stage1_config="${STAGE1_CONFIG:-${stage1_run}/config.json}"
stage3_root="${STAGE3_REFERENCE_ROOT:-../stage3/results_stage3b_seed0_dual_eval_straight}"
stage4_root="${STAGE4_REFERENCE_ROOT:-../stage4/results_stage4_seed0}"
boundary_root="${STAGE5_BOUNDARY_ROOT:-results_stage5_seed0_boundary}"
output_root="${STAGE5_OUTPUT_ROOT:-results_stage5_seed0}"

for artifact in "${ap_coordinates}" "${stage1_config}" \
    "${stage3_root}/completion.json" "${stage3_root}/provenance.json"; do
    if [[ ! -f "${artifact}" ]]; then
        echo "Frozen parent artifact not found: ${artifact}" >&2
        exit 1
    fi
done
for speed_kmh in 0 30 80; do
    for artifact in \
        "${stage3_root}/straight_${speed_kmh}_kmh/association_traces.npz" \
        "${stage3_root}/straight_${speed_kmh}_kmh/raw_metrics.npz" \
        "${stage4_root}/straight_${speed_kmh}_kmh/completion.json" \
        "${stage4_root}/straight_${speed_kmh}_kmh/config.json" \
        "${stage4_root}/straight_${speed_kmh}_kmh/provenance.json" \
        "${stage4_root}/straight_${speed_kmh}_kmh/environment.npz" \
        "${stage4_root}/straight_${speed_kmh}_kmh/update_traces.npz" \
        "${stage4_root}/straight_${speed_kmh}_kmh/csi_state_metrics.npz" \
        "${stage4_root}/straight_${speed_kmh}_kmh/raw_metrics.npz"; do
        if [[ ! -f "${artifact}" ]]; then
            echo "Frozen parent artifact not found: ${artifact}" >&2
            exit 1
        fi
    done
done
for root in "${boundary_root}" "${output_root}"; do
    if [[ -e "${root}" ]]; then
        echo "Stage 5 output already exists: ${root}" >&2
        exit 1
    fi
done

STAGE1C_RUN_DIR="${stage1_run}" python test_stage5.py

common=(
    --M 2
    --K 8
    --pmax_dbm 15
    --noise_power 1e-12
    --seed 0
    --decision_period_s 0.001
    --carrier_frequency_hz 2.6e9
    --episode_steps 2000
    --trajectories 10
    --eval_time_stride 1
    --batch_size 256
    --ap_coordinates "${ap_coordinates}"
    --stage1_config "${stage1_config}"
    --stage3_reference_root "${stage3_root}"
    --require_parent_reproduction
)

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "${boundary_root}"

for speed_kmh in 0 30 80; do
    python evaluate.py \
        "${common[@]}" \
        --phase boundary \
        --device cuda:0 \
        --stage4_boundary_device cpu \
        --speed_kmh "${speed_kmh}" \
        --stage4_reference "${stage4_root}/straight_${speed_kmh}_kmh" \
        --out_dir "${boundary_root}/straight_${speed_kmh}_kmh"
done
python evaluate.py --aggregate_root "${boundary_root}" --phase boundary

mkdir -p "${output_root}"
for speed_kmh in 0 30 80; do
    python evaluate.py \
        "${common[@]}" \
        --phase main \
        --device cuda:0 \
        --learned_diagnostics \
        --speed_kmh "${speed_kmh}" \
        --stage4_reference "${stage4_root}/straight_${speed_kmh}_kmh" \
        --boundary_reference "${boundary_root}/straight_${speed_kmh}_kmh" \
        --out_dir "${output_root}/straight_${speed_kmh}_kmh"
done
python evaluate.py \
    --aggregate_root "${output_root}" \
    --phase main \
    --boundary_root "${boundary_root}"

echo "Stage 5A boundary and modular joint matrix completed."
