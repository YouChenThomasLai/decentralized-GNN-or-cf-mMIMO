#!/bin/bash

set -euo pipefail

default_stage1_run="../results_stage1c_bpp_noise_1e-12/M2_K8_P15.0/run0"
if [[ ! -d "${default_stage1_run}" ]]; then
    default_stage1_run="../stage1/remote_backup_2026-08-24/results_stage1c_bpp_noise_1e-12/M2_K8_P15.0/run0"
fi
stage1_run="${STAGE1C_RUN_DIR:-${default_stage1_run}}"
checkpoint="${STAGE1_CHECKPOINT:-${stage1_run}/models/model_final_run0.pt}"
ap_coordinates="${STAGE1_AP_COORDINATES:-${stage1_run}/arrays/BS_0.txt}"
stage1_config="${STAGE1_CONFIG:-${stage1_run}/config.json}"
stage3_gnn_root="${STAGE3_GNN_REFERENCE_ROOT:-../stage3/results_stage3a_bpp}"
stage3_root="${STAGE3_REFERENCE_ROOT:-../stage3/results_stage3b_seed0_dual_eval_straight}"
stage4_root="${STAGE4_REFERENCE_ROOT:-../stage4/results_stage4_seed0}"
stage5a_root="${STAGE5A_REFERENCE_ROOT:-results_stage5_seed0_rerun2}"
output_root="${STAGE5B_OUTPUT_ROOT:-results_stage5b_gnn_gate_seed0}"

for artifact in "${checkpoint}" "${ap_coordinates}" "${stage1_config}" \
    "${stage3_root}/completion.json" "${stage3_root}/provenance.json" \
    "${stage5a_root}/completion.json"; do
    if [[ ! -f "${artifact}" ]]; then
        echo "Frozen artifact not found: ${artifact}" >&2
        exit 1
    fi
done
for speed_kmh in 0 30 80; do
    for artifact in \
        "${stage3_gnn_root}/straight_${speed_kmh}_kmh/completion.json" \
        "${stage3_gnn_root}/straight_${speed_kmh}_kmh/provenance.json" \
        "${stage3_gnn_root}/straight_${speed_kmh}_kmh/association_traces.npz" \
        "${stage3_gnn_root}/straight_${speed_kmh}_kmh/raw_metrics.npz" \
        "${stage4_root}/straight_${speed_kmh}_kmh/completion.json" \
        "${stage5a_root}/straight_${speed_kmh}_kmh/completion.json"; do
        if [[ ! -f "${artifact}" ]]; then
            echo "Frozen reference not found: ${artifact}" >&2
            exit 1
        fi
    done
done
if [[ -e "${output_root}" ]]; then
    echo "Stage 5B output already exists: ${output_root}" >&2
    exit 1
fi

STAGE1C_RUN_DIR="${stage1_run}" python test_stage5.py

common=(
    --phase gnn
    --beamformers rzf centralized_gnn decentralized_gnn
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
    --checkpoint "${checkpoint}"
    --ap_coordinates "${ap_coordinates}"
    --stage1_config "${stage1_config}"
    --stage3_reference_root "${stage3_root}"
    --require_parent_reproduction
    --device cuda:0
)

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "${output_root}"
for speed_kmh in 0 30 80; do
    python evaluate.py \
        "${common[@]}" \
        --speed_kmh "${speed_kmh}" \
        --stage4_reference "${stage4_root}/straight_${speed_kmh}_kmh" \
        --stage3_gnn_reference "${stage3_gnn_root}/straight_${speed_kmh}_kmh" \
        --stage5a_reference "${stage5a_root}/straight_${speed_kmh}_kmh" \
        --out_dir "${output_root}/straight_${speed_kmh}_kmh"
done
python evaluate.py --aggregate_root "${output_root}" --phase gnn

echo "Stage 5B Gate 5.5 frozen-GNN qualification completed."
