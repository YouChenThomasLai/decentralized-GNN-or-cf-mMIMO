#!/bin/bash

set -euo pipefail

stage1_run="${STAGE1C_RUN_DIR:-../results_stage1c_bpp_noise_1e-12/M2_K8_P15.0/run0}"
checkpoint="${STAGE1_CHECKPOINT:-${stage1_run}/models/model_final_run0.pt}"
ap_coordinates="${STAGE1_AP_COORDINATES:-${stage1_run}/arrays/BS_0.txt}"
stage1_config="${STAGE1_CONFIG:-${stage1_run}/config.json}"
stage2_root="${STAGE2_RESULTS_ROOT:-../stage2/results_stage2_bpp}"
output_root="${STAGE3_OUTPUT_ROOT:-results_stage3a_bpp}"

for artifact in "${checkpoint}" "${ap_coordinates}" "${stage1_config}"; do
    if [[ ! -f "${artifact}" ]]; then
        echo "Frozen Stage 1C artifact not found: ${artifact}" >&2
        exit 1
    fi
done
for setting in \
    straight_0_kmh straight_30_kmh straight_80_kmh \
    hotspot_3_kmh hotspot_30_kmh hotspot_80_kmh; do
    if [[ ! -f "${stage2_root}/${setting}/completion.json" ]]; then
        echo "Completed Stage 2 reference not found: ${setting}" >&2
        exit 1
    fi
done
if [[ -e "${output_root}" ]]; then
    echo "Stage 3 output already exists: ${output_root}" >&2
    exit 1
fi

export STAGE1C_RUN_DIR="${stage1_run}"
python test_stage3.py

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
    --eval_time_stride 10
    --batch_size 32
    --checkpoint "${checkpoint}"
    --ap_coordinates "${ap_coordinates}"
    --stage1_config "${stage1_config}"
    --device cuda:0
)

for speed_kmh in 0 30 80; do
    setting="straight_${speed_kmh}_kmh"
    python evaluate.py \
        "${common[@]}" \
        --mobility_model straight \
        --speed_kmh "${speed_kmh}" \
        --stage2_reference "${stage2_root}/${setting}" \
        --out_dir "${output_root}/${setting}"
done

for speed_kmh in 3 30 80; do
    setting="hotspot_${speed_kmh}_kmh"
    python evaluate.py \
        "${common[@]}" \
        --mobility_model hotspot_semi_markov \
        --speed_kmh "${speed_kmh}" \
        --stage2_reference "${stage2_root}/${setting}" \
        --out_dir "${output_root}/${setting}"
done

echo "Stage 3A six-setting sweep completed."
