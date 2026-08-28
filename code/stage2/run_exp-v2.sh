#!/bin/bash

set -euo pipefail

default_checkpoint="../results_stage1c_bpp_noise_1e-12/"
default_checkpoint+="M2_K8_P15.0/run0/models/model_final_run0.pt"
checkpoint="${STAGE1_CHECKPOINT:-${default_checkpoint}}"
default_ap_coordinates="../results_stage1c_bpp_noise_1e-12/"
default_ap_coordinates+="M2_K8_P15.0/run0/arrays/BS_0.txt"
ap_coordinates="${STAGE1_AP_COORDINATES:-${default_ap_coordinates}}"
default_stage1_config="../results_stage1c_bpp_noise_1e-12/"
default_stage1_config+="M2_K8_P15.0/run0/config.json"
stage1_config="${STAGE1_CONFIG:-${default_stage1_config}}"
output_root="${STAGE2_OUTPUT_ROOT:-results_stage2_bpp}"

if [[ ! -f "${checkpoint}" ]]; then
    echo "Stage 1C checkpoint not found: ${checkpoint}" >&2
    exit 1
fi
for artifact in "${ap_coordinates}" "${stage1_config}"; do
    if [[ ! -f "${artifact}" ]]; then
        echo "Stage 1C artifact not found: ${artifact}" >&2
        exit 1
    fi
done
if [[ -e "${output_root}" ]]; then
    echo "Stage 2 output already exists: ${output_root}" >&2
    exit 1
fi

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
    --checkpoint "${checkpoint}"
    --ap_coordinates "${ap_coordinates}"
    --stage1_config "${stage1_config}"
    --device cuda:0
)

for speed_kmh in 0 30 80; do
    python trainer_2.py \
        "${common[@]}" \
        --mobility_model straight \
        --speed_kmh "${speed_kmh}" \
        --out_dir "${output_root}/straight_${speed_kmh}_kmh"
done

for speed_kmh in 3 30 80; do
    python trainer_2.py \
        "${common[@]}" \
        --mobility_model hotspot_semi_markov \
        --speed_kmh "${speed_kmh}" \
        --out_dir "${output_root}/hotspot_${speed_kmh}_kmh"
done

python trainer_2.py \
    --diagnostics_only \
    --mobility_model hotspot_semi_markov \
    --speed_kmh 30 \
    --hotspot_stickiness 0.2 \
    --hotspot_dwell_mean_s 2 \
    --seed 0 \
    --trajectories 10 \
    --ap_coordinates "${ap_coordinates}" \
    --stage1_config "${stage1_config}" \
    --out_dir "${output_root}/hotspot_low_stickiness_diagnostics"

python trainer_2.py \
    --diagnostics_only \
    --mobility_model hotspot_semi_markov \
    --speed_kmh 30 \
    --hotspot_stickiness 0.8 \
    --hotspot_dwell_mean_s 10 \
    --seed 0 \
    --trajectories 10 \
    --ap_coordinates "${ap_coordinates}" \
    --stage1_config "${stage1_config}" \
    --out_dir "${output_root}/hotspot_high_stickiness_diagnostics"
