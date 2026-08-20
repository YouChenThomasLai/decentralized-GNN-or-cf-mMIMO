#!/bin/bash

set -euo pipefail

default_checkpoint="../stage1/results_stage1b_full_noise_1e-12/"
default_checkpoint+="M2_K8_P15.0/run0/models/model_final_run0.pt"
checkpoint="${STAGE1_CHECKPOINT:-${default_checkpoint}}"
output_root="${STAGE2_OUTPUT_ROOT:-results_stage2_mobility}"

if [[ ! -f "${checkpoint}" ]]; then
    echo "Stage 1B checkpoint not found: ${checkpoint}" >&2
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
    --out_dir "${output_root}/hotspot_low_stickiness_diagnostics"

python trainer_2.py \
    --diagnostics_only \
    --mobility_model hotspot_semi_markov \
    --speed_kmh 30 \
    --hotspot_stickiness 0.8 \
    --hotspot_dwell_mean_s 10 \
    --seed 0 \
    --trajectories 10 \
    --out_dir "${output_root}/hotspot_high_stickiness_diagnostics"
