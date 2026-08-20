#!/bin/bash

set -euo pipefail

# Development gate: reuse the Stage 1B seed-0 model and evaluate mobility only.
default_checkpoint="../stage1/results_stage1b_full_noise_1e-12/"
default_checkpoint+="M2_K8_P15.0/run0/models/model_final_run0.pt"
checkpoint="${STAGE1_CHECKPOINT:-${default_checkpoint}}"
if [[ ! -f "${checkpoint}" ]]; then
    echo "Stage 1B checkpoint not found: ${checkpoint}" >&2
    exit 1
fi

for speed_kmh in 0 30 80; do
    python trainer_2.py \
        --M 2 \
        --K 8 \
        --pmax_dbm 15 \
        --batch_size 8 \
        --runs 1 \
        --seed 0 \
        --n_iter 2000 \
        --noise_power 1e-12 \
        --speed_kmh "${speed_kmh}" \
        --decision_period_s 0.001 \
        --carrier_frequency_hz 2.6e9 \
        --episode_steps 2000 \
        --train_trajectories 1 \
        --test_sample_val 1 \
        --test_sample_final 10 \
        --eval_time_stride 10 \
        --checkpoint "${checkpoint}" \
        --device cuda:0 \
        --out_dir "results_stage2a_frozen_stage1_seed0_speed_${speed_kmh}"
done
