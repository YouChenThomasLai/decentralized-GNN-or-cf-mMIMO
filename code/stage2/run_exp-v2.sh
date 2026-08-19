#!/bin/bash

set -euo pipefail

# One-run CPU smoke test (run separately):
# python trainer_2.py --M 2 --K 3 --batch_size 2 --runs 1 --n_iter 2 \
#   --episode_steps 8 --test_sample_val 2 --test_sample_final 2 \
#   --eval_frame_batch_size 2 --bootstrap_samples 20 --device cpu \
#   --out_dir results_stage2_debug

for speed_kmh in 0 3 30 80; do
    python trainer_2.py \
        --M 2 \
        --K 8 \
        --pmax_dbm 15 \
        --batch_size 8 \
        --runs 5 \
        --seed 0 \
        --n_iter 2000 \
        --noise_power 1e-12 \
        --speed_kmh "${speed_kmh}" \
        --decision_period_s 0.001 \
        --carrier_frequency_hz 2.6e9 \
        --episode_steps 2000 \
        --device cuda:0 \
        --out_dir "results_stage2_speed_${speed_kmh}"
done
