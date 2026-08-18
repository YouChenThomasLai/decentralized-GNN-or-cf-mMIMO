#!/bin/bash

set -e

# One-run debug example (run separately):
# python trainer_2.py --M 2 --K 8 --pmax_dbm 15 --batch_size 8 --runs 1 \
#   --seed 0 --device cpu --out_dir results_stage1_debug

for antennas in 1 2 3 4 5; do
    python trainer_2.py \
        --M "${antennas}" \
        --K 8 \
        --pmax_dbm 15 \
        --batch_size 8 \
        --runs 5 \
        --seed 0 \
        --device cuda:0 \
        --out_dir results_stage1_vary_M
done

for pmax_dbm in 5 10 15 20 25 30 35; do
    python trainer_2.py \
        --M 2 \
        --K 8 \
        --pmax_dbm "${pmax_dbm}" \
        --batch_size 8 \
        --runs 5 \
        --seed 0 \
        --device cuda:0 \
        --out_dir results_stage1_vary_Pmax
done
