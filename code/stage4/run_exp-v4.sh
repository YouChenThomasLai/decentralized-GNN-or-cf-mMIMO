#!/bin/bash

set -euo pipefail

stage1_run="${STAGE1C_RUN_DIR:-../results_stage1c_bpp_noise_1e-12/M2_K8_P15.0/run0}"
ap_coordinates="${stage1_run}/arrays/BS_0.txt"
stage1_config="${stage1_run}/config.json"
stage2_root="${STAGE2_REFERENCE_ROOT:-../stage2/results_stage2_bpp}"
output_root="${STAGE4_OUTPUT_ROOT:-results_stage4_seed0}"

for artifact in "${ap_coordinates}" "${stage1_config}"; do
    if [[ ! -f "${artifact}" ]]; then
        echo "Frozen Stage 1C artifact not found: ${artifact}" >&2
        exit 1
    fi
done
if [[ -e "${output_root}" ]]; then
    echo "Stage 4 output already exists: ${output_root}" >&2
    exit 1
fi

STAGE1C_RUN_DIR="${stage1_run}" python test_stage4.py

common=(
    --M 2
    --K 8
    --pmax_dbm 15
    --noise_power 1e-12
    --seed 0
    --scheduler_seed 4000
    --decision_period_s 0.001
    --carrier_frequency_hz 2.6e9
    --episode_steps 2000
    --trajectories 10
    --eval_time_stride 1
    --batch_size 256
    --budgets 0 1 2 3 8
    --schedulers round_robin random mobility_age_priority
    --ap_coordinates "${ap_coordinates}"
    --stage1_config "${stage1_config}"
    --device cuda:0
)

speeds=(0 30 80)
for speed_kmh in "${speeds[@]}"; do
    reference="${stage2_root}/straight_${speed_kmh}_kmh"
    if [[ ! -f "${reference}/completion.json" ]]; then
        echo "Stage 2 reference not found: ${reference}" >&2
        exit 1
    fi
done

mkdir -p "${output_root}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

pids=()
for speed_kmh in "${speeds[@]}"; do
    reference="${stage2_root}/straight_${speed_kmh}_kmh"
    (
        python evaluate.py \
            "${common[@]}" \
            --speed_kmh "${speed_kmh}" \
            --stage2_reference "${reference}" \
            --out_dir "${output_root}/straight_${speed_kmh}_kmh"
    ) >"${output_root}/straight_${speed_kmh}_kmh.launch.log" 2>&1 &
    pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
    if ! wait "${pids[index]}"; then
        echo "Stage 4 speed ${speeds[index]} km/h failed" >&2
        failed=1
    fi
done
if ((failed)); then
    exit 1
fi

python evaluate.py --aggregate_root "${output_root}"
