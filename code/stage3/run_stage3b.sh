#!/bin/bash

set -euo pipefail

mode="${1:-smoke}"
stage1_run="${STAGE1C_RUN_DIR:-../results_stage1c_bpp_noise_1e-12/M2_K8_P15.0/run0}"
checkpoint="${STAGE1_CHECKPOINT:-${stage1_run}/models/model_final_run0.pt}"
ap_coordinates="${STAGE1_AP_COORDINATES:-${stage1_run}/arrays/BS_0.txt}"
stage1_config="${STAGE1_CONFIG:-${stage1_run}/config.json}"
stage3a_evidence="${STAGE3A_EVIDENCE:-results_stage3a_bpp/straight_30_kmh}"
output_root="${STAGE3B_OUTPUT_ROOT:-results_stage3b_straight_${mode}}"

if [[ "${mode}" != "smoke" && "${mode}" != "pilot" ]]; then
    echo "Usage: $0 [smoke|pilot]" >&2
    exit 2
fi
for artifact in "${checkpoint}" "${ap_coordinates}" "${stage1_config}"; do
    if [[ ! -f "${artifact}" ]]; then
        echo "Frozen Stage 1C artifact not found: ${artifact}" >&2
        exit 1
    fi
done
if [[ ! -f "${stage3a_evidence}/completion.json" ]]; then
    echo "Completed moving Stage 3A evidence not found: ${stage3a_evidence}" >&2
    exit 1
fi
if [[ -e "${output_root}" ]]; then
    echo "Stage 3B output already exists: ${output_root}" >&2
    exit 1
fi

python test_stage3b.py

common=(
    --policy_seed 0
    --lambda_switch 0.5
    --checkpoint "${checkpoint}"
    --ap_coordinates "${ap_coordinates}"
    --stage1_config "${stage1_config}"
    --stage3a_evidence "${stage3a_evidence}"
    --device cuda:0
)

if [[ "${mode}" == "smoke" ]]; then
    common+=(
        --smoke
        --max_steps 4
        --warmup_steps 1
        --batch_size 2
        --replay_capacity 32
        --validation_interval 2
        --early_stop_after 100
        --training_trajectories 4
        --validation_trajectories 4
        --generation_batch_size 1
        --episode_steps 100
        --reward_time_stride 10
    )
else
    common+=(
        --max_steps 500000
        --warmup_steps 400
        --batch_size 256
        --replay_capacity 1000000
        --validation_interval 10000
        --early_stop_after 100000
        --early_stop_patience 10
        --training_trajectories 256
        --validation_trajectories 16
        --generation_batch_size 8
        --episode_steps 2000
        --reward_time_stride 10
    )
fi

status=0
for variant in current history; do
    if ! python train_association.py \
        "${common[@]}" \
        --variant "${variant}" \
        --out_dir "${output_root}/sac_${variant}_seed0_lambda0.5"; then
        status=1
    fi
done

if [[ "${status}" -ne 0 ]]; then
    echo "One or more Stage 3B ${mode} variants failed." >&2
    exit "${status}"
fi
echo "Stage 3B ${mode} completed."
