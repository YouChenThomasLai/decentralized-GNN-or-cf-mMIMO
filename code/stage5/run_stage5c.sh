#!/bin/bash

set -euo pipefail

phase="${1:-all}"
if [[ ! " smoke pilot evaluate formal all " =~ " ${phase} " ]]; then
    echo "Usage: $0 [smoke|pilot|evaluate|formal|all]" >&2
    exit 2
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_bin="${PYTHON_BIN}"
elif [[ -x "${HOME}/miniforge3/envs/decentralized-inference/bin/python" ]]; then
    python_bin="${HOME}/miniforge3/envs/decentralized-inference/bin/python"
elif [[ -x "${HOME}/miniconda3/envs/decentralized-inference/bin/python" ]]; then
    python_bin="${HOME}/miniconda3/envs/decentralized-inference/bin/python"
else
    python_bin="python"
fi

default_stage1_run="../results_stage1c_bpp_noise_1e-12/M2_K8_P15.0/run0"
if [[ ! -d "${default_stage1_run}" ]]; then
    default_stage1_run="../stage1/remote_backup_2026-08-24/results_stage1c_bpp_noise_1e-12/M2_K8_P15.0/run0"
fi
stage1_run="${STAGE1C_RUN_DIR:-${default_stage1_run}}"
checkpoint="${STAGE1_CHECKPOINT:-${stage1_run}/models/model_final_run0.pt}"
ap_coordinates="${STAGE1_AP_COORDINATES:-${stage1_run}/arrays/BS_0.txt}"
stage1_config="${STAGE1_CONFIG:-${stage1_run}/config.json}"
gate_root="${STAGE5B_GATE_ROOT:-results_stage5b_gnn_gate_seed0}"
smoke_root="${STAGE5B_SMOKE_ROOT:-results_stage5b_rl_smoke_seed0}"
pilot_root="${STAGE5B_PILOT_ROOT:-results_stage5b_rl_pilot_seed0}"
evaluation_root="${STAGE5B_EVALUATION_ROOT:-results_stage5b_rl_evaluation_seed0}"
formal_root="${STAGE5B_FORMAL_ROOT:-results_stage5b_rl_formal}"
device="${STAGE5B_DEVICE:-cuda:0}"

for artifact in "${checkpoint}" "${ap_coordinates}" "${stage1_config}" \
    "${gate_root}/completion.json" "${gate_root}/summary.json"; do
    if [[ ! -f "${artifact}" ]]; then
        echo "Required frozen artifact not found: ${artifact}" >&2
        exit 1
    fi
done
"${python_bin}" - "${gate_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
completion = json.loads((root / "completion.json").read_text())
summary = json.loads((root / "summary.json").read_text())
if completion.get("status") != "complete" or not summary.get("completed"):
    raise SystemExit("Gate 5.5 must pass before RL execution")
PY

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

require_complete() {
    "${python_bin}" - "$1" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Required completion artifact not found: {path}")
if json.loads(path.read_text()).get("status") != "complete":
    raise SystemExit(f"Required predecessor did not complete: {path}")
PY
}

train_one() {
    local beamformer="$1"
    local output="$2"
    local seed="$3"
    shift 3
    if [[ -e "${output}" ]]; then
        echo "Fresh output required; already exists: ${output}" >&2
        exit 1
    fi
    "${python_bin}" train_rl.py \
        --beamformer "${beamformer}" \
        --policy_seed "${seed}" \
        --gate5_5_root "${gate_root}" \
        --checkpoint "${checkpoint}" \
        --ap_coordinates "${ap_coordinates}" \
        --stage1_config "${stage1_config}" \
        --device "${device}" \
        --out_dir "${output}" \
        "$@"
}

run_smoke() {
    mkdir -p "${smoke_root}"
    train_one rzf "${smoke_root}/pi_rzf" 0 \
        --max_steps 12 --warmup_steps 4 --batch_size 4 \
        --replay_capacity 128 --validation_interval 12 \
        --training_trajectories 2 --validation_trajectories 1 \
        --episode_steps 100 --smoke
    train_one decentralized_gnn "${smoke_root}/pi_d" 0 \
        --max_steps 12 --warmup_steps 4 --batch_size 4 \
        --replay_capacity 128 --validation_interval 12 \
        --training_trajectories 2 --validation_trajectories 1 \
        --episode_steps 100 --smoke
    train_one centralized_gnn "${smoke_root}/pi_c" 0 \
        --max_steps 12 --warmup_steps 4 --batch_size 4 \
        --replay_capacity 128 --validation_interval 12 \
        --training_trajectories 2 --validation_trajectories 1 \
        --episode_steps 100 --smoke
    "${python_bin}" - "${smoke_root}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
for policy in ("pi_rzf", "pi_d", "pi_c"):
    if json.loads((root / policy / "completion.json").read_text()).get("status") != "complete":
        raise SystemExit(f"Incomplete smoke: {policy}")
(root / "completion.json").write_text('{"status": "complete"}\n')
PY
}

run_pilot() {
    require_complete "${smoke_root}/completion.json"
    mkdir -p "${pilot_root}"
    common=(
        --max_steps "${PILOT_MAX_STEPS:-256}"
        --warmup_steps "${PILOT_WARMUP_STEPS:-32}"
        --batch_size 32
        --replay_capacity 4096
        --validation_interval "${PILOT_VALIDATION_INTERVAL:-64}"
        --training_trajectories 8
        --validation_trajectories 4
        --episode_steps 500
    )
    train_one rzf "${pilot_root}/pi_rzf" 0 "${common[@]}"
    train_one decentralized_gnn "${pilot_root}/pi_d" 0 "${common[@]}"
    train_one centralized_gnn "${pilot_root}/pi_c" 0 "${common[@]}"
}

evaluate_three() {
    local policies="$1"
    local output="$2"
    local environment_seed="$3"
    if [[ -e "${output}" ]]; then
        echo "Fresh output required; already exists: ${output}" >&2
        exit 1
    fi
    mkdir -p "${output}"
    for speed in 0 30 80; do
        "${python_bin}" evaluate_rl.py \
            --speed_kmh "${speed}" \
            --environment_seed "${environment_seed}" \
            --trajectories 10 --episode_steps 2000 \
            --gate5_5_root "${gate_root}" \
            --pi_rzf_root "${policies}/pi_rzf" \
            --pi_d_root "${policies}/pi_d" \
            --pi_c_root "${policies}/pi_c" \
            --checkpoint "${checkpoint}" \
            --ap_coordinates "${ap_coordinates}" \
            --device "${device}" \
            --out_dir "${output}/straight_${speed}_kmh"
    done
    "${python_bin}" evaluate_rl.py --aggregate_root "${output}"
}

run_evaluation() {
    for policy in pi_rzf pi_d pi_c; do
        require_complete "${pilot_root}/${policy}/completion.json"
    done
    evaluate_three "${pilot_root}" "${evaluation_root}" 0
}

run_formal() {
    require_complete "${evaluation_root}/completion.json"
    mkdir -p "${formal_root}"
    for seed in 0 1 2 3 4; do
        seed_root="${formal_root}/seed_${seed}"
        mkdir -p "${seed_root}/policies"
        common=(
            --max_steps "${FORMAL_MAX_STEPS:-2000}"
            --warmup_steps "${FORMAL_WARMUP_STEPS:-200}"
            --batch_size 64
            --replay_capacity 20000
            --validation_interval "${FORMAL_VALIDATION_INTERVAL:-200}"
            --training_trajectories 64
            --validation_trajectories 8
            --episode_steps 2000
            --formal
        )
        train_one rzf "${seed_root}/policies/pi_rzf" "${seed}" "${common[@]}"
        train_one decentralized_gnn "${seed_root}/policies/pi_d" "${seed}" "${common[@]}"
        train_one centralized_gnn "${seed_root}/policies/pi_c" "${seed}" "${common[@]}"
        evaluate_three "${seed_root}/policies" "${seed_root}/evaluation" "${seed}"
    done
    "${python_bin}" evaluate_rl.py --aggregate_formal_root "${formal_root}"
}

case "${phase}" in
    smoke) run_smoke ;;
    pilot) run_pilot ;;
    evaluate) run_evaluation ;;
    formal) run_formal ;;
    all)
        run_smoke
        run_pilot
        run_evaluation
        run_formal
        ;;
esac
