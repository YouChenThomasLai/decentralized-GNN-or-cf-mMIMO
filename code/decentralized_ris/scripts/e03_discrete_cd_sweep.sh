#!/usr/bin/env bash
# E03: evaluate the greedy 2-bit phase reference over retained R0 sweeps.
set -euo pipefail

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

ris_python=${RIS_PYTHON:-python}
device=${DEVICE:-cuda:0}
rounds=${ROUNDS:-4}
artifact_root=${ARTIFACT_ROOT:-../../artifacts/decentralized_ris}
m_root="$artifact_root/e08_baseline_sweeps/vary_m"
p_root="$artifact_root/e08_baseline_sweeps/vary_pmax"
out="$artifact_root/e03_phase_headroom/discrete_cd"

run_cell() {
    local checkpoint=$1
    local output=$2
    local samples=$3
    local antennas=$4
    local power=$5
    if [ ! -f "$checkpoint" ]; then
        echo "[skip] missing $checkpoint"
        return
    fi
    "$ris_python" -m experiments.discrete_cd \
        --ckpt "$checkpoint" --M "$antennas" --pmax_dbm "$power" \
        --samples "$samples" --rounds "$rounds" --device "$device" \
        --out_dir "$output"
}

run_cell "$m_root/M2_N30_L4_K8_P15.0/run0/models/model_final_run0.pt" \
    "$out/headline_M2_P15" 800 2 15

for power in 5 10 15 20 25 30 35; do
    run_cell "$p_root/M2_N30_L4_K8_P${power}.0/run0/models/model_final_run0.pt" \
        "$out/vary_Pmax/P${power}" 160 2 "$power"
done

for antennas in 1 2 3 4 5; do
    run_cell "$m_root/M${antennas}_N30_L4_K8_P15.0/run0/models/model_final_run0.pt" \
        "$out/vary_M/M${antennas}" 160 "$antennas" 15
done
