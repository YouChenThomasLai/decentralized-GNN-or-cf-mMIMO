#!/usr/bin/env bash
# E02: evaluate centralized, paper-decentralized, and own-only trajectories.
# Sum rate versus training steps in all three CSI visibility modes.
#
# Every checkpoint named iter<iteration>.pt under CKPT_DIR is evaluated with
# experiments.local_csi, which scores centralized, paper-decentralized and
# own-only inference on the same channel realizations, so the three curves are
# paired at every point of the trajectory.
set -euo pipefail

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

ris_python=${RIS_PYTHON:-python}
device=${DEVICE:-cuda:0}
samples=${SAMPLES:-800}
artifact_root=${ARTIFACT_ROOT:-../../artifacts/decentralized_ris}
ckpt_dir=${CKPT_DIR:-trajectory_ckpts}
out=${OUT_DIR:-$artifact_root/e02_input_modes}

shopt -s nullglob
checkpoints=("$ckpt_dir"/iter*.pt)
if [ ${#checkpoints[@]} -eq 0 ]; then
    echo "[error] no iter*.pt checkpoints under $ckpt_dir" >&2
    exit 1
fi

for checkpoint in "${checkpoints[@]}"; do
    cell=$(basename "$checkpoint" .pt)
    if [ -f "$out/$cell/summary.json" ]; then
        echo "[skip] $cell already evaluated"
        continue
    fi
    echo "[run] $cell -> $out/$cell"
    "$ris_python" -m experiments.local_csi \
        --ckpt "$checkpoint" --samples "$samples" --device "$device" \
        --out_dir "$out/$cell"
done

"$ris_python" -m experiments.summarize_trajectory --root "$out" \
    --out "$out/trajectory.txt" --json "$out/trajectory.json"
