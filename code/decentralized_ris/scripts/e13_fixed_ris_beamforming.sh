#!/usr/bin/env bash
# E13: active-beamforming controls under the frozen G2-150k RIS phase.
#
# Benchmark item 7 of doc/research_positioning.md.  The fused paper-decentralized
# G2 phase and its local-energy consensus are held fixed; only the AP-side active
# design changes, across the native G2 readout, RIS-aware local MRT and
# RIS-aware local RZF.  Nothing is trained here, so the run is evaluation only.
#
# The decision thresholds and the RZF regularizer live in
# experiments/fixed_ris_beamforming.py.  The artifact records the declared
# protocol; the current history does not establish an immutable pre-run commit.
set -euo pipefail

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

ris_python=${RIS_PYTHON:-python}
device=${DEVICE:-cuda:0}
samples=${SAMPLES:-400}
eval_seed=${EVAL_SEED:-20260918}
replication_seed=${REPLICATION_SEED:-20260915}
artifact_root=${ARTIFACT_ROOT:-../../artifacts/decentralized_ris}
run=${G2_RUN:-$artifact_root/e06_graph_energy_training/g2_long_training/g2/iter150000}
checkpoint=${G2_CHECKPOINT:-iter150000.pt}
out=${OUT_DIR:-$artifact_root/e13_fixed_ris_beamforming}

if [ ! -f "$run/summary.json" ]; then
    echo "[error] no summary.json under $run" >&2
    exit 1
fi

mkdir -p "$out"
PYTHONUNBUFFERED=1 "$ris_python" -m experiments.fixed_ris_beamforming \
    --run "$run" --checkpoint "$checkpoint" \
    --samples "$samples" --eval_seed "$eval_seed" \
    --replication_seed "$replication_seed" \
    --device "$device" --out_dir "$out" \
    2>&1 | tee "$out/e13_fixed_ris_beamforming.log"
