#!/usr/bin/env bash
# E12: the mandatory centralized/distributed model-based joint-optimization pair.
#
# Benchmark item 5 of doc/research_positioning.md.  Nothing is trained here: both
# arms are model-based solvers run on the shared 400-sample holdout, so the run
# is evaluation only.  The centralized arm optimizes from full CSI at the CPU;
# the distributed arm is Huang et al.'s incremental consensus ADMM over AP-local
# copies of the multi-RIS reflection vector.
#
# This corrected protocol uses the stronger declared `matched` initialization,
# calibrates the penalty on 400 samples disjoint from every evaluation seed, and
# stores its outputs separately from the original failed-gate run.
set -euo pipefail

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

ris_python=${RIS_PYTHON:-python}
device=${DEVICE:-cuda:0}
samples=${SAMPLES:-400}
eval_seed=${EVAL_SEED:-20260915}
replication_seed=${REPLICATION_SEED:-20260918}
calibration_seed=${CALIBRATION_SEED:-20260912}
calibration_samples=${CALIBRATION_SAMPLES:-400}
max_iterations=${MAX_ITERATIONS:-2000}
max_sweeps=${MAX_SWEEPS:-200}
mm_iterations=${MM_ITERATIONS:-200}
dual_period=${DUAL_PERIOD:-sweep}
artifact_root=${ARTIFACT_ROOT:-../../artifacts/decentralized_ris}
run=${CONFIG_RUN:-$artifact_root/e06_graph_energy_training/g2_long_training/g2/iter150000}
out=${OUT_DIR:-$artifact_root/e12_model_based_optimization/corrected_protocol}

if [ ! -f "$run/summary.json" ]; then
    echo "[error] no summary.json under $run" >&2
    exit 1
fi

mkdir -p "$out"

if [ -f "$out/calibration.json" ]; then
    echo "[skip] calibration already recorded in $out/calibration.json"
else
    PYTHONUNBUFFERED=1 "$ris_python" -m experiments.model_based_joint \
        --run "$run" --calibrate \
        --calibration_seed "$calibration_seed" \
        --calibration_samples "$calibration_samples" \
        --max_iterations "$max_iterations" --max_sweeps "$max_sweeps" \
        --mm_iterations "$mm_iterations" --init matched \
        --device "$device" --out_dir "$out" \
        2>&1 | tee "$out/e12_calibration.log"
fi

# The holdout uses the penalty the calibration pass selected, not a CLI default,
# so the frozen value cannot silently disagree with the recorded selection.
selected=$("$ris_python" - "$out/calibration.json" "$calibration_seed" \
    "$calibration_samples" "$dual_period" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    selected = json.load(handle)["selected"]
expected = {
    "calibration_seed": int(sys.argv[2]),
    "calibration_samples": int(sys.argv[3]),
    "dual_period": sys.argv[4],
    "init": "matched",
}
for key, value in expected.items():
    if selected.get(key) != value:
        raise SystemExit(
            f"stored calibration {key}={selected.get(key)!r}, expected {value!r}"
        )
scale = selected["rho_scale"]
if scale is None:
    raise SystemExit("calibration selected no feasible penalty scale")
print(scale)
PY
)
rho_scale=$selected
echo "[e12] penalty scale $rho_scale, multiplier schedule $dual_period"

PYTHONUNBUFFERED=1 "$ris_python" -m experiments.model_based_joint \
    --run "$run" --samples "$samples" \
    --eval_seed "$eval_seed" --replication_seed "$replication_seed" \
    --max_iterations "$max_iterations" --max_sweeps "$max_sweeps" \
    --mm_iterations "$mm_iterations" \
    --rho_scale "$rho_scale" --dual_period "$dual_period" \
    --init matched \
    --device "$device" --out_dir "$out" \
    2>&1 | tee "$out/e12_model_based_joint.log"
