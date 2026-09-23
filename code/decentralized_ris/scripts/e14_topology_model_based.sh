#!/usr/bin/env bash
# E14: frozen E12 fixed-budget solvers on the declared T1 and T2 layouts.
set -euo pipefail

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

ris_python=${RIS_PYTHON:-python}
device=${DEVICE:-cuda:0}
artifact_root=${ARTIFACT_ROOT:-../../artifacts/decentralized_ris}
run=${CONFIG_RUN:-$artifact_root/e06_graph_energy_training/g2_long_training/g2/iter150000}

for topology in t1 t2; do
    out=$artifact_root/e14_topology_stress/${topology}_solver
    mkdir -p "$out"
    if [ -f "$out/summary.json" ]; then
        echo "[skip] $out/summary.json exists"
        continue
    fi
    echo "[e14] solving $topology"
    PYTHONUNBUFFERED=1 "$ris_python" -m experiments.model_based_joint \
        --run "$run" --layout "experiments/topology_${topology}.json" \
        --samples 400 --eval_seed 20260920 --replication_seed 0 \
        --max_iterations 2000 --max_sweeps 200 --mm_iterations 200 \
        --rho_scale 0.8 --dual_period sweep --init matched \
        --precision double --device "$device" --out_dir "$out" \
        > "$out/run.log" 2>&1
    echo "[done] $out/summary.json"
done
