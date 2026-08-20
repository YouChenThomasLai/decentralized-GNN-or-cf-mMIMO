#!/bin/bash

set -euo pipefail

phase=${1:-smoke}
python_bin=${PYTHON_BIN:-python}
export PYTHON_BIN="${python_bin}"

(
    cd stage1
    ./run_exp-v2.sh "${phase}"
)
(
    cd stage0
    ./run_exp-v2.sh "${phase}"
)

if [[ "${phase}" == "smoke" ]]; then
    results_root="results_snapshot_scaling/smoke"
else
    results_root="results_snapshot_scaling"
fi
"${python_bin}" plot-v2.py --results-root "${results_root}"
