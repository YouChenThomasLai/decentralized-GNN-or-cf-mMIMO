#!/bin/bash

set -euo pipefail

phase=${1:-implementation}
python_bin=${PYTHON_BIN:-python}
export PYTHON_BIN="${python_bin}"

if [[ "${phase}" == "plot" ]]; then
    "${python_bin}" plot-v2.py \
        --results-root "${RESULTS_ROOT:-../results_snapshot_scaling_v2_bpp}"
    exit
fi

(
    cd stage0
    ./run_exp-v2.sh "${phase}"
)
