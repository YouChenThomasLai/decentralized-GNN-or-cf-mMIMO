#!/bin/bash
# Discrete RIS coordinate-descent baseline over the decentralized RIS checkpoints.
# Headline cell is M=2, Pmax=15 dBm; the sweeps use fewer samples per point.
set -u

PY=${PY:-python}
DEVICE=${DEVICE:-cuda:0}
ROUNDS=${ROUNDS:-4}
M_ROOT=results_batch_8_BS-radius_200_RIS-radius_100_vary_M
P_ROOT=results_batch_8_BS-radius_200_RIS-radius_100_vary_Pmax
OUT=${OUT:-results_discrete_cd}

run_cell () {
    local ckpt=$1 out=$2 samples=$3 m=$4 pmax=$5
    if [ ! -f "$ckpt" ]; then
        echo "[skip] missing $ckpt"
        return
    fi
    echo "=== $out (M=$m, Pmax=$pmax, samples=$samples)"
    $PY discrete_cd_baseline.py \
        --ckpt "$ckpt" --M "$m" --pmax_dbm "$pmax" \
        --samples "$samples" --rounds "$ROUNDS" \
        --device "$DEVICE" --out_dir "$out"
}

run_cell "$M_ROOT/M2_N30_L4_K8_P15.0/run0/models/model_final_run0.pt" \
         "$OUT/headline_M2_P15" 800 2 15

for p in 5 10 15 20 25 30 35; do
    run_cell "$P_ROOT/M2_N30_L4_K8_P${p}.0/run0/models/model_final_run0.pt" \
             "$OUT/vary_Pmax/P${p}" 160 2 "$p"
done

for m in 1 2 3 4 5; do
    run_cell "$M_ROOT/M${m}_N30_L4_K8_P15.0/run0/models/model_final_run0.pt" \
             "$OUT/vary_M/M${m}" 160 "$m" 15
done

echo "[done] discrete coordinate-descent sweep"
