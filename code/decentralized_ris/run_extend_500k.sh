#!/bin/bash
# Continue decentralized RIS training from the 150k checkpoint to 500,000 iterations on
# lab301-5090-tailscale (one RTX 5090).
#
# Why this host and this length: the 5090 sustains 397 iterations/min under the
# production validation cadence, 4.1-4.9x the 81-97 it/min measured on meow2's
# RTX 4090, and it enforces no GPU quota. The quota was what capped every
# previous continuation at 50-60k iterations. +350,000 iterations therefore
# costs about 14.7 hours here, and gives 175 validation points -- enough slope
# leverage (se ~ 0.006 bps/Hz per 10k) to tell a plateau apart from the slow
# positive slope that doc/decentralized_ris_evidence.md could
# only just resolve at 150k.
#
# nvidia-smi is unusable on this host (NVML 595.91 against kernel module
# 595.84), so unlike run_extend.sh and run_chain_150k.sh this script does no
# device picking and no quota accounting: the host exposes a single GPU and
# CUDA itself works.
set -u
cd ~/ThomasLai/code/decentralized_ris
PY=~/miniforge3/envs/decentralized-inference/bin/python

RESUME=resume_150k.pt
N_ITER=350000
OUT=results_extend_500k
RUN=$OUT/M2_N30_L4_K8_P15.0_iter${N_ITER}_seed0/run0
SNAP_EVERY=50000

log() { echo "$(date +%F' '%T) $*"; }

if [ ! -f "$RESUME" ]; then
    log "missing $RESUME; aborting"
    exit 1
fi
if [ -e "$OUT" ]; then
    log "$OUT already exists; refusing to overwrite it"
    exit 1
fi

# Sidecar snapshotter. train_fast.py only keeps a single rolling
# resumable_latest.pt, so the intermediate checkpoints needed to evaluate the
# trajectory (rather than just the endpoint) would otherwise be overwritten.
# Copying from the outside keeps train_fast.py byte-identical to the version
# whose SHA-256 the report records.
snapshot_loop() {
    local next=$((150000 + SNAP_EVERY))
    local latest=$RUN/models/resumable_latest.pt
    while :; do
        sleep 120
        [ -f "$latest" ] || continue
        cp "$latest" "$RUN/models/.snap_tmp.pt" 2>/dev/null || continue
        it=$($PY - "$RUN/models/.snap_tmp.pt" <<'PYEOF' 2>/dev/null
import sys, torch
try:
    print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False)["iteration"]))
except Exception:
    print(-1)
PYEOF
)
        [ -n "${it:-}" ] && [ "$it" -ge "$next" ] 2>/dev/null || { rm -f "$RUN/models/.snap_tmp.pt"; continue; }
        mv "$RUN/models/.snap_tmp.pt" "$RUN/models/resume_$((it/1000))k.pt"
        log "[snapshot] wrote resume_$((it/1000))k.pt"
        while [ "$next" -le "$it" ]; do next=$((next + SNAP_EVERY)); done
    done
}

mkdir -p "$RUN/models"
snapshot_loop &
SNAP_PID=$!
trap 'kill $SNAP_PID 2>/dev/null' EXIT

log "starting: +${N_ITER} iterations from $RESUME (150000) -> 500000 total"
$PY -u train_fast.py \
    --resume "$RESUME" --n_iter "$N_ITER" --seed 0 \
    --log_eval_interval 2000 --save_every 5000 \
    --test_sample_val 400 --test_sample_final 3200 \
    --device cuda:0 --out_dir "$OUT"
rc=$?

kill $SNAP_PID 2>/dev/null
if [ $rc -ne 0 ]; then
    log "[fail] train_fast.py exited $rc; resume from $RUN/models/resumable_latest.pt"
    exit $rc
fi
log "[done] 500000 iterations in total"
