#!/bin/bash
# Chain the finishing 100k extension into an evaluation and a further 50k of
# training, so the GPU is never idle between the two and the plateau search
# continues without a manual step.
#
# The evaluation runs first and deliberately reuses the reference settings and
# sample counts already applied to the 2k and 40k checkpoints in
# doc/decentralized_ris_evidence.md, so the three points stay comparable.
#
# meow2 enforces a 24 GPU-hour daily quota that resets at 00:00 UTC+8 and
# multiplies the deduction rate by (N+0.2)*N when a user holds N GPUs while
# fewer than N sit idle. Every stage below therefore pins exactly one device;
# the six-seed sweep that held two was killed by that penalty.
set -u
cd /tmp2/b12902052/ThomasLai/code/decentralized_ris
PY=/tmp2/b12902052/miniforge3/envs/decentralized-inference/bin/python

RUN=results_extend_100k/M2_N30_L4_K8_P15.0_iter60000_seed0/run0
FINAL=$RUN/models/resumable_final.pt
PLAIN=ckpt_100k.pt       # plain state_dict; the evaluation scripts expect one
RESUME=resume_100k.pt    # bundle carrying the corrected total iteration count
NEXT_ITER=50000
NEXT_OUT=results_extend_150k

log() { echo "$(date +%T) $*"; }

pick_gpu() {
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '
}

quota_left() {
    ws-status 2>/dev/null | grep -oP 'GPU quota remaining: \K-?[0-9]+' | head -1
}

# Stay off the GPU across the reset boundary and until the budget covers the
# stage, so a long job is never started only to be killed part way through.
wait_for_quota() {
    local need=$1 left
    while :; do
        if [ "$(date +%H%M)" -ge 2330 ] || [ "$(date +%H%M)" -lt 3 ]; then
            log "waiting for the daily GPU quota reset"
            sleep 120
            continue
        fi
        left=$(quota_left)
        if [ -z "${left:-}" ]; then
            log "quota unknown; proceeding"
            return 0
        fi
        if [ "$left" -ge "$need" ]; then
            log "quota remaining: ${left}s (this stage needs ${need}s)"
            return 0
        fi
        log "quota remaining: ${left}s, below the ${need}s this stage needs; waiting"
        sleep 300
    done
}

# ------------------------------------------------------------------ 1. wait
log "waiting for the 100k run to write $FINAL"
while [ ! -f "$FINAL" ]; do
    if ! pgrep -u "$(id -u)" -f "out_dir results_extend_100k" >/dev/null 2>&1; then
        sleep 90
        if [ ! -f "$FINAL" ]; then
            log "the 100k run exited without a final checkpoint; aborting"
            exit 1
        fi
    fi
    sleep 60
done
sleep 30   # let the final write settle before reading it
log "100k run finished"

# --------------------------------------------------------------- 2. derive
# ckpt_40k.pt carried no iteration counter, so the 100k run's own bundle reports
# 60000. Record the true total here so the next warm start reports 150000.
$PY - "$FINAL" "$PLAIN" "$RESUME" <<'PYEOF'
import sys, torch
src, plain, resume = sys.argv[1:4]
b = torch.load(src, map_location="cpu", weights_only=False)
torch.save(b["model"], plain)
torch.save({"model": b["model"], "optimizer": b["optimizer"], "iteration": 100000}, resume)
print("[derive] wrote %s and %s (bundle iteration %s recorded as 100000)"
      % (plain, resume, b.get("iteration")))
PYEOF

# ------------------------------------------------------------- 3. evaluate
wait_for_quota 3600
GPU=$(pick_gpu)
log "evaluation on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU $PY ablation_local_csi.py --ckpt "$PLAIN" \
    --samples 320 --greedy --device cuda:0 \
    --out_dir results_local_csi_ablation/trained100k_with_greedy
CUDA_VISIBLE_DEVICES=$GPU $PY discrete_cd_baseline.py --ckpt "$PLAIN" \
    --samples 800 --rounds 4 --device cuda:0 \
    --out_dir results_discrete_cd/trained100k_M2_P15
log "evaluation done"

# ------------------------------------------------------------- 4. continue
if [ -e "$NEXT_OUT" ]; then
    log "$NEXT_OUT already exists; refusing to overwrite it"
    exit 1
fi
wait_for_quota 33000
GPU=$(pick_gpu)
log "continuation training on GPU $GPU: +${NEXT_ITER} iterations into $NEXT_OUT"
CUDA_VISIBLE_DEVICES=$GPU $PY train_fast.py \
    --resume "$RESUME" --n_iter "$NEXT_ITER" --seed 0 \
    --log_eval_interval 2000 --save_every 5000 \
    --test_sample_val 400 --test_sample_final 3200 \
    --device cuda:0 --out_dir "$NEXT_OUT"
log "[done] continuation finished; 150000 iterations in total"
