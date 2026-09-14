#!/bin/bash
# Push the single 40k-iteration run to 100k to locate the training plateau.
#
# meow2 enforces a 24 GPU-hour daily quota and multiplies the deduction rate by
# (N+0.2)*N when a user holds N GPUs while fewer than N sit idle. The previous
# six-seed attempt held two GPUs and was killed once that penalty exhausted the
# quota, so this run pins exactly one device.
set -u
cd /tmp2/b12902052/ThomasLai/code/decentralized_ris
PY=/tmp2/b12902052/miniforge3/envs/decentralized-inference/bin/python

# The quota resets at 00:00 UTC+8; wait past it before touching a GPU.
while [ "$(date +%H%M)" -ge 2330 ] || [ "$(date +%H%M)" -lt 3 ]; do
    echo "$(date +%T) waiting for the daily GPU quota reset"
    sleep 120
done

remaining=$(ws-status 2>/dev/null | grep -oP 'GPU quota remaining: \K-?[0-9]+' | head -1)
echo "$(date +%T) quota remaining: ${remaining:-unknown} secs"
if [ -n "${remaining:-}" ] && [ "$remaining" -le 0 ]; then
    echo "quota still exhausted; aborting rather than being killed again"
    exit 1
fi

# Pick the idlest device and pin it, so PyTorch cannot grab more than one.
GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
      | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' ')
echo "$(date +%T) using GPU $GPU"

CUDA_VISIBLE_DEVICES=$GPU $PY train_fast.py \
    --resume ckpt_40k.pt --n_iter 60000 --seed 0 \
    --log_eval_interval 2000 --save_every 5000 \
    --test_sample_val 400 --test_sample_final 3200 \
    --device cuda:0 --out_dir results_extend_100k
echo "[done] extension finished at $(date)"
