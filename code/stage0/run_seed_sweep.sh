#!/bin/bash
# Six-seed sweep at a budget long enough to reach (or approach) convergence.
# Seeds run three-to-a-GPU on the two idle devices; the benchmark showed only a
# 14% per-process slowdown at that concurrency, so wall clock is set by one seed.
set -u
cd /tmp2/b12902052/ThomasLai/code/stage0
PY=/tmp2/b12902052/miniforge3/envs/decentralized-inference/bin/python
OUT=results_seed_sweep
ITER=${ITER:-60000}

for s in 0 1 2; do
    $PY train_fast.py --seed "$s" --n_iter "$ITER" --device cuda:0 \
        --log_eval_interval 2000 --save_every 5000 \
        --test_sample_val 400 --test_sample_final 3200 \
        --out_dir "$OUT" > "$OUT/seed${s}.log" 2>&1 &
done
for s in 3 4 5; do
    $PY train_fast.py --seed "$s" --n_iter "$ITER" --device cuda:3 \
        --log_eval_interval 2000 --save_every 5000 \
        --test_sample_val 400 --test_sample_final 3200 \
        --out_dir "$OUT" > "$OUT/seed${s}.log" 2>&1 &
done
wait
echo "[done] six-seed sweep finished at $(date)"
