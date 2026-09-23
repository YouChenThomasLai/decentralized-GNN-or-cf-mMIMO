#!/usr/bin/env bash
# E07 continuation: matched-bit-budget comparison of the G2 executable-phase
# message against the R0 latent message.  Both checkpoints are frozen and
# scored on identical channel draws; the codec sits on the AP->CPU wire.
#
# Rules are declared in
# artifacts/decentralized_ris/e07_message_codec/prereg_matched_budget.json
# before this script is run.  A failed gate is reported, never retuned.
set -euo pipefail
cd "$(dirname -- "$0")/.."
OUT=${1:-../../artifacts/decentralized_ris/e07_message_codec}
PYTHONPATH=. /home/youchen/miniconda3/envs/decentralized-inference/bin/python \
    -m experiments.matched_budget_codec \
    --calib_samples 800 --calib_seed 20260916 \
    --eval_samples 400 --eval_seed 20260915 --confirm_seed 20260923 \
    --batch_size 8 --device cuda:0 \
    --out_dir "$OUT"
