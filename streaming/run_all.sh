#!/usr/bin/env bash
# Train XLM-R and mBERT in parallel (one GPU each), then write val + test predictions
# and score them with cs_eval.py. Same hyper-parameters for both models.
#   bash streaming/run_all.sh DATA_ROOT OUT_DIR [extra train.py args...]
set -euo pipefail
DATA=$1; OUT=$2; shift 2
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

python "$HERE/tests/test_causal.py"

CUDA_VISIBLE_DEVICES=0 python "$HERE/train.py" --data_root "$DATA" --model xlmr  --out "$OUT/xlmr"  "$@" > "$OUT/xlmr.log"  2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 python "$HERE/train.py" --data_root "$DATA" --model mbert --out "$OUT/mbert" "$@" > "$OUT/mbert.log" 2>&1 &
P2=$!
echo "training xlmr (pid $P1) and mbert (pid $P2); follow with: tail -f $OUT/*.log"
wait $P1; wait $P2

for M in xlmr mbert; do
  for S in val test; do
    CUDA_VISIBLE_DEVICES=0 python "$HERE/predict.py" --data_root "$DATA" --run "$OUT/$M" --split $S > "$OUT/${M}_${S}_eval.txt" 2>&1
  done
done
grep -h -A30 "Model:" "$OUT"/*_test_eval.txt
