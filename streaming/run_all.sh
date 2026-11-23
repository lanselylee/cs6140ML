#!/usr/bin/env bash
# Train one backbone with the XLM-R baseline's hyper-parameters (train.py defaults),
# then write val + test predictions and score them with cs_eval.py.
#   bash streaming/run_all.sh DATA_ROOT OUT_DIR [MODEL=mbert] [extra train.py args...]
set -euo pipefail
DATA=$1; OUT=$2; M=${3:-mbert}; shift $(( $# >= 3 ? 3 : 2 ))
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT"

python "$HERE/tests/test_causal.py"

python "$HERE/train.py" --data_root "$DATA" --model "$M" --out "$OUT/$M" "$@" 2>&1 | tee "$OUT/$M.log"

for S in val test; do
  python "$HERE/predict.py" --data_root "$DATA" --run "$OUT/$M" --split $S 2>&1 | tee "$OUT/${M}_${S}_eval.txt"
done
