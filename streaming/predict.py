"""Write a cs_eval.py prediction CSV for a trained run and score it.

  python streaming/predict.py --data_root DATA --run runs/xlmr --split val
  python streaming/predict.py --data_root DATA --run runs/xlmr --split test   # final, run ONCE

Uses the threshold chosen on validation (meta.json). Scoring uses the group's
unmodified cs_eval.py (DATA/code/vendor/cs_eval.py). Besides cs_eval's
macro accuracy (all three cs_type values), it also reports the 2-type macro
(Intersentential + Intrasentential only), which is what spec V-A.3 names.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import load_windows, load_gold
from model import CausalDualHead
from train import get_device, run_inference, prediction_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--limit_per_pair", type=int, default=0)
    args = ap.parse_args()

    run = Path(args.run)
    meta = json.load(open(run / "meta.json"))
    device = get_device()
    model = CausalDualHead(meta["model_name"]).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))

    rows = load_windows(args.data_root, meta["model"], args.split, limit_per_pair=args.limit_per_pair)
    gold = load_gold(args.data_root, args.split)
    inf = run_inference(model, rows, meta["pad_id"], device, amp=device.type == "cuda")
    df = pd.DataFrame(prediction_rows(rows, gold, inf, meta["threshold"]))
    csv_path = run / f"preds_{args.split}.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {len(df)} word rows -> {csv_path} (threshold {meta['threshold']})")

    sys.path.insert(0, str(Path(args.data_root) / "code" / "vendor"))
    from cs_eval import evaluate, load_predictions
    res = evaluate(str(csv_path), model_name=meta["model_name"], out_dir=str(run / f"eval_{args.split}"))
    scored = load_predictions(str(csv_path))
    two = scored[scored["cs_type"].isin(["Intersentential", "Intrasentential"])]
    res["macro_accuracy_inter_intra_only"] = float(np.mean(
        [(g["sw_true"] == g["sw_pred"]).mean() for _, g in two.groupby("cs_type")]))
    print(f"2-type macro accuracy (inter+intra only): {res['macro_accuracy_inter_intra_only']:.4f}")
    json.dump(res, open(run / f"eval_{args.split}" / "summary.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
