"""Paired bootstrap CIs for the XLM-R vs mBERT comparison (test split).

  python streaming/bootstrap_ci.py \
      --xlmr_preds results/phase2/xlmr_baseline/test_predictions.csv.gz \
      --mbert_preds results/phase2/mbert/preds_test.csv.gz

Resamples SOURCE RECORDS (id prefix before ':s<k>') with replacement, keeps the two
models paired on the same words, and reports 95% intervals for:
  pooled switch F1 difference, 12-pair mean F1 difference, sigma difference
  (population std of per-pair F1), and each model's sigma.
Covers test-set sampling only, not training randomness (one seed per model).
"""
import argparse

import numpy as np
import pandas as pd

KEYS = ["xtp", "xfp", "xfn", "mtp", "mfp", "mfn"]


def f1(tp, fp, fn):
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def stats(S):
    """S: pairs x 6 counts -> (dF1 pooled, dSigma, dPairMean, sigma_x, sigma_m)."""
    fx, fm = f1(S[:, 0], S[:, 1], S[:, 2]), f1(S[:, 3], S[:, 4], S[:, 5])
    t = S.sum(0)
    return (f1(t[3], t[4], t[5]) - f1(t[0], t[1], t[2]), fm.std() - fx.std(),
            fm.mean() - fx.mean(), fx.std(), fm.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlmr_preds", required=True)
    ap.add_argument("--mbert_preds", required=True)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    x, m = pd.read_csv(args.xlmr_preds), pd.read_csv(args.mbert_preds)
    d = x.merge(m, on=["id", "word_idx"], suffixes=("_x", "_m"))
    assert len(d) == len(x) == len(m), "the two prediction files must cover the same positions"
    d["rec"] = d.id.str.rsplit(":", n=1).str[0]
    y = d.sw_true_x
    for s, p in (("x", d.sw_pred_x), ("m", d.sw_pred_m)):
        d[s + "tp"] = ((p == 1) & (y == 1)).astype(int)
        d[s + "fp"] = ((p == 1) & (y == 0)).astype(int)
        d[s + "fn"] = ((p == 0) & (y == 1)).astype(int)
    g = d.groupby(["rec", "pair_x"])[KEYS].sum().reset_index()
    recs, pairs = g.rec.unique(), sorted(g.pair_x.unique())
    ri = g.rec.map({r: i for i, r in enumerate(recs)}).values
    pi = g.pair_x.map({p: i for i, p in enumerate(pairs)}).values
    A = np.zeros((len(recs), len(pairs), len(KEYS)))
    for c, k in enumerate(KEYS):
        A[ri, pi, c] = g[k].values

    names = ["pooled F1 (mBERT - XLM-R)", "sigma (mBERT - XLM-R)", "pair-mean F1 (mBERT - XLM-R)",
             "sigma XLM-R", "sigma mBERT"]
    obs = stats(A.sum(0))
    rng = np.random.default_rng(args.seed)
    draws = np.array([stats(np.tensordot(np.bincount(rng.integers(0, len(recs), len(recs)),
                                                     minlength=len(recs)), A, 1))
                      for _ in range(args.draws)])
    print(f"{len(recs)} test records, {len(d)} positions, {args.draws} paired draws")
    for i, n in enumerate(names):
        lo, hi = np.percentile(draws[:, i], [2.5, 97.5])
        print(f"{n:30s} {obs[i]*100:+7.2f} pp   95% CI [{lo*100:+.2f}, {hi*100:+.2f}]")


if __name__ == "__main__":
    main()
