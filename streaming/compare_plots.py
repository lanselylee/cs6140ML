"""Compare the two backbones (XLM-R vs mBERT) on the test split and draw the Update 2 figures.

  python streaming/compare_plots.py \
      --xlmr_preds results/phase2/xlmr_baseline/test_predictions.csv.gz \
      --xlmr_train_loss results/phase2/xlmr_baseline/baseline_training_sampled_loss.csv \
      --xlmr_val_loss results/phase2/xlmr_baseline/baseline_validation_loss.csv \
      --mbert_run results/phase2/mbert --out results/phase2/figures

Both models are scored from their per-word prediction CSVs (cs_eval.py format)
with the SAME code below, so every number is computed identically.
Writes comparison.csv, comparison.md and four PNGs:
  fig_pair_f1.png          test switch F1 per language pair (grouped bars)
  fig_overall.png          pooled test metrics (grouped bars)
  fig_duration_class.png   test duration F1 per length bin
  fig_loss_curves.png      training + validation loss, switch and duration heads
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

# Reference categorical palette, slots 1-2 (validated adjacent pair, light surface).
COLORS = {"XLM-R": "#2a78d6", "mBERT": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BINS = ["1–2 words", "3–6 words", "7+ words"]
TWO_TYPES = ["Intersentential", "Intrasentential"]


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def metrics(df):
    y, pred, prob = df.sw_true.values, df.sw_pred.values, df.sw_prob.values
    tp = int(((pred == 1) & (y == 1)).sum())
    sw = df[df.sw_true == 1]
    pair_f1 = {p: f1_score(g.sw_true, g.sw_pred, zero_division=0) for p, g in df.groupby("pair")}
    acc_type = {t: float((g.sw_true == g.sw_pred).mean()) for t, g in df.groupby("cs_type")}
    dur_f1 = f1_score(sw.dur_true, sw.dur_pred, labels=[0, 1, 2], average=None, zero_division=0)
    vals = np.array(list(pair_f1.values()))
    return dict(
        positions=len(df), true_switches=int(y.sum()),
        switch_f1=f1_score(y, pred), precision=tp / max(pred.sum(), 1), recall=tp / max(y.sum(), 1),
        auroc=roc_auc_score(y, prob), average_precision=average_precision_score(y, prob),
        duration_accuracy=float((sw.dur_true == sw.dur_pred).mean()),
        duration_macro_f1=float(dur_f1.mean()),
        type_macro_accuracy_3=float(np.mean(list(acc_type.values()))),
        type_macro_accuracy_2=float(np.mean([acc_type[t] for t in TWO_TYPES if t in acc_type])),
        pair_mean_f1=float(vals.mean()), pair_std_f1=float(vals.std(ddof=0)),
        pair_f1=pair_f1, duration_class_f1=dict(zip(BINS, dur_f1)),
    )


def grouped_bars(ax, labels, series, value_labels=False):
    x = np.arange(len(labels))
    w = 0.38
    for k, (name, vals) in enumerate(series.items()):
        off = (k - (len(series) - 1) / 2) * (w + 0.02)
        bars = ax.bar(x + off, vals, w, color=COLORS[name], label=name, edgecolor="white", linewidth=1)
        if value_labels:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v*100:.1f}", ha="center",
                        va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x, labels)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)


def smooth(s, n=25):
    return s.rolling(n, min_periods=1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlmr_preds", required=True)
    ap.add_argument("--xlmr_train_loss", required=True)
    ap.add_argument("--xlmr_val_loss", required=True)
    ap.add_argument("--mbert_run", required=True, help="dir with preds_test.csv, train_log.csv, val_log.csv")
    ap.add_argument("--mbert_preds", default=None, help="override (default: <mbert_run>/preds_test.csv)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    run = Path(args.mbert_run)

    preds = {"XLM-R": pd.read_csv(args.xlmr_preds),
             "mBERT": pd.read_csv(args.mbert_preds or run / "preds_test.csv")}
    a, b = (preds["XLM-R"][["id", "word_idx"]], preds["mBERT"][["id", "word_idx"]])
    same = len(a) == len(b) and a.merge(b, how="inner").shape[0] == len(a)
    print(f"scored positions: XLM-R {len(a)}, mBERT {len(b)}, identical set: {same}")
    m = {k: metrics(v) for k, v in preds.items()}

    # ---- tables
    rows = ["switch_f1", "precision", "recall", "auroc", "average_precision", "duration_accuracy",
            "duration_macro_f1", "type_macro_accuracy_2", "type_macro_accuracy_3", "pair_mean_f1", "pair_std_f1"]
    table = pd.DataFrame({k: [m[k][r] for r in rows] for k in m}, index=rows)
    pairs = sorted(m["XLM-R"]["pair_f1"], key=lambda p: -m["XLM-R"]["pair_f1"][p])
    per_pair = pd.DataFrame({k: [m[k]["pair_f1"].get(p, np.nan) for p in pairs] for k in m}, index=pairs)
    for t in (table, per_pair):
        t["mBERT − XLM-R (pp)"] = (t["mBERT"] - t["XLM-R"]) * 100
    pd.concat([table, per_pair.rename(index=lambda p: "F1 " + p)]).to_csv(out / "comparison.csv")
    with open(out / "comparison.md", "w") as f:
        f.write(f"Test split, {m['XLM-R']['positions']:,} scored word positions "
                f"({m['XLM-R']['true_switches']:,} switches); identical position set: {same}. "
                "Threshold 0.5. pair_std_f1 = population std over 12 pairs.\n\n")
        f.write((table.iloc[:, :2] * 100).round(2).join(table.iloc[:, 2].round(2)).to_markdown() + "\n\n")
        f.write((per_pair.iloc[:, :2] * 100).round(2).join(per_pair.iloc[:, 2].round(2)).to_markdown() + "\n")
    print((table.iloc[:, :2] * 100).round(2).join(table.iloc[:, 2].round(2)).to_string())

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                         "axes.titlecolor": INK, "axes.labelcolor": INK2})

    # ---- 1. per-pair switch F1
    fig, ax = plt.subplots(figsize=(11, 4.2))
    grouped_bars(ax, [p.replace("-English", "") for p in pairs],
                 {k: [m[k]["pair_f1"].get(p, np.nan) for p in pairs] for k in m})
    for k in m:
        ax.axhline(m[k]["pair_mean_f1"], color=COLORS[k], linewidth=1, linestyle="--", alpha=0.8)
    ax.set_ylim(0, 1); ax.set_ylabel("Switch F1 (test)")
    ax.set_title("Switch F1 by language pair (X–English); dashed = mean over 12 pairs", loc="left")
    style(ax); fig.tight_layout(); fig.savefig(out / "fig_pair_f1.png", dpi=200); plt.close(fig)

    # ---- 2. pooled metrics
    keys = [("switch_f1", "Switch F1"), ("precision", "Precision"), ("recall", "Recall"),
            ("average_precision", "AP"), ("duration_accuracy", "Duration acc."),
            ("duration_macro_f1", "Duration macro-F1")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    grouped_bars(ax, [l for _, l in keys], {k: [m[k][r] for r, _ in keys] for k in m}, value_labels=True)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score (test)")
    ax.set_title("Pooled test metrics, causal dual-head, same data and hyper-parameters", loc="left")
    style(ax); fig.tight_layout(); fig.savefig(out / "fig_overall.png", dpi=200); plt.close(fig)

    # ---- 3. duration F1 per bin
    fig, ax = plt.subplots(figsize=(6, 4))
    grouped_bars(ax, BINS, {k: [m[k]["duration_class_f1"][b_] for b_ in BINS] for k in m}, value_labels=True)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1 (test, true switches)")
    ax.set_title("Duration head: F1 per length bin", loc="left")
    style(ax); fig.tight_layout(); fig.savefig(out / "fig_duration_class.png", dpi=200); plt.close(fig)

    # ---- 4. loss curves
    xt = pd.read_csv(args.xlmr_train_loss, encoding="utf-8-sig")
    xv = pd.read_csv(args.xlmr_val_loss, encoding="utf-8-sig")
    mt = pd.read_csv(run / "train_log.csv")
    mv = pd.read_csv(run / "val_log.csv")
    train = {"XLM-R": (xt.step, xt.switch_loss, xt.duration_loss),
             "mBERT": (mt.step, mt.loss_switch, mt.loss_duration)}
    if "step" not in xv.columns:
        raise SystemExit("XLM-R validation loss file needs a step column")
    val = {"XLM-R": (xv.step, xv.switch_loss, xv.duration_loss),
           "mBERT": (mv.step, mv.val_loss_switch, mv.val_loss_duration)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, j, title in ((axes[0], 1, "Switch head loss"), (axes[1], 2, "Duration head loss")):
        for k in ("XLM-R", "mBERT"):
            ax.plot(train[k][0], smooth(train[k][j]), color=COLORS[k], linewidth=1.5, alpha=0.55,
                    label=f"{k} train")
            ax.plot(val[k][0], val[k][j], color=COLORS[k], linewidth=2, marker="o", markersize=5,
                    markeredgecolor="white", label=f"{k} validation")
        ax.set_title(title, loc="left"); ax.set_xlabel("optimizer step"); ax.set_ylabel("cross-entropy")
        style(ax)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    fig.text(0.01, 0.005, "Train curves smoothed over 25 log points (XLM-R: sampled last micro-batch; "
             "mBERT: mean over each 25-step window). Validation = full-set average.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out / "fig_loss_curves.png", dpi=200); plt.close(fig)
    print("wrote", ", ".join(sorted(p.name for p in out.iterdir())))


if __name__ == "__main__":
    main()
