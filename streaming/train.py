"""Train the causal dual-head model (XLM-R or mBERT) on the shared Phase 1 data.

  python streaming/train.py --data_root DATA --model mbert --out runs/mbert
  python streaming/train.py --data_root DATA --model mbert --out runs/smoke --limit_per_pair 50 --epochs 1

Hyper-parameters and model selection MATCH the teammate's XLM-R baseline
(XLMR_Update1_Update2 package, results/baseline/training_config.json), so the
only difference between the two runs is the backbone:
  micro-batch 4 x gradient accumulation 4 (effective 16), AdamW lr 2e-5 on all
  parameters, weight decay 0.01, linear warm-up over 3% of optimizer steps then
  linear decay to 0, grad-clip 1.0, fp16 autocast (GradScaler init 128),
  lambda_sw = lambda_dur = 1, seed 6130, random window order per epoch
  (randperm seeded with seed + epoch), switch threshold fixed at 0.5.
  Validation every 10,000 optimizer steps and at each epoch end; the checkpoint
  with the best mean of (val switch F1, val duration macro-F1) is kept.

Writes to --out:
  train_log.csv      every 25 optimizer steps: mean total / switch / duration loss
                     over the micro-batches since the last row (multitask loss curves)
  val_log.csv        validation metrics + validation losses at every evaluation
  val_pairs.jsonl    per-pair validation switch F1 at every evaluation
  best.pt            weights with the best selection score
  meta.json          run config, best step, threshold (0.5)
Model selection uses ONLY the validation split. Test predictions are made
separately with predict.py, once, at the end.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig

from data import load_windows, load_gold, make_collate, length_bucketed_batches
from model import MODELS, CausalDualHead, multitask_loss

THRESHOLD = 0.5


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def run_inference(model, rows, pad_id, device, batch_size=128, amp=False):
    """Per scored position: window index, sub-word index, p(switch), duration argmax,
    plus summed validation CE for both heads (for the validation loss curve)."""
    model.eval()
    collate = make_collate(rows, pad_id)
    out = {"win": [], "pos": [], "sw_prob": [], "dur_pred": []}
    ce = {"sw": 0.0, "n_sw": 0, "dur": 0.0, "n_dur": 0}
    for b in length_bucketed_batches(rows, batch_size, shuffle=False):
        batch = collate(b)
        ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            sw_logits, dur_logits = model(ids, mask)
        swl, durl = batch["sw_labels"].to(device), batch["dur_labels"].to(device)
        for key, logits, labels in (("sw", sw_logits, swl), ("dur", dur_logits, durl)):
            keep = labels >= 0
            if keep.any():
                ce[key] += float(torch.nn.functional.cross_entropy(
                    logits[keep].float(), labels[keep], reduction="sum"))
                ce["n_" + key] += int(keep.sum())
        p = torch.softmax(sw_logits.float(), -1)[..., 1].cpu()
        d = dur_logits.argmax(-1).cpu()
        scored = batch["sw_labels"] != -1
        w_i, s_i = scored.nonzero(as_tuple=True)
        out["win"].append(batch["idx"][w_i])
        out["pos"].append(s_i)
        out["sw_prob"].append(p[w_i, s_i])
        out["dur_pred"].append(d[w_i, s_i])
    res = {k: torch.cat(v).numpy() for k, v in out.items()}
    res["loss_switch"] = ce["sw"] / max(ce["n_sw"], 1)
    res["loss_duration"] = ce["dur"] / max(ce["n_dur"], 1)
    return res


def prediction_rows(rows, gold, inf, threshold):
    """One row per word in cs_eval.py format (mirrors Phase 1 load_data.prediction_rows)."""
    for w, j, p, d in zip(inf["win"], inf["pos"], inf["sw_prob"], inf["dur_pred"]):
        win = rows[w]
        g = gold[win["id"]]
        word = win["word_ids"][j]
        assert win["sw_labels"][j] == g["sw"][word]
        yield dict(id=g["id"], word_idx=word, pair=g["pair"], cs_type=g["cs_type"],
                   sw_true=g["sw"][word], dur_true=g["dur"][word], sw_prob=float(p),
                   sw_pred=int(p >= threshold), dur_pred=int(d))


def f1_binary(y, pred):
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def macro_f1(y, pred, k):
    return float(np.mean([f1_binary((y == c).astype(int), pred == c) for c in range(k)]))


def val_metrics(rows, inf):
    y = np.array([rows[w]["sw_labels"][j] for w, j in zip(inf["win"], inf["pos"])])
    dur_true = np.array([rows[w]["dur_labels"][j] for w, j in zip(inf["win"], inf["pos"])])
    pairs = np.array([rows[w]["pair"] for w in inf["win"]])
    pred = inf["sw_prob"] >= THRESHOLD
    sw = y == 1
    sw_f1 = f1_binary(y, pred)
    dur_mf1 = macro_f1(dur_true[sw], inf["dur_pred"][sw], 3) if sw.any() else float("nan")
    by_pair = {p: f1_binary(y[pairs == p], pred[pairs == p]) for p in sorted(set(pairs))}
    vals = np.array(list(by_pair.values()))
    m = dict(switch_f1=sw_f1,
             duration_acc=float(np.mean(inf["dur_pred"][sw] == dur_true[sw])) if sw.any() else float("nan"),
             duration_macro_f1=dur_mf1,
             selection_score=(sw_f1 + dur_mf1) / 2,
             pair_mean_f1=float(vals.mean()), pair_std_f1=float(vals.std(ddof=0)),
             val_loss_switch=inf["loss_switch"], val_loss_duration=inf["loss_duration"],
             n_positions=int(len(y)), switch_rate=float(sw.mean()))
    return m, by_pair


def lr_factor(step, total, warmup):
    """Same schedule as the XLM-R baseline: linear warm-up, then linear decay to 0."""
    if step < warmup:
        return (step + 1) / warmup
    return max(0.0, (total - step) / max(1, total - warmup))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=4, help="micro-batch (windows)")
    ap.add_argument("--accumulation", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--lambda_sw", type=float, default=1.0)
    ap.add_argument("--lambda_dur", type=float, default=1.0)
    ap.add_argument("--eval_every", type=int, default=10000, help="optimizer steps (0 = epoch end only)")
    ap.add_argument("--log_every", type=int, default=25, help="optimizer steps")
    ap.add_argument("--seed", type=int, default=6130)
    ap.add_argument("--limit_per_pair", type=int, default=0, help="debug: windows per pair (0 = all)")
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop after N optimizer steps")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = get_device()
    amp = device.type == "cuda"
    name = MODELS[args.model]
    print(f"device={device} amp={amp} model={name}", flush=True)

    train_rows = load_windows(args.data_root, args.model, "train", limit_per_pair=args.limit_per_pair, seed=args.seed)
    val_rows = load_windows(args.data_root, args.model, "val", limit_per_pair=args.limit_per_pair, seed=args.seed)
    print(f"train windows={len(train_rows)} val windows={len(val_rows)}", flush=True)

    pad_id = AutoConfig.from_pretrained(name).pad_token_id
    model = CausalDualHead(name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, enabled=amp)

    micro_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    steps_per_epoch = math.ceil(micro_per_epoch / args.accumulation)
    total = steps_per_epoch * args.epochs
    if args.max_steps:
        total = min(total, args.max_steps)
    warmup = max(1, int(total * args.warmup))
    print(f"optimizer steps: {steps_per_epoch}/epoch, {total} total, warm-up {warmup}", flush=True)

    collate = make_collate(train_rows, pad_id)
    meta = dict(vars(args), model_name=name, pad_id=pad_id, device=str(device), threshold=THRESHOLD,
                selection="mean of validation switch F1 (p>=0.5) and duration macro-F1",
                steps_per_epoch=steps_per_epoch, total_steps=total)
    tlog = open(out / "train_log.csv", "w", newline="")
    tw = csv.writer(tlog); tw.writerow(["step", "epoch", "loss", "loss_switch", "loss_duration", "lr", "elapsed_s"])
    vlog_path, plog_path = out / "val_log.csv", out / "val_pairs.jsonl"
    for p in (vlog_path, plog_path):
        p.unlink(missing_ok=True)
    best, step, t0 = -1.0, 0, time.time()
    acc = {"loss": 0.0, "sw": 0.0, "dur": 0.0, "n": 0}

    def evaluate(epoch):
        nonlocal best
        inf = run_inference(model, val_rows, pad_id, device, amp=amp)
        m, by_pair = val_metrics(val_rows, inf)
        m = dict(epoch=epoch, step=step, **m)
        print("VAL", json.dumps(m), flush=True)
        new = not vlog_path.exists()
        with open(vlog_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(m)); new and w.writeheader(); w.writerow(m)
        with open(plog_path, "a") as f:
            f.write(json.dumps(dict(epoch=epoch, step=step, f1_by_pair=by_pair)) + "\n")
        if m["selection_score"] > best:
            best = m["selection_score"]
            torch.save(model.state_dict(), out / "best.pt")
            meta.update(best_epoch=epoch, best_step=step, best_val=m)
            json.dump(meta, open(out / "meta.json", "w"), indent=2)
        model.train()

    done = False
    for epoch in range(1, args.epochs + 1):
        model.train()
        g = torch.Generator().manual_seed(args.seed + epoch - 1)
        order = torch.randperm(len(train_rows), generator=g).tolist()
        micro = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(micro):
            batch = collate(b)
            ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            swl, durl = batch["sw_labels"].to(device), batch["dur_labels"].to(device)
            group_start = (i // args.accumulation) * args.accumulation
            divisor = min(args.accumulation, len(micro) - group_start)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                sw_logits, dur_logits = model(ids, mask)
            loss, l_sw, l_dur = multitask_loss(sw_logits, dur_logits, swl, durl,
                                               args.lambda_sw, args.lambda_dur)
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}")
            scaler.scale(loss / divisor).backward()
            acc["loss"] += loss.item(); acc["sw"] += l_sw.item(); acc["dur"] += l_dur.item(); acc["n"] += 1
            if (i + 1) % args.accumulation != 0 and i + 1 != len(micro):
                continue
            for grp in opt.param_groups:
                grp["lr"] = args.lr * lr_factor(step, total, warmup)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            step += 1
            if step % args.log_every == 0:
                n = acc["n"]
                tw.writerow([step, epoch, acc["loss"] / n, acc["sw"] / n, acc["dur"] / n,
                             opt.param_groups[0]["lr"], round(time.time() - t0, 1)])
                tlog.flush()
                if step % (args.log_every * 20) == 0:
                    eta = (time.time() - t0) / step * (total - step)
                    print(f"ep{epoch} step {step}/{total} loss {acc['loss']/n:.4f} "
                          f"(sw {acc['sw']/n:.4f} dur {acc['dur']/n:.4f}) "
                          f"{time.time()-t0:.0f}s, ETA {eta/60:.0f} min", flush=True)
                acc = {"loss": 0.0, "sw": 0.0, "dur": 0.0, "n": 0}
            if step >= total:
                done = True
                break
            if args.eval_every and step % args.eval_every == 0 and i + 1 < len(micro):
                evaluate(epoch)
        evaluate(epoch)
        if done:
            break
    tlog.close()
    print(f"done. best selection score {best:.4f} at epoch {meta.get('best_epoch')} step {meta.get('best_step')}")


if __name__ == "__main__":
    main()
