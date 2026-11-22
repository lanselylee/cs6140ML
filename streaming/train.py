"""Train the causal dual-head model (XLM-R or mBERT) on the shared Phase 1 data.

  python streaming/train.py --data_root DATA --model xlmr --out runs/xlmr
  python streaming/train.py --data_root DATA --model xlmr --out runs/smoke --limit_per_pair 50 --epochs 1

Writes to --out:
  train_log.csv      step-level total / switch / duration loss (deliverable: multitask loss logs)
  val_log.csv        validation metrics after every epoch
  best.pt            weights of the epoch with the best validation switch F1
  meta.json          run config + the switch threshold chosen on validation
Model selection (epoch, threshold) uses ONLY the validation split.
Test predictions are made separately with predict.py, once, at the end.
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, get_linear_schedule_with_warmup

from data import load_windows, load_gold, make_collate, length_bucketed_batches
from model import MODELS, CausalDualHead, multitask_loss


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def run_inference(model, rows, pad_id, device, batch_size=128, amp=False):
    """Return per-scored-position arrays: window index, sub-word index, p(switch), duration argmax."""
    model.eval()
    collate = make_collate(rows, pad_id)
    out = {"win": [], "pos": [], "sw_prob": [], "dur_pred": []}
    for b in length_bucketed_batches(rows, batch_size, shuffle=False):
        batch = collate(b)
        ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            sw_logits, dur_logits = model(ids, mask)
        p = torch.softmax(sw_logits.float(), -1)[..., 1].cpu()
        d = dur_logits.argmax(-1).cpu()
        scored = batch["sw_labels"] != -1
        w_i, s_i = scored.nonzero(as_tuple=True)
        out["win"].append(batch["idx"][w_i])
        out["pos"].append(s_i)
        out["sw_prob"].append(p[w_i, s_i])
        out["dur_pred"].append(d[w_i, s_i])
    return {k: torch.cat(v).numpy() for k, v in out.items()}


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


def f1_at(y, p, thr):
    pred = p >= thr
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def val_metrics(rows, inf):
    y = np.array([rows[w]["sw_labels"][j] for w, j in zip(inf["win"], inf["pos"])])
    dur_true = np.array([rows[w]["dur_labels"][j] for w, j in zip(inf["win"], inf["pos"])])
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    f1s = [f1_at(y, inf["sw_prob"], t) for t in grid]
    best = int(np.argmax(f1s))
    sw = y == 1
    return dict(f1_at_0_5=float(f1_at(y, inf["sw_prob"], 0.5)), best_threshold=float(grid[best]),
                f1_at_best=float(f1s[best]),
                duration_acc=float(np.mean(inf["dur_pred"][sw] == dur_true[sw])) if sw.any() else float("nan"),
                n_positions=int(len(y)), switch_rate=float(sw.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-5, help="encoder learning rate")
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--lambda_sw", type=float, default=1.0)
    ap.add_argument("--lambda_dur", type=float, default=1.0)
    ap.add_argument("--sw_pos_weight", type=float, default=1.0,
                    help="class weight of SWITCH in the switch loss (1.0 = unweighted)")
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--limit_per_pair", type=int, default=0, help="debug: windows per pair (0 = all)")
    ap.add_argument("--max_steps", type=int, default=0, help="debug: stop after N steps")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
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
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"using DataParallel on {torch.cuda.device_count()} GPUs", flush=True)
        model = torch.nn.DataParallel(model)
    core = model.module if hasattr(model, "module") else model

    head_params = list(core.switch_head.parameters()) + list(core.duration_head.parameters())
    opt = torch.optim.AdamW([
        {"params": core.encoder.parameters(), "lr": args.lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=0.01)
    steps_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    total = steps_per_epoch * args.epochs if not args.max_steps else min(args.max_steps, steps_per_epoch * args.epochs)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    sw_weight = torch.tensor([1.0, args.sw_pos_weight], device=device) if args.sw_pos_weight != 1.0 else None

    collate = make_collate(train_rows, pad_id)
    meta = dict(vars(args), model_name=name, pad_id=pad_id, device=str(device))
    tlog = open(out / "train_log.csv", "w", newline="")
    tw = csv.writer(tlog); tw.writerow(["step", "epoch", "loss", "loss_switch", "loss_duration", "lr", "elapsed_s"])
    vlog_path = out / "val_log.csv"
    best_f1, step, t0 = -1.0, 0, time.time()
    acc = {"loss": 0.0, "sw": 0.0, "dur": 0.0, "n": 0}

    for epoch in range(1, args.epochs + 1):
        model.train()
        for b in length_bucketed_batches(train_rows, args.batch_size, shuffle=True, seed=args.seed + epoch):
            batch = collate(b)
            ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
            swl, durl = batch["sw_labels"].to(device), batch["dur_labels"].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                sw_logits, dur_logits = model(ids, mask)
            loss, l_sw, l_dur = multitask_loss(sw_logits, dur_logits, swl, durl,
                                               args.lambda_sw, args.lambda_dur, sw_weight)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            acc["loss"] += loss.item(); acc["sw"] += l_sw.item(); acc["dur"] += l_dur.item(); acc["n"] += 1
            if step % args.log_every == 0:
                n = acc["n"]
                tw.writerow([step, epoch, acc["loss"] / n, acc["sw"] / n, acc["dur"] / n,
                             sched.get_last_lr()[0], round(time.time() - t0, 1)])
                tlog.flush()
                print(f"ep{epoch} step {step}/{total} loss {acc['loss']/n:.4f} "
                      f"(sw {acc['sw']/n:.4f} dur {acc['dur']/n:.4f}) {time.time()-t0:.0f}s", flush=True)
                acc = {"loss": 0.0, "sw": 0.0, "dur": 0.0, "n": 0}
            if args.max_steps and step >= args.max_steps:
                break

        inf = run_inference(model, val_rows, pad_id, device, amp=amp)
        m = dict(epoch=epoch, step=step, **val_metrics(val_rows, inf))
        print("VAL", json.dumps(m), flush=True)
        new = not vlog_path.exists()
        with open(vlog_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(m)); new and w.writeheader(); w.writerow(m)
        if m["f1_at_best"] > best_f1:
            best_f1 = m["f1_at_best"]
            torch.save(core.state_dict(), out / "best.pt")
            meta.update(best_epoch=epoch, threshold=m["best_threshold"], best_val=m)
            json.dump(meta, open(out / "meta.json", "w"), indent=2)
        if args.max_steps and step >= args.max_steps:
            break
    tlog.close()
    print(f"done. best val F1 {best_f1:.4f} at epoch {meta.get('best_epoch')}, threshold {meta.get('threshold')}")


if __name__ == "__main__":
    main()
