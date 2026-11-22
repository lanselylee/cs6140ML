"""Encode the shared Phase 1 word-level labels for another tokenizer (e.g. mBERT).

Uses the SAME word-aligned fragment encoding and windowing as the group's
Phase 1 code (functions copied verbatim from code/prepare.py: token_pieces,
windows, encode_window), so the new model sees the same words, labels and
scored positions. Windows are computed jointly with XLM-R's lengths, so window
boundaries stay identical to encoded/xlmr.

  # sanity check: re-encode XLM-R and compare with the shipped encoded/xlmr
  python streaming/encode_model.py --data_root DATA --model xlmr --verify
  # produce encoded/mbert/{train,val,test}.pt
  python streaming/encode_model.py --data_root DATA --model mbert
"""
import argparse
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model import MODELS

XLMR_REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"  # from Phase 1 manifest.json


# ---- copied from Phase 1 code/prepare.py (unchanged logic) ----
@lru_cache(maxsize=150000)
def _fragment_ids(tok_key, fragment):
    return tuple(TOKENIZERS[tok_key].encode(fragment, add_special_tokens=False))


def token_pieces(sample, tok_key):
    pieces = []; cursor = 0
    for word, (start, end) in zip(sample["words"], sample["word_spans"]):
        fragment = sample["text"][cursor:end]
        ids = list(_fragment_ids(tok_key, fragment))
        assert ids, (sample["id"], word, "no subword")
        pieces.append(ids)
        cursor = end
    return pieces


def windows(lengths, capacities, overlap=24):
    n = len(next(iter(lengths.values()))); start = 0; owned = 0
    while owned < n:
        end = start; sums = {k: 0 for k in lengths}
        while end < n and all(sums[k] + lengths[k][end] <= capacities[k] for k in lengths):
            for k in sums: sums[k] += lengths[k][end]
            end += 1
        if end <= owned:
            if start < owned: start = owned; continue
            raise ValueError(f"Single word exceeds tokenizer capacity at word {owned}")
        yield start, end, owned
        owned = end
        if end == n: break
        start = max(start + 1, end - overlap)


def encode_window(s, pieces, tok, start, end, owned, windex):
    ids = []; word_ids = []; sws = []; durs = []; langs = []
    for i in range(start, end):
        part = pieces[i]; ids.extend(part); word_ids.extend([i] * len(part))
        langs.extend([s["lang"][i]] * len(part)); sws.extend([-1] * len(part)); durs.extend([-1] * len(part))
        if i >= owned: sws[-1] = s["sw"][i]; durs[-1] = s["dur"][i]
    full = add_special(tok, ids)
    probe = add_special(tok, [-991])
    marker = probe.index(-991)
    assert full[marker:marker + len(ids)] == ids
    mask = [1] * marker + [0] * len(ids) + [1] * (len(probe) - marker - 1)
    fields = [word_ids, langs, sws, durs]; padded = [[] for _ in fields]; j = 0
    for special in mask:
        for dest, src, default in zip(padded, fields, [-1, None, -1, -1]):
            dest.append(default if special else src[j])
        if not special: j += 1
    assert j == len(ids) and len(full) <= 256
    return dict(id=s["id"], window_id=f"{s['id']}:w{windex}", pair=s["pair"],
                word_start=start, word_end=end, owned_word_start=owned, input_ids=full,
                attention_mask=[1] * len(full), word_ids=padded[0], sub_lang=padded[1],
                sw_labels=padded[2], dur_labels=padded[3])
# ---- end of copied code ----


TOKENIZERS = {}


def add_special(tok, ids):
    """[CLS]/<s> + ids + [SEP]/</s>: what build_inputs_with_special_tokens did for
    single sequences in transformers 4.x (removed in 5.x). Verified against the
    shipped XLM-R encoding with --verify."""
    return [tok.cls_token_id] + list(ids) + [tok.sep_token_id]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--verify", action="store_true", help="compare with the shipped encoded/<model> files")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()
    root = Path(args.data_root)

    TOKENIZERS["xlmr"] = AutoTokenizer.from_pretrained(MODELS["xlmr"], revision=XLMR_REVISION)
    TOKENIZERS[args.model] = TOKENIZERS.get(args.model) or AutoTokenizer.from_pretrained(MODELS[args.model])
    tok = TOKENIZERS[args.model]

    for split in args.splits:
        kept = set((root / "encoded" / f"kept_ids_{split}.txt").read_text().split())
        # Window boundaries come from the shipped XLM-R encoding (which Phase 1 computed
        # jointly with Qwen), so every model sees exactly the same context per window.
        bounds = {}
        for r in torch.load(root / "encoded" / "xlmr" / f"{split}.pt", weights_only=False):
            bounds.setdefault(r["id"], []).append((r["word_start"], r["word_end"], r["owned_word_start"]))
        out, stats = [], Counter()
        with (root / "labels" / f"{split}_labels.jsonl").open(encoding="utf-8") as f:
            for line in f:
                s = json.loads(line)
                if s["id"] not in kept:
                    continue
                pieces = token_pieces(s, args.model)
                lengths = {args.model: [len(p) for p in pieces]}
                cap = 256 - len(add_special(tok, []))
                win = bounds[s["id"]]
                if any(sum(lengths[args.model][a:b]) > cap for a, b, _ in win):
                    # Rare: the shipped window is too long for this tokenizer. Re-window this
                    # instance with Phase 1's windows() over both tokenizers' lengths. Every word
                    # is still scored exactly once; only the context split differs. Logged below.
                    joint = {"xlmr": [len(p) for p in token_pieces(s, "xlmr")], args.model: lengths[args.model]}
                    win = list(windows(joint, {"xlmr": 254, args.model: cap}))
                    stats["rewindowed"] += 1
                    print(f"  re-windowed {s['id']}: {len(bounds[s['id']])} -> {len(win)} windows", flush=True)
                for wi, (a, b, o) in enumerate(win):
                    out.append(encode_window(s, pieces, tok, a, b, o, wi))  # asserts <= 256 tokens
                stats["words"] += len(s["words"])
                stats["subwords"] += sum(lengths[args.model])
                if tok.unk_token_id is not None:
                    stats["unk_words"] += sum(tok.unk_token_id in p for p in pieces)
        print(f"{split}: {len(out)} windows, subwords/word {stats['subwords']/stats['words']:.3f}, "
              f"words containing [UNK] {stats['unk_words']} ({stats['unk_words']/stats['words']:.4%})", flush=True)

        if args.verify:
            ref = torch.load(root / "encoded" / args.model / f"{split}.pt", weights_only=False)
            assert len(ref) == len(out), (len(ref), len(out))
            ref_by = {r["window_id"]: r for r in ref}
            keys = ["input_ids", "word_ids", "sw_labels", "dur_labels", "word_start", "word_end", "owned_word_start"]
            bad = sum(any(ref_by[r["window_id"]][k] != r[k] for k in keys) for r in out)
            print(f"  verify vs shipped encoded/{args.model}/{split}.pt: {bad} mismatching windows")
            assert bad == 0
        else:
            dest = root / "encoded" / args.model
            dest.mkdir(parents=True, exist_ok=True)
            torch.save(out, dest / f"{split}.pt")
            print(f"  saved {dest / f'{split}.pt'}")


if __name__ == "__main__":
    main()
