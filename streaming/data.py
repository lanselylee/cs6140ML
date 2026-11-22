"""Load the shared Phase 1 encodings (encoded/<model>/<split>.pt) for training.

Each item is one window. Only positions with sw_labels != -1 are scored
(the last sub-word of each word owned by that window).
"""
import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

EXTRA_PAIRS = {"Cantonese-English", "German-French", "Vietnamese-English"}


def load_windows(root, model, split, pairs=None, limit_per_pair=0, seed=0):
    rows = torch.load(Path(root) / "encoded" / model / f"{split}.pt", map_location="cpu", weights_only=False)
    rows = [r for r in rows if r["pair"] not in EXTRA_PAIRS]
    if pairs:
        rows = [r for r in rows if r["pair"] in pairs]
    if limit_per_pair:
        rng = random.Random(seed)
        by_pair = {}
        for r in rows:
            by_pair.setdefault(r["pair"], []).append(r)
        rows = []
        for p in sorted(by_pair):
            rng.shuffle(by_pair[p])
            rows.extend(by_pair[p][:limit_per_pair])
    return rows


def load_gold(root, split):
    gold = {}
    with (Path(root) / "labels" / f"{split}_labels.jsonl").open(encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)
            if s["pair"] not in EXTRA_PAIRS:
                gold[s["id"]] = s
    return gold


class WindowDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return i


def make_collate(rows, pad_id):
    def collate(idxs):
        batch = [rows[i] for i in idxs]
        n = max(len(r["input_ids"]) for r in batch)

        def pad(key, value):
            return torch.tensor([r[key] + [value] * (n - len(r[key])) for r in batch], dtype=torch.long)

        return dict(idx=torch.tensor(idxs), input_ids=pad("input_ids", pad_id),
                    attention_mask=pad("attention_mask", 0),
                    sw_labels=pad("sw_labels", -1), dur_labels=pad("dur_labels", -1))
    return collate


def length_bucketed_batches(rows, batch_size, shuffle, seed=0):
    """Group windows of similar length so padding is small (big speed-up: mean length is ~30)."""
    idx = list(range(len(rows)))
    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(idx)
    chunk = batch_size * 50
    batches = []
    for c in range(0, len(idx), chunk):
        part = sorted(idx[c:c + chunk], key=lambda i: len(rows[i]["input_ids"]))
        batches += [part[b:b + batch_size] for b in range(0, len(part), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches
