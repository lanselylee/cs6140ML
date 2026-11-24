"""Streaming simulation (spec deliverable 2): read text word by word and, after each
word, print the next-word switch probability and the anticipated burst duration.

  python streaming/demo.py --run results/phase2/mbert --text "我今天去 meeting 然后 we can grab lunch"
  echo "Je pense que this is a good idea pour le projet" | python streaming/demo.py --run RUN
  python streaming/demo.py --run RUN                     # interactive: type a sentence per line
  python streaming/demo.py --run RUN --text "..." --delay 0.4   # reveal one word at a time
  python streaming/demo.py --run RUN --text "..." --json        # one JSON object per word

--run is a training output folder holding best.pt and meta.json (train.py writes both).

It is a real streaming loop: at word t the model is run on the prefix x1..xt only
(nothing after t is ever given to it), and the prediction is read at the last
sub-word of word t, exactly as in training. Words are segmented and encoded with
Phase 1's rules (script runs, jieba for Chinese, fugashi for Japanese), so the model
sees the same sub-words it was trained on.
Long inputs keep only the most recent words that fit in 254 sub-words.
"""
import argparse
import json
import sys
import time
import unicodedata
from pathlib import Path

import regex as re
import torch
from transformers import AutoTokenizer

from model import CausalDualHead
from train import get_device

DUR_BINS = ["1-2 words", "3-6 words", "7+ words"]
MAX_SUBWORDS = 254  # 256 minus [CLS]/<s> and [SEP]/</s>


def load(run, device):
    run = Path(run)
    meta = json.load(open(run / "meta.json"))
    model = CausalDualHead(meta["model_name"]).to(device)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.eval()
    tok = AutoTokenizer.from_pretrained(meta["model_name"])
    return model, tok, meta


# ---- word segmentation: same rules as Phase 1 code/prepare.py (Labeler.words) ----
WORD = re.compile(r"\p{Han}+|[\p{Hiragana}\p{Katakana}ー]+|\p{Hangul}+|"
                  r"\p{Arabic}[\p{Arabic}\p{M}]*|\p{Devanagari}[\p{Devanagari}\p{M}]*|"
                  r"\p{Cyrillic}[\p{Cyrillic}\p{M}]*|"
                  r"\p{Latin}[\p{Latin}\p{M}]*(?:['’\-]\p{Latin}[\p{Latin}\p{M}]*)*|\p{N}+(?:[.,:]\p{N}+)*")
JP = re.compile(r"[\p{Han}\p{Hiragana}\p{Katakana}ー]+")
HAN, KANA = re.compile(r"\p{Han}"), re.compile(r"[\p{Hiragana}\p{Katakana}ー]")
_JA_TAGGER = None


def guess_lang(text):
    """'ja' if there is kana, 'zh' if there is Han without kana, else 'other'."""
    return "ja" if KANA.search(text) else "zh" if HAN.search(text) else "other"


def segment(text, lang):
    """Word spans as Phase 1 made them: script runs, jieba for Chinese, fugashi for
    Japanese; punctuation and spaces attach to the preceding word."""
    global _JA_TAGGER
    if lang == "ja":
        if _JA_TAGGER is None:
            import fugashi
            _JA_TAGGER = fugashi.Tagger()
        regions, cursor = [], 0
        for m in JP.finditer(text):
            regions += [(x.start() + cursor, x.end() + cursor) for x in WORD.finditer(text[cursor:m.start()])]
            pos = m.start()
            for token in _JA_TAGGER(m.group()):
                start = text.find(token.surface, pos, m.end())
                regions.append((start, start + len(token.surface)))
                pos = start + len(token.surface)
            cursor = m.end()
        regions += [(x.start() + cursor, x.end() + cursor) for x in WORD.finditer(text[cursor:])]
    else:
        regions = [(m.start(), m.end()) for m in WORD.finditer(text)]
    base = []
    for a, b in regions:
        if lang == "zh" and HAN.search(text[a:b]):
            import jieba
            jieba.setLogLevel(40)
            base += [(a + s, a + e) for _, s, e in jieba.tokenize(text[a:b])]
        else:
            base.append((a, b))
    spans = []
    for i, (a, b) in enumerate(base):
        start = a if i else 0
        end = base[i + 1][0] if i + 1 < len(base) else len(text)
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        spans.append((start, end))
    return spans


def word_pieces(text, tok, lang="auto"):
    """Segment into words, then encode each word together with the text since the
    previous word (Phase 1's fragment rule), so sub-words match the training data."""
    words, pieces, cursor = [], [], 0
    for a, b in segment(text, guess_lang(text) if lang == "auto" else lang):
        ids = tok.encode(text[cursor:b], add_special_tokens=False)
        cursor = b
        if ids:
            words.append(text[a:b])
            pieces.append(ids)
    return words, pieces


@torch.no_grad()
def predict_prefix(model, tok, pieces, device):
    """Run the model on the prefix only and read the heads at its last sub-word."""
    keep, n = [], 0
    for p in reversed(pieces):  # most recent words that fit the context window
        if n + len(p) > MAX_SUBWORDS:
            break
        keep.insert(0, p); n += len(p)
    ids = [tok.cls_token_id] + [i for p in keep for i in p] + [tok.sep_token_id]
    x = torch.tensor([ids], device=device)
    sw_logits, dur_logits = model(x, torch.ones_like(x))
    last = len(ids) - 2  # last sub-word of word t (the [SEP] after it is never attended to by it)
    p_switch = torch.softmax(sw_logits[0, last].float(), -1)[1].item()
    p_dur = torch.softmax(dur_logits[0, last].float(), -1).tolist()
    return p_switch, p_dur


def pad(s, width):
    """Left-justify by display width (CJK characters take two columns)."""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(1, width - w)


def stream(text, model, tok, meta, device, delay=0.0, as_json=False, lang="auto"):
    words, pieces = word_pieces(text, tok, lang)
    thr = meta.get("threshold", 0.5)
    if not as_json:
        print(f"\n{'t':>3}  {'word':<18}{'P(switch next)':>14}  {'':<20} anticipated duration")
    for t in range(len(words)):
        p, pd = predict_prefix(model, tok, pieces[:t + 1], device)
        k = max(range(3), key=lambda i: pd[i])
        if as_json:
            print(json.dumps(dict(t=t + 1, word=words[t], p_switch=round(p, 4), switch=p >= thr,
                                  duration=DUR_BINS[k], p_duration=[round(v, 4) for v in pd]),
                             ensure_ascii=False), flush=True)
        else:
            bar = "#" * round(p * 20)
            flag = f"SWITCH -> {DUR_BINS[k]} ({pd[k]:.2f})" if p >= thr else ""
            print(f"{t + 1:>3}  {pad(words[t], 18)}{p:>14.3f}  {bar:<20} {flag}", flush=True)
        if delay:
            time.sleep(delay)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="folder with best.pt and meta.json")
    ap.add_argument("--text", help="text to stream; omitted: read stdin (interactive if a terminal)")
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait between words")
    ap.add_argument("--json", action="store_true", help="one JSON object per word")
    ap.add_argument("--lang", choices=["auto", "zh", "ja", "other"], default="auto",
                    help="word segmentation: jieba for zh, fugashi for ja (auto: from the script)")
    args = ap.parse_args()

    device = get_device()
    model, tok, meta = load(args.run, device)
    if not args.json:
        print(f"model {meta['model_name']} (step {meta.get('best_step', '?')}), device {device}, "
              f"switch threshold {meta.get('threshold', 0.5)}", file=sys.stderr)
    if args.text is not None:
        stream(args.text, model, tok, meta, device, args.delay, args.json, args.lang)
        return
    if sys.stdin.isatty():
        print("Type a code-switched sentence (empty line to quit).", file=sys.stderr)
    for line in sys.stdin:
        if not line.strip():
            if sys.stdin.isatty():
                break
            continue
        stream(line, model, tok, meta, device, args.delay, args.json, args.lang)


if __name__ == "__main__":
    main()
