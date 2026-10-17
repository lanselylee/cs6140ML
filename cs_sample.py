"""
Build the shared code-switching samples for 12 SwitchLingua X-English pairs.

  pip install -r requirements.txt
  python cs_sample.py --data-dir data/SwitchLingua_text              # all 12 pairs, 400 each
  python cs_sample.py --data-dir data/SwitchLingua_text --pairs fr hi  # just some pairs

Writes samples/samples_<pair>.jsonl. Commit those files: every teammate (Claude, GPT,
Gemini ...) must score the exact same positions, then run `cs_predict.py run/eval`.

Sampling (per pair, fixed seed):
  - balanced: n/2 positions where the next token switches language, n/2 where it doesn't
    (the natural switch rate is ~6%, so 400 natural samples would hold only ~25 switches)
  - at most one position per conversation, duplicate conversations skipped
  - the prefix must hold at least --min-context tokens
  - only positions where the current AND next token have a confident language label

Tokens: every Chinese/Japanese character is one token; everything else is one token per
word. Punctuation, digits and whitespace are not tokens. Multi-turn conversations are
joined with newlines, so earlier turns are part of the prefix.

Language labels (no gold word-level labels exist in SwitchLingua):
  - zh ja ko ar ru : by script (Latin letters = English), always confident
  - fr de es it ms tl : per-word wordfreq frequencies (English vs X); words that are
    common in both languages fall back to lingua's segment-level detection
  - hi : the file mixes Devanagari and romanized Hindi (Hinglish). Devanagari = Hindi.
    Romanized words are compared with an in-domain English word list (the English words
    of the 5 script pairs); unclear words take the language of their neighbours
  - capitalised words in the middle of a sentence that aren't clearly either language
    (names, acronyms) take the language of the previous word and are never sampled
Labels from fallbacks are marked weak; weak / name tokens are never the current or next
token of a sampled position, which keeps label noise out of the samples.
"""

import argparse, ast, collections, json, os, random, re

import pandas as pd

# code: (language name, CSV file, labelling method)
PAIRS = {
    "zh": ("Chinese", "Chinese_eng.csv", "script"),
    "ja": ("Japanese", "Japanese_eng.csv", "script"),
    "ko": ("Korean", "Korean_eng.csv", "script"),
    "ar": ("Arabic", "Arabic_eng.csv", "script"),
    "ru": ("Russian", "Russian_eng.csv", "script"),
    "fr": ("French", "French_eng.csv", "latin"),
    "de": ("German", "German_Eng.csv", "latin"),
    "es": ("Spanish", "Spanish_eng.csv", "latin"),
    "it": ("Italian", "Italian_eng.csv", "latin"),
    "ms": ("Malay", "Malay_eng.csv", "latin"),
    "tl": ("Filipino", "Philippines_eng.csv", "latin"),
    "hi": ("Hindi", "Hindi_eng.csv", "hinglish"),
}
CHAR_TOKEN_PAIRS = {"zh", "ja"}
SCRIPT_PAIRS = [p for p, (_, _, m) in PAIRS.items() if m == "script"]
LINGUA_NAME = {"fr": "FRENCH", "de": "GERMAN", "es": "SPANISH", "it": "ITALIAN",
               "ms": "MALAY", "tl": "TAGALOG"}
WORDFREQ_CODE = {"fr": "fr", "de": "de", "es": "es", "it": "it", "ms": "ms", "tl": "fil"}
SAMPLE_DIR = "samples"

WF_MARGIN = 1.0          # |zipf_en - zipf_X| needed to trust wordfreq on its own
HI_RATIO, EN_RATIO = 8.0, 2.0   # Hinglish: relative frequency vs in-domain English
# romanized Hindi words spelled like common English words ("hote the" = were, "us samay"
# = that time); always unclear, so they are never the current/next token of a sample
HINDI_HOMOGRAPHS = {"the", "us", "main", "me", "to", "do", "par", "so", "hi", "na", "are",
                    "bus", "log", "ho", "mere", "tab", "jab", "pe", "ye", "kya", "din", "hum"}

# letters plus the combining marks of Latin, Arabic and Devanagari (Devanagari digits
# and danda excluded), so words aren't split at vowel signs
_LETTER = r"(?:[^\W\d_]|[̀-ًͯ-ٰٟऀ-ॣ॰-ॿ])"
WORD_RE = re.compile(rf"{_LETTER}+(?:['’]{_LETTER}+)*")
SENTENCE_END = re.compile(r"[.!?。！？\n]")


def sample_path(pair):
    return os.path.join(SAMPLE_DIR, f"samples_{pair}.jsonl")


# ---------------------------------------------------------------- text + tokens
def parse_turns(raw):
    """data_generation_result is a stringified list of turns (a few rows are dicts)."""
    try:
        obj = ast.literal_eval(raw)
    except Exception:
        return [str(raw)]
    if isinstance(obj, str):
        return [obj]
    turns = []
    for t in obj:
        if isinstance(t, dict):
            turns.extend(str(v) for v in t.values())
        else:
            turns.append(str(t))
    return turns


def conversation_text(raw):
    return "\n".join(t.strip() for t in parse_turns(raw))


def script_of(ch):
    o = ord(ch)
    if ch in "'’":
        return None
    if 0x3040 <= o <= 0x30FF or 0x3400 <= o <= 0x4DBF or 0x4E00 <= o <= 0x9FFF \
            or 0xF900 <= o <= 0xFAFF or o == 0x3005:
        return "cjk"
    if 0xAC00 <= o <= 0xD7AF or 0x1100 <= o <= 0x11FF or 0x3130 <= o <= 0x318F:
        return "hangul"
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0xFB50 <= o <= 0xFDFF \
            or 0xFE70 <= o <= 0xFEFF:
        return "arabic"
    if 0x0400 <= o <= 0x052F:
        return "cyrillic"
    if 0x0900 <= o <= 0x097F:
        return "devanagari"
    if o < 0x0250 or 0x1E00 <= o <= 0x1EFF or 0x0300 <= o <= 0x036F:
        return "latin"
    return "other"


def tokenize(text):
    """Words split further where the script changes; CJK runs split into characters."""
    toks = []

    def emit(start, end, script):
        if script == "cjk":
            toks.extend({"tok": text[i], "start": i, "end": i + 1, "script": script}
                        for i in range(start, end))
        else:
            toks.append({"tok": text[start:end], "start": start, "end": end, "script": script})

    for m in WORD_RE.finditer(text):
        start, script = m.start(), None
        for i in range(m.start(), m.end()):
            s = script_of(text[i])
            if s is None or s == script:
                continue
            if script is not None:
                emit(start, i, script)
                start = i
            script = s
        emit(start, m.end(), script)
    for k, tk in enumerate(toks):
        gap = text[toks[k - 1]["end"]:tk["start"]] if k else ""
        tk["initial"] = k == 0 or bool(SENTENCE_END.search(gap))
    return toks


def norm(word):
    return word.lower().replace("’", "'")


# ---------------------------------------------------------------- language labels
# each labeller returns [(lang, conf)], conf in {"strong", "weak", "name"}

def label_script(toks, pair):
    return [("en" if tk["script"] == "latin" else pair, "strong") for tk in toks]


def looks_like_name(tk):
    w = tk["tok"]
    return (len(w) >= 2 and w.isupper()) or (w[0].isupper() and not tk["initial"])


def fill_names(labels, pair):
    """Names/acronyms take the previous word's language (or the next one's at the start)."""
    out = list(labels)
    for i, (lang, conf) in enumerate(out):
        if lang is None:
            prev = next((out[j][0] for j in range(i - 1, -1, -1) if out[j][0]), None)
            nxt = next((labels[j][0] for j in range(i + 1, len(labels)) if labels[j][0]), None)
            out[i] = (prev or nxt or pair, conf)
    return out


class LatinLabeller:
    def __init__(self, pair):
        from lingua import Language, LanguageDetectorBuilder
        from wordfreq import zipf_frequency
        self.pair, self.zipf, self.wf = pair, zipf_frequency, WORDFREQ_CODE[pair]
        self.other = getattr(Language, LINGUA_NAME[pair])
        self.detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, self.other).build()

    def __call__(self, text, toks):
        spans = self.detector.detect_multiple_languages_of(text)

        def span_lang(tk):
            for r in spans:
                if r.start_index <= tk["start"] < r.end_index:
                    return "en" if r.language != self.other else self.pair
            return None

        labels = []
        for tk in toks:
            if tk["script"] != "latin":
                labels.append((self.pair, "strong"))
                continue
            w = norm(tk["tok"])
            d = self.zipf(w, "en") - self.zipf(w, self.wf)
            if abs(d) >= WF_MARGIN:
                labels.append(("en" if d > 0 else self.pair, "strong"))
            elif looks_like_name(tk):
                labels.append((None, "name"))
            else:
                labels.append((span_lang(tk) or ("en" if d > 0 else self.pair), "weak"))
        return fill_names(labels, self.pair)


def latin_words(texts):
    c = collections.Counter()
    for text in texts:
        c.update(norm(tk["tok"]) for tk in tokenize(text) if tk["script"] == "latin")
    return c


class HinglishLabeller:
    """Romanized Hindi vs English by relative frequency against in-domain English."""

    def __init__(self, hindi_texts, english_words):
        from wordfreq import zipf_frequency
        self.zipf = zipf_frequency
        self.hi, self.en = latin_words(hindi_texts), english_words
        self.n_hi, self.n_en = sum(self.hi.values()), sum(self.en.values())

    def ratio(self, w):
        return (self.hi[w] / self.n_hi) / ((self.en[w] + 0.5) / self.n_en)

    def __call__(self, text, toks):
        labels = []
        for tk in toks:
            if tk["script"] != "latin":
                labels.append(("hi", "strong"))
                continue
            w = norm(tk["tok"])
            r = self.ratio(w)
            if w in HINDI_HOMOGRAPHS:
                labels.append(("?", "weak"))
            elif self.hi[w] >= 3 and r >= HI_RATIO:
                labels.append(("hi", "strong"))
            elif self.en[w] >= 3 and r <= EN_RATIO:
                labels.append(("en", "strong"))
            elif looks_like_name(tk):
                labels.append((None, "name"))
            else:
                labels.append(("?", "weak"))
        # unclear words: majority of the confident words within 3 on either side
        out = list(labels)
        for i, (lang, conf) in enumerate(labels):
            if lang != "?":
                continue
            near = [labels[j][0] for j in range(max(0, i - 3), min(len(labels), i + 4))
                    if labels[j][1] == "strong"]
            en, hi = near.count("en"), near.count("hi")
            if en != hi:
                out[i] = ("en" if en > hi else "hi", "weak")
            else:
                out[i] = ("en" if self.zipf(norm(toks[i]["tok"]), "en") >= 3.5 else "hi", "weak")
        return fill_names(out, "hi")


# ---------------------------------------------------------------- positions + sampling
def duration_bin(n):
    return "1-2" if n <= 2 else ("3-6" if n <= 6 else "7+")


def candidates(toks, labels, min_context):
    """(t, switch) for every position whose current and next token are confidently labelled."""
    out = []
    for t in range(min_context - 1, len(toks) - 1):
        if labels[t][1] == "strong" and labels[t + 1][1] == "strong":
            out.append((t, labels[t + 1][0] != labels[t][0]))
    return out


def make_record(pair, conv_id, text, toks, labels, t):
    cur, nxt = labels[t][0], labels[t + 1][0]
    dur = None
    if cur != nxt:
        k = t + 1
        while k < len(toks) and labels[k][0] == nxt:
            k += 1
        dur = duration_bin(k - (t + 1))
    return {
        "pair": pair, "conv_id": conv_id, "t": t,
        "prefix": text[: toks[t]["end"]],              # cut right after token t
        "cur_token": toks[t]["tok"], "cur_lang": cur,
        "next_token": toks[t + 1]["tok"],              # for auditing only, never shown to models
        "label_switch": int(cur != nxt), "label_duration": dur,
    }


def sample_pair(pair, texts, labeller, n, min_context, seed):
    rng = random.Random(f"{seed}-{pair}")
    order = list(range(len(texts)))
    rng.shuffle(order)
    pos, neg, seen = [], [], set()
    stats = collections.Counter()
    for conv_id in order:
        if len(pos) >= n // 2 and len(neg) >= n - n // 2:
            break
        text = texts[conv_id]
        if not text or text in seen:
            stats["duplicate_or_empty"] += 1
            continue
        seen.add(text)
        toks = tokenize(text)
        labels = labeller(text, toks)
        stats["conversations"] += 1
        stats["tokens"] += len(toks)
        stats["weak_or_name_tokens"] += sum(conf != "strong" for _, conf in labels)
        cands = candidates(toks, labels, min_context)
        stats["candidates"] += len(cands)
        stats["candidate_switches"] += sum(sw for _, sw in cands)
        switches = [t for t, sw in cands if sw]
        stays = [t for t, sw in cands if not sw]
        # alternate classes so both are drawn from the same random stream of conversations
        want_pos = len(pos) < n // 2 and (len(pos) <= len(neg) or len(neg) >= n - n // 2)
        for take_pos in (want_pos, not want_pos):
            bucket, pool, cap = (pos, switches, n // 2) if take_pos else (neg, stays, n - n // 2)
            if pool and len(bucket) < cap:
                bucket.append(make_record(pair, conv_id, text, toks, labels, rng.choice(pool)))
                break
    if len(pos) < n // 2 or len(neg) < n - n // 2:
        raise SystemExit(f"{pair}: only {len(pos)} switches / {len(neg)} non-switches found")
    sample = pos + neg
    rng.shuffle(sample)
    return sample, stats


def read_texts(data_dir, pair):
    df = pd.read_csv(os.path.join(data_dir, PAIRS[pair][1]), usecols=["data_generation_result"])
    return [conversation_text(raw) if isinstance(raw, str) else "" for raw in df["data_generation_result"]]


def make_labeller(pair, data_dir, texts):
    method = PAIRS[pair][2]
    if method == "script":
        return lambda text, toks: label_script(toks, pair)
    if method == "latin":
        return LatinLabeller(pair)
    print("hi: building the in-domain English word list from", " ".join(SCRIPT_PAIRS), "...")
    english = collections.Counter()
    for p in SCRIPT_PAIRS:
        english += latin_words(read_texts(data_dir, p))
    return HinglishLabeller(texts, english)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/SwitchLingua_text", help="folder with the SwitchLingua CSVs")
    ap.add_argument("--pairs", nargs="+", default=list(PAIRS), choices=list(PAIRS))
    ap.add_argument("--n", type=int, default=400, help="samples per pair (half switches)")
    ap.add_argument("--min-context", type=int, default=5, help="minimum prefix length in tokens")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)

    os.makedirs(SAMPLE_DIR, exist_ok=True)
    for pair in a.pairs:
        texts = read_texts(a.data_dir, pair)
        sample, st = sample_pair(pair, texts, make_labeller(pair, a.data_dir, texts),
                                 a.n, a.min_context, a.seed)
        with open(sample_path(pair), "w", encoding="utf-8") as f:
            for r in sample:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        durs = collections.Counter(r["label_duration"] for r in sample if r["label_switch"])
        print(f"{pair}: scanned {st['conversations']} convs, natural switch rate "
              f"{st['candidate_switches'] / st['candidates']:.1%}, weak/name tokens "
              f"{st['weak_or_name_tokens'] / st['tokens']:.1%}, durations "
              f"{dict(sorted(durs.items()))} -> {sample_path(pair)}")


if __name__ == "__main__":
    main()
