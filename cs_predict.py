"""
Prefix-only code-switching prediction with LLM APIs (SwitchLingua, 12 X-English pairs:
zh ja ko ar ru fr de es it ms tl hi).

Three steps:
  1. sample   : label tokens, draw balanced positions (cs_sample.py)    -> samples/samples_*.jsonl
  2. run      : ask a model at each sampled position (prefix only)      -> preds/preds_*.jsonl
  3. eval     : compute Switch F1 (+ duration macro-F1)                 -> printed table

Usage:
  pip install -r requirements.txt

  # 1. sample ONCE (already done - samples/ is committed, so you can skip this)
  python cs_predict.py sample --data-dir data/SwitchLingua_text

  # 2. run: Claude (Anthropic SDK)
  export ANTHROPIC_API_KEY=sk-ant-...
  python cs_predict.py run --provider anthropic --model claude-opus-5-5 --effort low

  # 2. run: Claude via Claude Code on your Claude Pro/Max subscription (no API key)
  python cs_predict.py run --provider claude-cli --model claude-opus-5-5 --effort low

  # 2. run: any OpenAI-compatible API (OpenAI, DeepSeek, Qwen, Gemini, OpenRouter ...)
  export OPENAI_API_KEY=sk-...
  python cs_predict.py run --provider openai --model gpt-4o --temperature 0
  export DEEPSEEK_API_KEY=sk-...
  python cs_predict.py run --provider openai --model deepseek-chat \
      --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --temperature 0
  export GEMINI_API_KEY=...
  python cs_predict.py run --provider openai --model gemini-2.5-flash \
      --base-url https://generativelanguage.googleapis.com/v1beta/openai/ --api-key-env GEMINI_API_KEY

  # 3. eval
  python cs_predict.py eval --model claude-opus-5-5

Token definition and language labels: see cs_sample.py.
"""

import argparse, asyncio, json, os, re


from cs_sample import CHAR_TOKEN_PAIRS, PAIRS, sample_path

PRED_DIR = "preds"


def pred_path(pair, model):
    # model names like "openai/gpt-4o" (OpenRouter) would otherwise create sub-directories
    return os.path.join(PRED_DIR, f"preds_{pair}_{re.sub(r'[^A-Za-z0-9._-]', '_', model)}.jsonl")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ---------------------------------------------------------------- step 2: ask the model
SYSTEM = """You are evaluating code-switching in {name}-English text.
You see a text PREFIX that stops exactly after the current token. You cannot see anything after it.

Token definition: {token_rule}; punctuation, digits and whitespace are ignored.

Predict:
1. switch: will the NEXT token be in a different language from the current token?
2. duration: if switch is true, how many consecutive tokens will the new-language segment last?
   One of "1-2", "3-6", "7+". If switch is false, use null.

Reply with only a JSON object, e.g. {{"switch": true, "duration": "3-6"}} or {{"switch": false, "duration": null}}."""

USER = """Prefix:
<<<
{prefix}
>>>
Current token: "{tok}" (language: {lang})"""


def parse_reply(text):
    m = re.search(r"\{.*?\}", text, re.S)
    try:
        obj = json.loads(m.group()) if m else {}
    except Exception:
        obj = {}
    sw = obj.get("switch")
    if isinstance(sw, str):
        sw = sw.lower() == "true"
    dur = obj.get("duration")
    return (int(bool(sw)) if sw is not None else 0,
            dur if dur in ("1-2", "3-6", "7+") else None,
            sw is not None)


class Fatal(Exception):
    """Non-retryable error (bad key, unknown model, unsupported parameter)."""


CLI_LIMIT_WORDS = ("usage limit", "rate limit", "limit reached", "credit balance", "not logged in",
                   "please run /login", "invalid api key")


def make_claude_cli_caller(a):
    """Claude via the Claude Code CLI (`claude -p`), billed to the logged-in Claude subscription
    instead of an API key. Our SYSTEM prompt replaces Claude Code's; tools, MCP servers,
    skills, settings and session files are all turned off."""
    import shutil, tempfile
    exe = shutil.which("claude")
    if not exe:
        raise SystemExit("`claude` (Claude Code CLI) not found on PATH")
    workdir = tempfile.mkdtemp(prefix="cs_predict_")   # empty dir: no CLAUDE.md gets picked up
    base = [exe, "-p", "--model", a.model, "--output-format", "json", "--tools", "",
            "--strict-mcp-config", "--disable-slash-commands", "--setting-sources", "",
            "--no-session-persistence"] + (["--effort", a.effort] if a.effort else [])

    async def call(system, user):
        proc = await asyncio.create_subprocess_exec(
            *base, "--system-prompt", system, cwd=workdir,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            out, err = await asyncio.wait_for(proc.communicate(user.encode("utf-8")), timeout=300)
        except asyncio.TimeoutError:
            proc.kill()
            raise
        try:
            d = json.loads(out)
        except ValueError:
            d = {"is_error": True, "result": (err or out).decode("utf-8", "replace")[-500:]}
        if proc.returncode or d.get("is_error"):
            msg = str(d.get("result") or err.decode("utf-8", "replace")[-500:])
            if any(w in msg.lower() for w in CLI_LIMIT_WORDS):
                # stop instead of burning retries; rerun the same command later, it resumes
                raise Fatal(RuntimeError(msg)) from RuntimeError(msg)
            raise RuntimeError(msg)
        return d.get("result") or "", d.get("stop_reason") or d.get("subtype")
    return call


def make_caller(a):
    """Return an async fn(system, user) -> (text, stop_reason) for the chosen provider."""
    if a.provider == "claude-cli":
        return make_claude_cli_caller(a)
    if a.provider == "anthropic":
        import anthropic
        client = anthropic.AsyncAnthropic(max_retries=0)       # we retry ourselves below
        extra = {"output_config": {"effort": a.effort}} if a.effort else {}
        if a.temperature is not None:
            extra["temperature"] = a.temperature              # rejected by Opus 4.7+ / Fable models

        async def call(system, user):
            try:
                resp = await client.messages.create(
                    model=a.model, max_tokens=a.max_tokens, system=system,
                    messages=[{"role": "user", "content": user}], **extra)
            except (anthropic.BadRequestError, anthropic.AuthenticationError,
                    anthropic.PermissionDeniedError, anthropic.NotFoundError) as e:
                raise Fatal(e) from e
            return "".join(b.text for b in resp.content if b.type == "text"), resp.stop_reason
        return call

    import openai
    client = openai.AsyncOpenAI(api_key=os.environ.get(a.api_key_env), base_url=a.base_url,
                                max_retries=0)
    extra = {"temperature": a.temperature} if a.temperature is not None else {}
    # official OpenAI reasoning models only accept max_completion_tokens; most compatible APIs only max_tokens
    extra["max_completion_tokens" if a.base_url is None else "max_tokens"] = a.max_tokens

    async def call(system, user):
        try:
            resp = await client.chat.completions.create(
                model=a.model, **extra,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        except (openai.BadRequestError, openai.AuthenticationError,
                openai.PermissionDeniedError, openai.NotFoundError) as e:
            raise Fatal(e) from e
        ch = resp.choices[0]
        return ch.message.content or "", ch.finish_reason
    return call


async def run_pair(call, a, pair):
    src, out = sample_path(pair), pred_path(pair, a.model)
    samples = read_jsonl(src)
    done = set()
    if os.path.exists(out):                         # resume support
        done = {(d["conv_id"], d["t"]) for d in read_jsonl(out)}
    if a.limit:                                     # pilot runs: first N samples of each pair
        samples = samples[: a.limit]
    todo = [s for s in samples if (s["conv_id"], s["t"]) not in done]
    print(f"{pair}: {len(todo)} to run ({len(done)} already done) -> {out}")
    sem, lock = asyncio.Semaphore(a.concurrency), asyncio.Lock()
    name = PAIRS[pair][0]
    token_rule = (f"every {name} character is one token; every English word is one token"
                  if pair in CHAR_TOKEN_PAIRS else f"every word ({name} or English) is one token")
    failed = 0

    async def one(s):
        nonlocal failed
        async with sem:
            for attempt in range(5):
                try:
                    text, stop = await call(SYSTEM.format(name=name, token_rule=token_rule), USER.format(
                        prefix=s["prefix"], tok=s["cur_token"],
                        lang=name if s["cur_lang"] != "en" else "English"))
                    break
                except Fatal:
                    raise
                except Exception as e:
                    if attempt == 4:
                        failed += 1
                        print("failed:", s["conv_id"], s["t"], e); return
                    await asyncio.sleep(2 ** attempt)
            sw, dur, ok = parse_reply(text)
            rec = {**{k: s[k] for k in ("conv_id", "t", "label_switch", "label_duration")},
                   "pred_switch": sw, "pred_duration": dur, "parsed": ok,
                   "stop_reason": stop, "raw": text}
            async with lock:
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    await asyncio.gather(*(one(s) for s in todo))
    if failed:
        print(f"{pair}: {failed} samples failed -- rerun the same command to retry them")


def cmd_run(a):
    os.makedirs(PRED_DIR, exist_ok=True)

    async def main():
        call = make_caller(a)               # one client, one event loop for all pairs
        for pair in a.pairs:
            await run_pair(call, a, pair)
    try:
        asyncio.run(main())
    except Fatal as e:
        raise SystemExit(f"non-retryable API error, stopping: {e.__cause__}")


# ---------------------------------------------------------------- step 3: metrics
def cmd_eval(a):
    from sklearn.metrics import f1_score, precision_score, recall_score
    all_y, all_p, res = [], [], {}
    for pair in a.pairs:
        recs = read_jsonl(pred_path(pair, a.model))
        missing = len(read_jsonl(sample_path(pair))) - len(recs)
        y = [r["label_switch"] for r in recs]; p = [r["pred_switch"] for r in recs]
        all_y += y; all_p += p
        tp = [r for r in recs if r["label_switch"] and r["pred_switch"]]
        dur_f1 = (f1_score([r["label_duration"] for r in tp],
                           [r["pred_duration"] or "none" for r in tp],
                           labels=["1-2", "3-6", "7+"], average="macro", zero_division=0)
                  if tp else float("nan"))
        res[pair] = (f1_score(y, p, zero_division=0), precision_score(y, p, zero_division=0),
                     recall_score(y, p, zero_division=0), len(recs), sum(y),
                     sum(not r["parsed"] for r in recs), missing, dur_f1)
    print(f"\nModel: {a.model}")
    print(f"{'pair':6}{'F1':>8}{'P':>8}{'R':>8}{'n':>7}{'pos':>6}{'bad':>5}{'miss':>6}{'durF1':>8}")
    for pair, (f, pr, rc, n, pos, bad, miss, d) in res.items():
        print(f"{pair:6}{f:8.2%}{pr:8.2%}{rc:8.2%}{n:7}{pos:6}{bad:5}{miss:6}{d:8.2%}")
    print(f"{'all':6}{f1_score(all_y, all_p, zero_division=0):8.2%}   (pooled over all pairs)")
    print("durF1 = duration macro-F1 on correctly predicted switches; bad = unparseable replies "
          "(scored as no-switch); miss = samples with no prediction yet (not scored)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample", add_help=False)    # arguments are parsed by cs_sample.py
    for name in ("run", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--model", default="claude-opus-5-5")
        p.add_argument("--pairs", nargs="+", default=list(PAIRS), choices=list(PAIRS))
        if name == "run":
            p.add_argument("--provider", choices=["anthropic", "openai", "claude-cli"], default="anthropic",
                           help="'openai' = any OpenAI-compatible chat completions API; "
                                "'claude-cli' = Claude Code CLI on your Claude subscription (no API key)")
            p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint (provider=openai)")
            p.add_argument("--api-key-env", default="OPENAI_API_KEY",
                           help="env var holding the key (provider=openai)")
            p.add_argument("--max-tokens", type=int, default=16000,
                           help="includes thinking/reasoning tokens on reasoning models")
            p.add_argument("--temperature", type=float, default=None,
                           help="omitted by default; newer Claude models reject it")
            p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None,
                           help="Claude effort level (provider=anthropic / claude-cli)")
            p.add_argument("--concurrency", type=int, default=8)
            p.add_argument("--limit", type=int, default=None,
                           help="only run the first N samples of each pair (pilot runs)")
    a, rest = ap.parse_known_args()
    if a.cmd == "sample":
        import cs_sample
        cs_sample.main(rest)
    else:
        ap.parse_args()                         # reject unknown arguments for run / eval
        {"run": cmd_run, "eval": cmd_eval}[a.cmd](a)
