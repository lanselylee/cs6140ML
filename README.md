# cs6140ML

CS6140 Machine Learning course project: training a large language model for language prediction (next-token prediction).

## Project Structure

```
cs6140ML/
├── data/          # datasets (raw data is not committed)
├── src/           # model, training and evaluation code
├── notebooks/     # experiments and analysis
├── checkpoints/   # saved model weights (not committed)
└── requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Code-switching prediction benchmark (`cs_predict.py`)

Asks an LLM, given only a text prefix, whether the next token switches language
(and if so, for how long). Data: SwitchLingua, 12 X-English pairs:

| code | pair | code | pair | code | pair |
|---|---|---|---|---|---|
| zh | Chinese | ar | Arabic | es | Spanish |
| ja | Japanese | ru | Russian | it | Italian |
| ko | Korean | fr | French | ms | Malay |
| hi | Hindi (Devanagari + romanized Hinglish) | de | German | tl | Filipino |

The shared samples are already committed in `samples/` (400 per pair: 200 switches + 200
non-switches, one position per conversation). **Everyone runs on these same files**, so
results from Claude / GPT / Gemini are directly comparable.

```bash
pip install -r requirements.txt

# 1. run one model on all 12 pairs (results go to preds/, safe to stop and rerun - it resumes)
python cs_predict.py run --provider anthropic --model claude-opus-5-5 --effort low
python cs_predict.py run --provider openai --model gpt-4o --temperature 0
python cs_predict.py run --provider openai --model gemini-2.5-flash \
    --base-url https://generativelanguage.googleapis.com/v1beta/openai/ --api-key-env GEMINI_API_KEY
#    only some pairs: add  --pairs zh hi fr

# 2. score
python cs_predict.py eval --model claude-opus-5-5
```

`--provider openai` works with any OpenAI-compatible endpoint (OpenAI, DeepSeek, Qwen, Gemini, OpenRouter...).
Never commit API keys - pass them via environment variables.

### Re-creating the samples (normally not needed)

Download SwitchLingua_text from Hugging Face into `data/SwitchLingua_text/`, then
`python cs_sample.py` (same seed -> identical files). How tokens and language labels are
built for each pair is documented at the top of `cs_sample.py`. SwitchLingua has no
word-level gold labels, so labels are automatic: by script for zh/ja/ko/ar/ru, by word
frequency + lingua for the Latin-script pairs, and by an in-domain word list for Hinglish.
Only confidently labelled tokens are used as sample positions.
