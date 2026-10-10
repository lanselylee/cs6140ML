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
(and if so, for how long). Data: SwitchLingua zh-en / ja-en. Everyone uses the same script.

```bash
# 1. sample once (put Chinese_eng.csv / Japanese_eng.csv in the repo root), then commit samples_*.jsonl
python cs_predict.py sample --zh Chinese_eng.csv --ja Japanese_eng.csv --n 1000

# 2. run one model (results go to preds/, safe to stop and rerun - it resumes)
python cs_predict.py run --provider anthropic --model claude-opus-5-5 --effort low
python cs_predict.py run --provider openai --model gpt-4o --temperature 0
python cs_predict.py run --provider openai --model deepseek-chat \
    --base-url https://api.deepseek.com --api-key-env DEEPSEEK_API_KEY --temperature 0

# 3. score
python cs_predict.py eval --model claude-opus-5-5
```

`--provider openai` works with any OpenAI-compatible endpoint (OpenAI, DeepSeek, Qwen, Gemini, OpenRouter...).
Never commit API keys - pass them via environment variables.
