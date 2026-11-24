# Causal dual-head models (spec Section III)

Two multilingual encoders, **XLM-R** (`FacebookAI/xlm-roberta-base`) and **mBERT**
(`google-bert/bert-base-multilingual-cased`), trained with the same code and the same
hyper-parameters on the shared Phase 1 data (`Shared_Phase1_Common12_v4`, 12 X–English pairs).

| File | What it does |
|---|---|
| `model.py` | `CausalDualHead`: encoder with `config.is_decoder=True` (look-ahead mask: sub-word t attends only to ≤ t) + switch head (2 classes) + duration head (3 classes). `multitask_loss` = λ_sw·CE(switch) + λ_dur·CE(duration), both `ignore_index=-1`. |
| `tests/test_causal.py` | Leakage test: changing any token after t leaves outputs at ≤ t unchanged (XLM-R and mBERT); a bidirectional control must fail the same check. |
| `encode_model.py` | Encodes the shared word-level labels for a new tokenizer (used for mBERT). Same fragment encoding as Phase 1 `prepare.py`; window boundaries copied from `encoded/xlmr`. `--model xlmr --verify` reproduces the shipped XLM-R files exactly (0 mismatching windows on all splits). |
| `data.py` | Loads windows, pads, length-bucketed batches. |
| `train.py` | Training loop, hyper-parameters matched to the teammate's XLM-R baseline (micro-batch 4 × accumulation 4, lr 2e-5, 3% warm-up, seed 6130, threshold 0.5, select by mean of val switch F1 and duration macro-F1). Writes `train_log.csv` (total / switch / duration loss), `val_log.csv`, `val_pairs.jsonl`, `best.pt`, `meta.json`. |
| `predict.py` | One row per word in `cs_eval.py` format, scored with the group's unmodified `cs_eval.py`. Also reports the 2-type macro accuracy (inter + intra only). |
| `demo.py` | Streaming simulation: reads text word by word and prints the next-word switch probability and the anticipated duration bin, running the model on the prefix only. Uses Phase 1's word segmentation (jieba / fugashi), verified to reproduce the training encoding on all 28,876 test instances. `python streaming/demo.py --run RUN --text "..."` (`--delay`, `--json`, stdin). |
| `run_all.sh` | Trains one model (default mBERT), then predicts and scores val and test. |

## Data facts (checked 2026-11-22)
- Scored positions are identical for both models: train 3,850,758 / val 478,427 / test 481,277.
- Windows are identical for both models except one long train instance (`ms:3407:s0`), re-windowed for mBERT (3 → 4 windows) because mBERT needs more sub-words; every word is still scored once.
- Sub-words per word: XLM-R 1.56, mBERT 1.61. Words containing `[UNK]`: XLM-R 0.002%, mBERT 0.43% (highest in French–English).

## Decisions
- Model selection (best checkpoint) uses the validation split only; the switch threshold is fixed at 0.5; test is predicted once.
- Macro-average accuracy: the spec names inter- and intra-sentential switches; the data also has
  "Extra-sentential / Tag switching". We report the 2-type macro as the main number and
  `cs_eval.py`'s 3-type macro as a supplement.

## Run
```bash
python streaming/tests/test_causal.py
python streaming/encode_model.py --data_root DATA --model mbert        # once
bash streaming/run_all.sh DATA runs mbert                              # one GPU
```
On Kaggle: `kaggle/train_mbert.ipynb` (one GPU, Internet on; clones this repo for the code).
