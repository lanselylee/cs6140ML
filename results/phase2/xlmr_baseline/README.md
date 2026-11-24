# XLM-R baseline (teammate's run)

Copied unchanged from the team's `XLMR_Update1_Update2_Package` (results/baseline + tables).
XLM-R-base causal dual-head, trained on Shared Phase 1 v4 (12 pairs), RTX 3080, 3 epochs,
batch 4 x accumulation 4, lr 2e-5, seed 6130; checkpoint selected at epoch 2 (step 28,920)
by the mean of validation switch F1 and duration macro-F1. See `training_config.json`.

- `test_predictions.csv.gz` — per-word test predictions (cs_eval.py format)
- `baseline_training_sampled_loss.csv` — training loss (sampled last micro-batch every 25 steps)
- `baseline_validation_loss.csv` — validation loss / metrics at each evaluation
- `baseline_per_pair.csv` — per-pair metrics as reported by the teammate

The mBERT run in `../mbert` uses the same hyper-parameters; `streaming/compare_plots.py`
scores both prediction files with the same code (`../figures`).
