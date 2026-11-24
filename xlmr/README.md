# XLM-R Code-Switch Prediction

This directory contains our XLM-R-base implementation and experiments for the CS6140 project.

Given a bilingual prefix, the model predicts:
1. Whether the next word switches language.
2. The length of the upcoming language segment: 1–2, 3–6, or 7+ words.

## Model and Data

- Causal attention restricts each prediction to the visible prefix.
- A shared backbone supports switch and duration classification heads.
- Experiments use the shared 12-language-pair v4 dataset with an approximately 80/10/10 train/validation/test split.
- Supervision and evaluation use shared word-level positions. Task labels are placed on the final subword of each word.
- Duration loss is computed only at true switch positions.

## Files

| File | Purpose |
| --- | --- |
| model.py | Causal XLM-R backbone, dual heads, and multitask loss |
| word_data.py | Dataset loading, batching, and integrity checks |
| train.py | Multitask training with configurable duration loss weight |
| evaluate_common12_baseline.py | Prediction export |
| score_common12_baseline.py | Shared evaluation and additional metrics |
| run_short_experiments.py | Current-word baseline and threshold experiments |
| lid_features.py | Causal estimation of prefix language IDs |
| lid_model.py | XLM-R with language-ID embeddings |
| train_lid.py | Language-ID experiment training |
| single_task.py | Switch-only and duration-only controls |
| check_word_phase2.py | Baseline implementation checks |
| check_lid.py | Language-ID implementation checks |
| check_single_task.py | Single-task implementation checks |

## Experimental Setup

Models are trained from pretrained XLM-R-base weights for three epochs. Checkpoints and decision thresholds are selected using validation data, not test data.

Experiments include loss-weight adjustment, threshold tuning, prefix language-ID features, and single-task controls.

## Reproducibility Notes

These files are snapshots of the experiment code and currently depend on the original project directory structure, validated dataset files, and preflight artifacts. They are not yet a standalone installation package.

Training data, model checkpoints, and full prediction files are not included in this directory. Language labels are automatically generated silver references.
