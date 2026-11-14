# Round 2 — probability output + threshold (Claude Opus 5.5, effort low, claude-cli)

Pairs: ar zh fr de, all 800 samples each (400 switch / 400 non-switch). Both methods on the same positions.
`prob`: model outputs `p_switch`; threshold chosen by 5-fold cross-validation (F1-optimal on 4 folds, applied to the 5th).

| method | F1 | acc | P | R | AUROC | tokens/call (in + out) | API-equivalent $/call |
|---|---|---|---|---|---|---|---|
| base (yes/no) | 69.8% | **74.9%** | 87.8% | 57.9% | – | 868 + 34 | 0.0076 |
| prob + CV threshold | **70.2%** | 66.8% | 63.7% | 78.1% | 0.772 | 892 + 43 | 0.0080 |

Per-pair F1 (base → prob): ar 67.9 → 67.9, zh 78.1 → 78.7, fr 68.3 → 68.0, de 64.0 → 67.9.
Per-pair AUROC: ar 0.740, zh 0.848, fr 0.761, de 0.720.

Findings
- Threshold tuning only trades precision for recall (R 58 → 78%, P 88 → 64%). Even the oracle
  accuracy-optimal threshold chosen on the test set does not beat base accuracy on any pair.
- Verbalized probabilities are coarse: ~210/800 positions in fr and de get exactly 0.12.
- F1-optimal thresholds on a balanced set drift to "predict everything" (de: 0.06 → R 98%, P 52%).
- Base token counts come from the 1,600 newly run positions (rows 401–800); the first 400 were run before usage logging.
- Conclusion: the limit is Claude's ranking ability (AUROC 0.77), not its decision threshold.

Reproduce:
    python cs_predict.py run  --provider claude-cli --model claude-opus-5-5 --effort low --method prob --pairs ar zh fr de
    python cs_predict.py eval --method prob --pairs ar zh fr de
