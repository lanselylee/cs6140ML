# Claude Opus 5.5 — 12 pairs (400 samples each, effort low, via Claude Code CLI)

Run: `python cs_predict.py run --provider claude-cli --model claude-opus-5-5 --effort low --limit 400`

```

Model: claude-opus-5-5
pair        F1       P       R      n   pos  bad  miss   durF1
zh      79.89%  93.71%  69.63%    400   214    0   400  67.27%
ja      71.92%  89.06%  60.32%    400   189    0   400  61.98%
ko      76.88%  93.88%  65.09%    400   212    0   400  67.68%
ar      67.46%  80.28%  58.16%    400   196    0   400  56.89%
ru      62.96%  82.93%  50.75%    400   201    0   400  60.95%
fr      71.07%  91.13%  58.25%    400   194    0   400  71.06%
de      63.48%  88.57%  49.47%    400   188    0   400  65.27%
es      67.48%  89.43%  54.19%    400   203    0   400  64.74%
it      64.92%  88.39%  51.30%    400   193    0   400  68.40%
ms      64.43%  76.16%  55.83%    400   206    0   400  64.58%
tl      70.99%  79.25%  64.29%    400   196    0   400  51.81%
hi      71.04%  79.75%  64.04%    400   203    0   400  62.54%
all     69.61%   (pooled over all pairs)
durF1 = duration macro-F1 on correctly predicted switches; bad = unparseable replies (scored as no-switch); miss = samples with no prediction yet (not scored)
```

`miss 400` = the second half of each 800-sample file, not run yet (not scored).
