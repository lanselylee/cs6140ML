# Claude Opus 5.5 — first 4 pairs (400 samples each, effort low, via Claude Code CLI)

```

Model: claude-opus-5-5
pair        F1       P       R      n   pos  bad  miss   durF1
zh      79.89%  93.71%  69.63%    400   214    0   400  67.27%
ar      67.46%  80.28%  58.16%    400   196    0   400  56.89%
es      67.48%  89.43%  54.19%    400   203    0   400  64.74%
hi      71.04%  79.75%  64.04%    400   203    0   400  62.54%
all     71.70%   (pooled over all pairs)
durF1 = duration macro-F1 on correctly predicted switches; bad = unparseable replies (scored as no-switch); miss = samples with no prediction yet (not scored)
```
