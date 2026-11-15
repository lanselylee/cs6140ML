"""
Estimate what base / few-shot would score on the FULL data (every confidently labelled
position, natural ~10% switch rate) from the balanced 800-per-pair runs, without new calls.

  python cs_extrapolate.py

Per (pair, current token is English, label) stratum, the hit / false-alarm rate measured
on the sample is weighted by how many positions that stratum has in the full data.
95% intervals: bootstrap over the sampled rows (500 resamples).
Assumes the model's per-stratum error rates on the full data match the sample.
"""
import sys, json, random, collections
sys.path.insert(0, ".")
import cs_sample as c
PAIRS = ["ar","zh","fr","de"]
# population counts per (pair, cur_is_en, switch) over every confident position
pop = collections.Counter()
for pair in PAIRS:
    texts = c.read_texts("data/SwitchLingua_text", pair)
    lab = c.make_labeller(pair, "data/SwitchLingua_text", texts); seen=set()
    for t in texts:
        if not t or t in seen: continue
        seen.add(t); toks=c.tokenize(t); L=lab(t,toks)
        for i, sw in c.candidates(toks, L, 5):
            pop[(pair, L[i][0]=="en", int(sw))] += 1

def load(pair, suffix):
    S = {(r["conv_id"], r["t"]): r for r in map(json.loads, open(f"samples/samples_{pair}.jsonl"))}
    out = []
    for r in map(json.loads, open(f"preds/preds_{pair}_claude-opus-5-5{suffix}.jsonl")):
        s = S.get((r["conv_id"], r["t"]))
        if s is None or r.get("pred_switch") is None: continue
        out.append((pair, s["cur_lang"]=="en", r["label_switch"], int(r["pred_switch"])))
    return out

def metrics(rows, pairs):
    # stratum-level rates from sample, weighted by population counts
    rate = collections.defaultdict(lambda: [0,0])
    for p, en, y, yh in rows:
        rate[(p,en,y)][0] += yh; rate[(p,en,y)][1] += 1
    TP=FN=FP=TN=0
    for p in pairs:
        for en in (True, False):
            for y in (0,1):
                N = pop[(p,en,y)]
                if not N: continue
                k, n = rate[(p,en,y)]
                r = k/n if n else 0
                if y: TP += N*r; FN += N*(1-r)
                else: FP += N*r; TN += N*(1-r)
    P = TP/(TP+FP); R = TP/(TP+FN); acc=(TP+TN)/(TP+FN+FP+TN)
    return dict(acc=acc, P=P, R=R, F1=2*P*R/(P+R), FPR=FP/(FP+TN), alarms_per_true=(TP+FP)/(TP+FN))

def boot(rows, pairs, B=500):
    rng = random.Random(0); by = collections.defaultdict(list)
    for r in rows: by[r[0]].append(r)
    res = collections.defaultdict(list)
    for _ in range(B):
        s = [x for p in pairs for x in rng.choices(by[p], k=len(by[p]))]
        for k,v in metrics(s, pairs).items(): res[k].append(v)
    return {k: (sorted(v)[int(.025*B)], sorted(v)[int(.975*B)]) for k,v in res.items()}

for name, suf in [("base",""),("fewshot","_fewshot")]:
    rows = [x for p in PAIRS for x in load(p, suf)]
    # balanced-sample (as reported) metrics
    tp=sum(y and yh for *_,y,yh in rows); fp=sum((not y) and yh for *_,y,yh in rows)
    fn=sum(y and not yh for *_,y,yh in rows); tn=len(rows)-tp-fp-fn
    P=tp/(tp+fp); R=tp/(tp+fn)
    print(f"\n== {name}  n={len(rows)}  [balanced sample] acc {(tp+tn)/len(rows):.1%} P {P:.1%} R {R:.1%} F1 {2*P*R/(P+R):.1%}")
    m = metrics(rows, PAIRS); ci = boot(rows, PAIRS)
    print("   [full population est] " + "  ".join(f"{k} {m[k]:.1%} ({ci[k][0]:.1%}-{ci[k][1]:.1%})" if k!="alarms_per_true" else f"{k} {m[k]:.2f}" for k in m))
    for p in PAIRS:
        mp = metrics([r for r in rows if r[0]==p], [p])
        print(f"   {p}: acc {mp['acc']:.1%} P {mp['P']:.1%} R {mp['R']:.1%} F1 {mp['F1']:.1%}")
tot = sum(pop.values()); sw = sum(v for k,v in pop.items() if k[2])
print(f"\npopulation positions {tot}, switch rate {sw/tot:.1%}")
for p in PAIRS:
    a=sum(v for k,v in pop.items() if k[0]==p); s=sum(v for k,v in pop.items() if k[0]==p and k[2])
    print(p, a, f"{s/a:.1%}")
