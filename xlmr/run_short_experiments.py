"""Train-only current-word naive baseline + validation-only threshold choices."""
import csv,gzip,json,sys,unicodedata
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve,f1_score,precision_score,recall_score,roc_auc_score,average_precision_score
HERE=Path(__file__).resolve().parent
DATA=HERE.parent/'outputs/Shared_Phase1_Common12_v4'
BASE=HERE.parent/'outputs/XLMR_12pair_Baseline_Results'
OUT=HERE.parent/'outputs/Update12_Experiments'
sys.path.insert(0,str(HERE.parent/'shared_phase1/vendor'))
import cs_eval

def best_threshold(y,s):
    p,r,t=precision_recall_curve(y,s)
    f=2*p[:-1]*r[:-1]/np.maximum(p[:-1]+r[:-1],1e-15)
    optimum=f.max()
    # Deterministic tie-break: greatest threshold among exact maxima.
    return float(t[np.flatnonzero(np.isclose(f,optimum,rtol=0,atol=1e-12))[-1]])
def key(word):return unicodedata.normalize('NFKC',word).casefold().strip()
def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def evaluate(df,name):
    result=cs_eval.evaluate(df,model_name=name,out_dir=None,verbose=False)
    s=df[df.sw_true==1]
    result.update(switch_precision=precision_score(df.sw_true,df.sw_pred,zero_division=0),
        switch_recall=recall_score(df.sw_true,df.sw_pred,zero_division=0),
        duration_macro_f1=f1_score(s.dur_true,s.dur_pred,labels=[0,1,2],average='macro',zero_division=0),
        positions=len(df),true_switches=int(df.sw_true.sum()),
        auroc=roc_auc_score(df.sw_true,df.sw_prob),ap=average_precision_score(df.sw_true,df.sw_prob))
    return result

def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'short_experiments_complete.json').exists():raise RuntimeError('Short experiment outputs already exist')
    words=defaultdict(lambda:[0,0]);dur_counts=Counter();total=positive=0
    for line in (DATA/'labels/train_labels.jsonl').open(encoding='utf-8'):
        s=json.loads(line)
        for word,y,d in zip(s['words'],s['sw'],s['dur']):
            if y<0:continue
            c=words[key(word)];c[0]+=1;c[1]+=y;total+=1;positive+=y
            if d>=0:dur_counts[d]+=1
    prior=positive/total;strength=10.;majority=dur_counts.most_common(1)[0][0]
    dump(OUT/'naive_model.json',dict(method='Current normalized word only, including attached punctuation; no prefix, pair metadata or language ID',
        smoothing_pseudocount=strength,train_switch_prior=prior,train_positions=total,unique_words=len(words),
        duration_method='Constant train-majority class',duration_class=majority,
        word_counts=dict(words)))
    frames={};oov={}
    for split in ['val','test']:
        rows=[];unseen=0
        for line in (DATA/'labels'/f'{split}_labels.jsonl').open(encoding='utf-8'):
            s=json.loads(line)
            for i,(w,y,d) in enumerate(zip(s['words'],s['sw'],s['dur'])):
                if y<0:continue
                n,p=words.get(key(w),(0,0));unseen+=int(n==0)
                prob=(p+strength*prior)/(n+strength)
                rows.append(dict(id=s['id'],word_idx=i,pair=s['pair'],cs_type=s['cs_type'] or 'unknown',
                    sw_true=y,dur_true=d,sw_prob=prob,sw_pred=int(prob>=.5),dur_pred=majority))
        frames[split]=pd.DataFrame(rows);oov[split]=unseen/len(rows)
    threshold=best_threshold(frames['val'].sw_true,frames['val'].sw_prob)
    dump(OUT/'naive_frozen_settings.json',dict(threshold=threshold,selected_on='validation only',oov_rates=oov,
        smoothing_pseudocount=10.,features='current word including attached punctuation only',duration_class=majority))
    result={}
    for split,df in frames.items():
        df.to_csv(OUT/f'naive_{split}_predictions.csv.gz',index=False,compression={'method':'gzip','compresslevel':1})
        for name,cut in [('CurrentWord_fixed05',.5),('CurrentWord_val_threshold',threshold)]:
            pred=df.copy();pred['sw_pred']=(pred.sw_prob>=cut).astype(int)
            result.setdefault(name,{})[split]=evaluate(pred,name)
    print('NAIVE_BASELINE_COMPLETE',flush=True)
    # XLM-R threshold variants only re-score the previously frozen probabilities.
    val=pd.read_csv(BASE/'val_predictions.csv.gz');test=pd.read_csv(BASE/'test_predictions.csv.gz')
    global_t=best_threshold(val.sw_true,val.sw_prob)
    pair_t={p:best_threshold(g.sw_true,g.sw_prob) for p,g in val.groupby('pair')}
    dump(OUT/'threshold_frozen_settings.json',dict(global_threshold=global_t,per_pair_thresholds=pair_t,
        selection='Validation switch F1 only, tie-break greatest threshold',test_used_for_selection=False,
        fixed_sweep=[.22,.35,.5,.6,.7,.8],model='Original 12-pair equal-lambda baseline'))
    sweep=[]
    for split,df in [('val',val),('test',test)]:
        variants={'XLMR_fixed05':np.full(len(df),.5),'XLMR_global_val_threshold':np.full(len(df),global_t),
            'XLMR_per_pair_val_threshold':df.pair.map(pair_t).to_numpy()}
        for name,cuts in variants.items():
            p=df.copy();p['sw_pred']=(p.sw_prob.to_numpy()>=cuts).astype(int)
            result.setdefault(name,{})[split]=evaluate(p,name)
        for cut in [.22,.35,.5,.6,.7,.8]:
            y=df.sw_true.to_numpy();pred=df.sw_prob.to_numpy()>=cut
            sweep.append(dict(split=split,threshold=cut,precision=precision_score(y,pred,zero_division=0),
                recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),
                false_positives=int(((y==0)&pred).sum()),false_negatives=int(((y==1)&~pred).sum())))
    pd.DataFrame(sweep).to_csv(OUT/'threshold_sweep.csv',index=False,encoding='utf-8-sig')
    dump(OUT/'short_experiment_metrics.json',result)
    rows=[]
    for name,splits in result.items():
        for split,r in splits.items():
            rows.append(dict(method=name,split=split,switch_f1=r['anticipatory_f1'],precision=r['switch_precision'],
                recall=r['switch_recall'],duration_accuracy=r['duration_accuracy'],
                duration_macro_f1=r['duration_macro_f1'],auroc=r['auroc'],ap=r['ap'],
                pair_mean_f1=float(np.mean(list(r['f1_by_pair'].values()))),pair_std_f1=r['sigma_universality']))
    pd.DataFrame(rows).to_csv(OUT/'short_experiment_comparison.csv',index=False,encoding='utf-8-sig')
    dump(OUT/'short_experiments_complete.json',dict(status='complete',threshold_global=global_t,naive_threshold=threshold,
        lambda_training_started=False,language_id_training_started=False))
    print('SHORT_EXPERIMENTS_COMPLETE',flush=True)

if __name__=='__main__':main()
