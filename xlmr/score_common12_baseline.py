"""Score immutable exported predictions with the team's script plus diagnostics."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
    roc_auc_score,average_precision_score,confusion_matrix,precision_recall_fscore_support)

HERE=Path(__file__).resolve().parent
OUT=HERE.parent/'outputs/XLMR_12pair_Baseline_Results'
DATA=HERE.parent/'outputs/Shared_Phase1_Common12_v4'
RUN=HERE/'runs/xlmr_common12_word_v4'
sys.path.insert(0,str(HERE.parent/'shared_phase1/vendor'))
import cs_eval

def score(df):
    y=df.sw_true;p=df.sw_pred;prob=df.sw_prob;s=df[df.sw_true==1]
    return dict(positions=len(df),true_switches=int(y.sum()),switch_accuracy=accuracy_score(y,p),
        switch_precision=precision_score(y,p,zero_division=0),switch_recall=recall_score(y,p,zero_division=0),
        switch_f1=f1_score(y,p,zero_division=0),auroc=roc_auc_score(y,prob) if y.nunique()==2 else None,
        average_precision=average_precision_score(y,prob) if y.sum() else None,
        duration_accuracy=accuracy_score(s.dur_true,s.dur_pred),
        duration_macro_f1=f1_score(s.dur_true,s.dur_pred,labels=[0,1,2],average='macro',zero_division=0),
        switch_confusion=confusion_matrix(y,p,labels=[0,1]).tolist(),
        duration_confusion=confusion_matrix(s.dur_true,s.dur_pred,labels=[0,1,2]).tolist())

def main():
    global RUN,OUT
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=RUN);ap.add_argument('--output',type=Path,default=OUT)
    args=ap.parse_args();RUN=args.run;OUT=args.output
    infer=json.loads((OUT/'inference_manifest.json').read_text(encoding='utf-8'))
    summaries={};all_pair=[];all_type=[];all_class=[]
    for split in ['val','test']:
        df=pd.read_csv(OUT/f'{split}_predictions.csv.gz',keep_default_na=False)
        assert not df.duplicated(['id','word_idx']).any()
        assert ((df.sw_prob>=.5).astype(int)==df.sw_pred).all()
        expected={}
        for line in (DATA/'labels'/f'{split}_labels.jsonl').open(encoding='utf-8'):
            s=json.loads(line)
            for w,(sw,dur) in enumerate(zip(s['sw'],s['dur'])):
                if sw!=-1:expected[s['id'],w]=(sw,dur,s['pair'],s['cs_type'] or 'unknown')
        assert len(expected)==len(df)
        for r in df.itertuples(index=False):
            assert expected.pop((r.id,r.word_idx))==(r.sw_true,r.dur_true,r.pair,r.cs_type)
        assert not expected
        team=cs_eval.evaluate(df,model_name='FacebookAI/xlm-roberta-base',out_dir=str(OUT/'official'/split),verbose=False)
        result=score(df);result.update(official=team,split=split)
        result['type_macro_accuracy']=team['macro_average_accuracy']
        for pair,g in df.groupby('pair'):
            values=score(g)
            all_pair.append(dict(split=split,pair=pair,**{k:v for k,v in values.items() if not k.endswith('confusion')}))
        for kind,g in df.groupby('cs_type'):
            values=score(g)
            all_type.append(dict(split=split,cs_type=kind,**{k:v for k,v in values.items() if not k.endswith('confusion')}))
        s=df[df.sw_true==1]
        prec,rec,f1,support=precision_recall_fscore_support(s.dur_true,s.dur_pred,labels=[0,1,2],zero_division=0)
        for i,label in enumerate(['1–2 words','3–6 words','7+ words']):
            all_class.append(dict(split=split,duration_class=i,bin=label,support=int(support[i]),
                precision=float(prec[i]),recall=float(rec[i]),f1=float(f1[i])))
        pair_rows=[r for r in all_pair if r['split']==split]
        result['pair_mean_f1']=float(np.mean([r['switch_f1'] for r in pair_rows]))
        result['pair_std_f1']=float(np.std([r['switch_f1'] for r in pair_rows]))
        result['best_switch_pair']=max(pair_rows,key=lambda r:r['switch_f1'])['pair']
        result['worst_switch_pair']=min(pair_rows,key=lambda r:r['switch_f1'])['pair']
        assert abs(result['switch_f1']-team['anticipatory_f1'])<1e-12
        assert abs(result['duration_accuracy']-team['duration_accuracy'])<1e-12
        assert abs(result['pair_std_f1']-team['sigma_universality'])<1e-12
        summaries[split]=result
        print(json.dumps({k:v for k,v in result.items() if k not in ['official','switch_confusion','duration_confusion']},indent=2),flush=True)
    for name,rows in [('per_pair.csv',all_pair),('per_type.csv',all_type),('duration_classes.csv',all_class)]:
        pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
    events=[json.loads(l) for l in (RUN/'train.jsonl').open() if l.strip()]
    validation_events=[r for r in events if 'validation' in r]
    best=max(validation_events,key=lambda r:r['selection_score'])
    epoch_rows=[];val_rows=[];train_rows=[]
    for r in events:
        if 'mean_loss' in r:train_rows.append({k:r[k] for k in ['epoch','step','mean_loss','switch_loss','duration_loss','lr','elapsed_epoch_seconds']})
        if 'validation' in r:
            v=r['validation'];entry=dict(epoch=r['epoch'],step=r['step'],switch_f1=v['official']['anticipatory_f1'],
                duration_accuracy=v['official']['duration_accuracy'],duration_macro_f1=v['pooled']['duration']['macro_f1'],
                switch_loss=v['loss']['switch'],duration_loss=v['loss']['duration'],selection_score=r['selection_score'])
            val_rows.append(entry)
            if 'batch' not in r:epoch_rows.append(entry)
    for name,rows in [('epoch_results.csv',epoch_rows),('validation_history.csv',val_rows),('training_history.csv',train_rows)]:
        pd.DataFrame(rows).to_csv(OUT/name,index=False,encoding='utf-8-sig')
    logval=best['validation']['official']
    summaries['validation_reproduction']=dict(logged_f1=logval['anticipatory_f1'],exported_f1=summaries['val']['switch_f1'],
        difference=summaries['val']['switch_f1']-logval['anticipatory_f1'])
    assert abs(summaries['validation_reproduction']['difference'])<.001,'Unexpected checkpoint validation disagreement'
    config=json.loads((RUN/'config.json').read_text())
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    summaries['experiment']=dict(model='FacebookAI/xlm-roberta-base',architecture='Causal shared backbone + switch/duration linear heads',
        train_pairs=12,epochs=3,selected_epoch=best['epoch'],selected_step=best['step'],threshold=.5,
        lambda_switch=1.,lambda_duration=config.get('duration_weight',1.),seed=config['seed'],microbatch=config['batch_size'],gradient_accumulation=config['accumulation'],
        learning_rate=config['lr'],training_hours=((RUN/'train.jsonl').stat().st_mtime-(RUN/'config.json').stat().st_mtime)/3600,
        data_version=manifest['version'],train_from='Original pretrained backbone; no old trained checkpoint',
        checkpoint_selection='Mean of validation switch F1 and duration macro-F1',
        reference_labels='Automatic silver labels; human audit deferred',
        ap_definition='average_precision_score (AP), not trapezoidal PR area',
        duration_condition='All true switches, including switch false negatives',
        type_grouping='Original record-level cs_type; all 3 known types included',
        prediction_integrity='Exact key set and labels checked against shared v4 gold; both splits passed')
    (OUT/'metrics.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SCORING_COMPLETE_OFFICIAL_PARITY_AND_GOLD_ALIGNMENT_PASSED',flush=True)

if __name__=='__main__':main()
