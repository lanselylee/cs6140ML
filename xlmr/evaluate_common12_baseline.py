"""Frozen-checkpoint baseline evaluation: no fitting, tuning, or test selection."""
import os
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import argparse,csv,gzip,json,time
from functools import partial
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import CausalMultitask,local_snapshot
from word_data import WordWindows,collate_word,csv_rows,sha

HERE=Path(__file__).resolve().parent
RUN=HERE/'runs/xlmr_common12_word_v4'
DATA=HERE.parent/'outputs/Shared_Phase1_Common12_v4'
OUT=HERE.parent/'outputs/XLMR_12pair_Baseline_Results'

def main():
    global RUN,OUT
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,default=RUN);ap.add_argument('--output',type=Path,default=OUT);ap.add_argument('--language-id',action='store_true')
    args=ap.parse_args();RUN=args.run;OUT=args.output
    OUT.mkdir(exist_ok=True)
    if (OUT/'inference_manifest.json').exists():raise RuntimeError('Frozen evaluation already exists; do not overwrite')
    events=[json.loads(l) for l in (RUN/'train.jsonl').open() if l.strip()]
    assert events[-1]['status']=='training_complete'
    best=max((e for e in events if 'validation' in e),key=lambda e:e['selection_score'])
    checkpoint=torch.load(RUN/'best.pt',map_location='cpu',weights_only=False)
    assert checkpoint['step']==best['step']
    assert checkpoint['manifest_hash']==sha(DATA/'manifest.json')
    assert not checkpoint['args']['resume']
    duration_weight=checkpoint['args'].get('duration_weight',1.0)
    has_lid='language_embedding.weight' in checkpoint['model']
    assert has_lid==args.language_id,'Checkpoint/model feature mode mismatch'
    if args.language_id:
        from lid_model import LIDCausalMultitask
        from lid_features import LIDWindows
        model=LIDCausalMultitask(checkpointing=False)
        dataset_class=LIDWindows
        from lid_features import collate_lid
        collate_function=collate_lid
    else:
        model=CausalMultitask(checkpointing=False)
        dataset_class=WordWindows;collate_function=collate_word
    model.load_state_dict(checkpoint['model']);del checkpoint
    model=model.cuda().eval()
    torch.backends.cuda.matmul.allow_tf32=True
    tokenizer=AutoTokenizer.from_pretrained(local_snapshot(),token=False,local_files_only=True)
    col=partial(collate_function,pad=tokenizer.pad_token_id)
    # Declare frozen settings BEFORE reading any test predictions.
    frozen=dict(model='FacebookAI/xlm-roberta-base',checkpoint_step=best['step'],checkpoint_epoch=best['epoch'],
        selection='Mean of validation switch F1 and duration macro-F1',threshold=.5,
        threshold_source='Fixed baseline setting, no threshold search',
        data_manifest_sha256=sha(DATA/'manifest.json'),checkpoint_sha256=sha(RUN/'best.pt'),
        train_scope='12 pairs only; fresh pretrained backbone, 3 epochs',
        inference_batch_size=4,precision='CUDA FP16 autocast; float32 softmax',
        lambda_switch=1.0,lambda_duration=duration_weight,
        language_id=args.language_id,
        source_code_sha256=sha(Path(__file__)))
    (OUT/'frozen_evaluation_settings.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    results={}
    for split in ['val','test']:
        start=time.time();ds=dataset_class(DATA,split)
        loader=DataLoader(ds,batch_size=4,collate_fn=col,num_workers=0,pin_memory=True)
        path=OUT/f'{split}_predictions.csv.gz'
        n=0
        with gzip.open(path,'wt',encoding='utf-8',newline='',compresslevel=1) as f:
            writer=csv.DictWriter(f,fieldnames=['id','word_idx','pair','cs_type','sw_true','dur_true','sw_prob','sw_pred','dur_pred'])
            writer.writeheader()
            with torch.inference_mode():
                for i,batch in enumerate(loader):
                    ids=batch['input_ids'].cuda(non_blocking=True);mask=batch['attention_mask'].cuda(non_blocking=True)
                    with torch.autocast('cuda',dtype=torch.float16):
                        sw,dur=model(ids,mask,batch['lid_ids'].cuda(non_blocking=True)) if args.language_id else model(ids,mask)
                    rows=list(csv_rows(batch,sw.float().softmax(-1)[...,1],dur.argmax(-1)))
                    writer.writerows(rows);n+=len(rows)
                    if (i+1)%500==0:print(json.dumps(dict(split=split,percent=100*(i+1)/len(loader),positions=n)),flush=True)
        expected=json.loads((DATA/'delivery_checks.json').read_text(encoding='utf-8'))['results'][split]['models']['xlmr']['scored_positions']
        assert n==expected,(split,n,expected)
        results[split]=dict(instances=len(ds.types),windows=len(ds),scored_positions=n,seconds=time.time()-start,
            predictions_sha256=sha(path))
        del ds,loader
        print('SPLIT_COMPLETE '+split+' '+str(results[split]),flush=True)
    (OUT/'inference_manifest.json').write_text(json.dumps(dict(settings=frozen,results=results),indent=2),encoding='utf-8')
    print('FROZEN_BASELINE_EVALUATION_COMPLETE',flush=True)

if __name__=='__main__':main()
