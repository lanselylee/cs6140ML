"""Task-isolated controls. Same pretrained initialization/data/budget as multitask."""
from pathlib import Path
import argparse,csv,gzip,json,math,os,random,time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader,Subset
from functools import partial
from transformers import AutoTokenizer
from model import CausalMultitask,local_snapshot
from word_data import CURRENT_DATA,WordWindows,collate_word,sha
from train import statistics,move
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,accuracy_score

HERE=Path(__file__).resolve().parent

class SingleTaskModel(CausalMultitask):
    def __init__(self,task,**kw):
        super().__init__(**kw);assert task in ['switch','duration'];self.task=task
        inactive=self.duration_head if task=='switch' else self.switch_head
        inactive.requires_grad_(False)
    def forward(self,input_ids,attention_mask):
        h=self.backbone(input_ids=input_ids,attention_mask=attention_mask,use_cache=False).last_hidden_state
        h=self.dropout(h)
        return (self.switch_head if self.task=='switch' else self.duration_head)(h)

def task_loss(logits,labels):
    keep=labels>=0
    return torch.nn.functional.cross_entropy(logits[keep].float(),labels[keep]) if keep.any() else logits.sum()*0.
def target(batch,task):return batch['switch_labels' if task=='switch' else 'duration_labels']
def selection(report,task):return report['pooled']['class_f1'][1] if task=='switch' else report['pooled']['macro_f1']

@torch.no_grad()
def evaluate(model,loader,task,path=None):
    model.eval();k=2 if task=='switch' else 3;counts={};loss_sum=0.;n=0;keys=set();probabilities=[];truth=[];predictions=[]
    f=gzip.open(path,'wt',encoding='utf-8',newline='',compresslevel=1) if path else None
    writer=None
    try:
        for cpu in loader:
            b=move(cpu)
            with torch.autocast('cuda',dtype=torch.float16):logits=model(b['input_ids'],b['attention_mask'])
            labels=target(b,task);mask=labels>=0;prob=logits.float().softmax(-1)
            pred=(prob[...,1]>=.5).long() if task=='switch' else prob.argmax(-1)
            if mask.any():
                loss_sum+=float(torch.nn.functional.cross_entropy(logits[mask].float(),labels[mask],reduction='sum'));n+=int(mask.sum())
            for i,pair in enumerate(cpu['pair']):
                y=labels[i][mask[i]];p=pred[i][mask[i]]
                c=torch.bincount(k*y+p,minlength=k*k).reshape(k,k).cpu()
                for group in ['pooled',pair,'type:'+cpu['cs_type'][i]]:
                    counts.setdefault(group,torch.zeros(k,k,dtype=torch.long));counts[group]+=c
            if writer is not None or path:
                positions=mask.cpu().nonzero().tolist();p_cpu=prob.cpu();pred_cpu=pred.cpu()
                for i,j in positions:
                    word=int(cpu['word_ids'][i,j]);key=(cpu['id'][i],word)
                    assert key not in keys;keys.add(key)
                    row=dict(id=key[0],word_idx=word,pair=cpu['pair'][i],cs_type=cpu['cs_type'][i],
                        sw_true=int(cpu['switch_labels'][i,j]),dur_true=int(cpu['duration_labels'][i,j]),
                        prediction=int(pred_cpu[i,j]))
                    row.update({'prob_'+str(c):float(p_cpu[i,j,c]) for c in range(k)})
                    if writer is None:writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader()
                    writer.writerow(row)
                    truth.append(row['sw_true'] if task=='switch' else row['dur_true']);predictions.append(row['prediction'])
                    if task=='switch':probabilities.append(row['prob_1'])
    finally:
        if f:f.close()
    report={g:statistics(c) for g,c in counts.items()}
    report['loss']=loss_sum/max(n,1);report['task']=task;report['eligible_positions']=n
    if truth:
        assert abs(report['pooled']['accuracy']-accuracy_score(truth,predictions))<1e-12
        assert abs(report['pooled']['macro_f1']-f1_score(truth,predictions,labels=list(range(k)),average='macro',zero_division=0))<1e-12
        report['independent_metric_parity']='passed'
    if task=='switch':
        pairs=[g for g in counts if g!='pooled' and not g.startswith('type:')]
        values=[report[g]['class_f1'][1] for g in pairs]
        report.update(pair_mean_f1=float(np.mean(values)),pair_std_f1=float(np.std(values)))
        if truth:report.update(auroc=roc_auc_score(truth,probabilities),ap=average_precision_score(truth,probabilities))
    return report

def train(args):
    assert torch.cuda.is_available()
    gate=json.loads((HERE/'phase2_single_task/preflight_pass.json').read_text())
    assert gate['status']=='passed' and gate['manifest_sha256']==sha(args.data/'manifest.json')
    assert args.task in gate['checked_tasks']
    for name,digest in gate['code_sha256'].items():assert sha(HERE/name)==digest,('Code changed',name)
    random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    model=SingleTaskModel(args.task).cuda()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.lr,weight_decay=.01,foreach=False)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    tok=AutoTokenizer.from_pretrained(local_snapshot(),token=False,local_files_only=True)
    col=partial(collate_word,pad=tok.pad_token_id)
    tr=WordWindows(args.data,'train');val=WordWindows(args.data,'val')
    args.output.mkdir(exist_ok=True)
    last=args.output/'last.pt';manifest=sha(args.data/'manifest.json')
    if last.exists() and not args.resume:raise RuntimeError('Existing run; explicit resume required')
    batches=math.ceil(len(tr)/4);total=args.epochs*math.ceil(batches/4)
    start_epoch=start_batch=step=0;best=-1.
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(model='FacebookAI/xlm-roberta-base',selection='validation switch F1' if args.task=='switch' else 'validation duration macro-F1',
        other_head='frozen, excluded from optimizer and forward/loss',manifest_sha256=manifest,
        batch_size=4,accumulation=4,from_pretrained=True,test_used_for_training_or_selection=False)
    if args.resume:
        state=torch.load(last,map_location='cpu',weights_only=False)
        assert state['manifest_hash']==manifest and state['task']==args.task
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        start_epoch,start_batch,step,best=[state[k] for k in ['next_epoch','next_batch','step','best']]
        torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);random.setstate(state['python_rng']);del state
    (args.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    def save(epoch,batch,path):
        temp=path.with_suffix('.tmp')
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),task=args.task,
            manifest_hash=manifest,next_epoch=epoch,next_batch=batch,step=step,best=best,args=config,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate()),temp)
        os.replace(temp,path)
    log=(args.output/'train.jsonl').open('a',encoding='utf-8')
    def emit(row):log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
    devloader=DataLoader(val,batch_size=4,collate_fn=col,num_workers=0,pin_memory=True)
    def validate(epoch,batch):
        nonlocal best
        result=evaluate(model,devloader,args.task);score=selection(result,args.task)
        emit(dict(epoch=epoch+1,step=step,validation=result,selection_score=score))
        if score>best:best=score;save(epoch,batch,args.output/'best.pt')
        save(epoch,batch,last);model.train()
    try:
        for epoch in range(start_epoch,args.epochs):
            order=torch.randperm(len(tr),generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
            offset=start_batch*4 if epoch==start_epoch else 0
            loader=DataLoader(Subset(tr,order[offset:]),batch_size=4,collate_fn=col,num_workers=0,pin_memory=True,
                generator=torch.Generator().manual_seed(args.seed+epoch+10000))
            model.train();opt.zero_grad(set_to_none=True);running=0.;pending=0;group_targets=0;timer=time.time()
            for local,b in enumerate(loader):
                i=local+offset//4;b=move(b);labels=target(b,args.task)
                assert not ((b['duration_labels']>=0)&(b['switch_labels']!=1)).any()
                divisor=min(4,batches-(i//4)*4)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(b['input_ids'],b['attention_mask']);loss=task_loss(logits,labels)
                assert torch.isfinite(loss)
                scaler.scale(loss/divisor).backward();running+=float(loss.detach());pending+=1;group_targets+=int((labels>=0).sum())
                if (i+1)%4==0 or i+1==batches:
                    warmup=max(1,int(total*.03));rate=(step+1)/warmup if step<warmup else max(0.,(total-step)/max(1,total-warmup))
                    for group in opt.param_groups:group['lr']=args.lr*rate
                    skipped=False
                    if group_targets:
                        scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                        old=scaler.get_scale();scaler.step(opt);scaler.update();skipped=scaler.get_scale()<old
                    opt.zero_grad(set_to_none=True);step+=1
                    if step%25==0:
                        emit(dict(epoch=epoch+1,step=step,total_steps=total,task=args.task,mean_loss=running/max(1,pending),
                            scaler_skipped_update=skipped,empty_target_group=group_targets==0,elapsed_epoch_seconds=time.time()-timer))
                        running=0.;pending=0
                    group_targets=0
                    if step%500==0:save(epoch,i+1,last)
                    if step%10000==0 and i+1<batches:validate(epoch,i+1)
            validate(epoch,batches)
            save(epoch+1,0,last);start_batch=0
        emit(dict(status='training_complete',task=args.task,best_validation_score=best,test_evaluated=False))
    finally:log.close()

def infer(args):
    args.output.mkdir(exist_ok=True)
    if (args.output/'metrics.json').exists():raise RuntimeError('Evaluation already exists')
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    assert checkpoint['manifest_hash']==sha(args.data/'manifest.json')
    model=SingleTaskModel(args.task,checkpointing=False)
    model.load_state_dict(checkpoint['model']);model=model.cuda().eval()
    frozen=dict(task=args.task,checkpoint=str(args.checkpoint),step=checkpoint['step'],
        checkpoint_sha256=sha(args.checkpoint),data_manifest_sha256=sha(args.data/'manifest.json'),threshold=.5,
        selection='validation switch F1' if args.task=='switch' else 'validation duration macro-F1',unused_task='not applicable')
    del checkpoint
    (args.output/'frozen_settings.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    tok=AutoTokenizer.from_pretrained(local_snapshot(),token=False,local_files_only=True);col=partial(collate_word,pad=tok.pad_token_id)
    result={'settings':frozen}
    for split in ['val','test']:
        ds=WordWindows(args.data,split);loader=DataLoader(ds,batch_size=4,collate_fn=col,num_workers=0,pin_memory=True)
        r=evaluate(model,loader,args.task,args.output/f'{split}_predictions.csv.gz')
        checks=json.loads((args.data/'delivery_checks.json').read_text(encoding='utf-8'))
        if args.task=='switch':assert r['eligible_positions']==checks['results'][split]['models']['xlmr']['scored_positions']
        else:
            expected=sum(int(y==1) for line in (args.data/'labels'/f'{split}_labels.jsonl').open(encoding='utf-8') for y in json.loads(line)['sw'])
            assert r['eligible_positions']==expected
        result[split]=r;print('EVALUATED',args.task,split,r['pooled'],flush=True)
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['train','infer']);p.add_argument('--task',choices=['switch','duration'],required=True)
    p.add_argument('--data',type=Path,default=CURRENT_DATA);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path);p.add_argument('--epochs',type=int,default=3);p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--seed',type=int,default=6130);p.add_argument('--resume',action='store_true')
    a=p.parse_args();train(a) if a.mode=='train' else infer(a)
