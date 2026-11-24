"""XLM-R causal dual-head training. Test data are never loaded here."""
import os
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from model import CausalMultitask, local_snapshot
from transformers import AutoTokenizer
from word_data import WordWindows, collate_word, validate_contract, CURRENT_DATA, sha

HERE=Path(__file__).resolve().parent


def move(batch):return {k:v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


def forward(model,batch):
    return model(batch['input_ids'],batch['attention_mask'])


def statistics(matrix):
    c=matrix.double();tp=c.diag();support=c.sum(1)
    precision=tp/c.sum(0).clamp_min(1);recall=tp/support.clamp_min(1)
    f1=2*tp/(c.sum(0)+support).clamp_min(1)
    return dict(n=int(c.sum()),accuracy=float(tp.sum()/c.sum().clamp_min(1)),
                macro_f1=float(f1.mean()),class_f1=f1.tolist(),
                precision=precision.tolist(),recall=recall.tolist(),support=support.int().tolist(),
                confusion=c.int().tolist())


@torch.no_grad()
def evaluate(model,loader,limit=0):
    model.eval();counts={};losses=[0.,0.,0,0]
    for i,b in enumerate(loader):
        if limit and i>=limit:break
        b=move(b)
        with torch.autocast('cuda',dtype=torch.float16):
            sw,dur=forward(model,b)
        for key,logits,labels,classes in [('switch',sw,b['switch_labels'],2),('duration',dur,b['duration_labels'],3)]:
            pred=(logits.float().softmax(-1)[...,1]>=.5).long() if key=='switch' else logits.argmax(-1)
            mask=labels>=0
            if mask.any():
                idx=0 if key=='switch' else 1
                losses[idx]+=float(torch.nn.functional.cross_entropy(logits[mask].float(),labels[mask],reduction='sum'))
                losses[idx+2]+=int(mask.sum())
            for row,pair in enumerate(b['pair']):
                valid=mask[row]
                c=torch.bincount(classes*labels[row][valid]+pred[row][valid],minlength=classes**2).reshape(classes,classes).cpu()
                for group in ['pooled',pair,'type:'+b['cs_type'][row]]:
                    if group not in counts:counts[group]={'switch':torch.zeros(2,2,dtype=torch.long),'duration':torch.zeros(3,3,dtype=torch.long)}
                    counts[group][key]+=c
    report={g:{k:statistics(c) for k,c in tasks.items()} for g,tasks in counts.items()}
    report['loss']={'switch':losses[0]/max(losses[2],1),'duration':losses[1]/max(losses[3],1)}
    pairs=[g for g in counts if g!='pooled' and not g.startswith('type:')]
    values=[report[g]['switch']['class_f1'][1] for g in pairs]
    types=[report[g]['switch']['accuracy'] for g in counts if g.startswith('type:') and g[5:].strip().lower()!='unknown']
    report['official']={
        'anticipatory_f1':report['pooled']['switch']['class_f1'][1],
        'duration_accuracy':report['pooled']['duration']['accuracy'],
        'macro_average_accuracy':sum(types)/len(types) if types else None,
        'f1_by_pair':dict(zip(pairs,values)),
        'pair_mean_f1':sum(values)/len(values),
        'sigma_universality':float(torch.tensor(values,dtype=torch.float64).std(correction=0))}
    return report


def main(args):
    assert torch.cuda.is_available(),'CUDA required'
    assert math.isfinite(args.duration_weight) and args.duration_weight>0
    validate_contract(args.data)
    random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    tok=AutoTokenizer.from_pretrained(local_snapshot(),local_files_only=True,token=False)
    model=CausalMultitask().cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01,foreach=False)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    train=WordWindows(args.data,'train');dev=WordWindows(args.data,'val')
    from functools import partial
    batcher=partial(collate_word,pad=tok.pad_token_id)
    args.output.mkdir(parents=True,exist_ok=True)
    latest=args.output/'last.pt'
    if latest.exists() and not args.resume and not args.check:
        raise RuntimeError('Existing run found; use --resume or another --output')
    if not args.check and __import__('shutil').disk_usage(args.output).free < 15*1024**3:
        raise RuntimeError('At least 15 GiB free disk required for optimizer checkpoints')
    manifest_hash=__import__('hashlib').sha256((args.data/'manifest.json').read_bytes()).hexdigest()
    if not args.check:
        gate=json.loads((HERE/'phase2_common12_v4'/'preflight_pass.json').read_text())
        assert gate['status']=='passed' and gate['manifest_sha256']==manifest_hash,'Training preflight not passed for this data version'
        assert gate['delivery_checks_sha256']==sha(args.data/'delivery_checks.json')
        assert args.duration_weight in gate.get('checked_duration_weights',[1.0]),'Loss weight not preflight checked'
        for name,digest in gate['code_sha256'].items():assert sha(HERE/name)==digest,('Code changed since preflight',name)
    batches=math.ceil(len(train)/args.batch_size)
    total_steps=args.epochs*math.ceil(batches/args.accumulation)
    epoch0=batch0=step=0;best=-1.
    if args.resume:
        state=torch.load(latest,map_location='cpu',weights_only=False)
        assert state['manifest_hash']==manifest_hash,'Dataset changed'
        for key in ['batch_size','accumulation','epochs','lr','seed','duration_weight']:
            assert state['args'][key]==getattr(args,key),('Resume configuration changed',key)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
        epoch0,batch0,step,best=[state[k] for k in ['next_epoch','next_batch','step','best']]
        torch.set_rng_state(state['rng']);torch.cuda.set_rng_state_all(state['cuda_rng']);random.setstate(state['python_rng'])
        del state
    def save(next_epoch,next_batch,path):
        state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                   next_epoch=next_epoch,next_batch=next_batch,step=step,best=best,
                   rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),
                   manifest_hash=manifest_hash,args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
        temp=path.with_suffix('.tmp');torch.save(state,temp);os.replace(temp,path)
    if args.check:
        # Actual full-size model, maximum 256-token batch, optimizer allocation.
        examples=[train[i] for i in range(min(500,len(train)))]
        examples.sort(key=lambda x:len(x['input_ids']),reverse=True)
        b=move(batcher(examples[:args.batch_size]))
        for k,v in list(b.items()):
            if torch.is_tensor(v):
                b[k]=torch.nn.functional.pad(v,(0,256-v.shape[1]),value=tok.pad_token_id if k=='input_ids' else -1 if k.endswith('labels') else 0)
        model.train();torch.cuda.reset_peak_memory_stats();start=time.time()
        with torch.autocast('cuda',dtype=torch.float16):
            sw,dur=forward(model,b);loss,_,_=model.loss(sw,dur,b['switch_labels'],b['duration_labels'],duration_weight=args.duration_weight)
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
        assert torch.isfinite(loss)
        assert len(optimizer.state)>0,'Optimizer step skipped by AMP; feasibility is not confirmed'
        torch.cuda.synchronize()
        # Serialize the real optimizer state and reload it to prove resumability.
        checkpoint=args.output/'resume_check.pt'
        save(0,0,checkpoint)
        loaded=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert loaded['manifest_hash']==manifest_hash
        assert loaded['optimizer']['state'],'Missing AdamW state in checkpoint'
        original_head=model.switch_head.weight.detach().cpu().clone()
        with torch.no_grad():model.switch_head.weight.add_(1.)
        model.load_state_dict(loaded['model']);optimizer.load_state_dict(loaded['optimizer']);scaler.load_state_dict(loaded['scaler'])
        assert torch.equal(model.switch_head.weight.detach().cpu(),original_head),'Checkpoint restore failed'
        del loaded
        checkpoint.unlink()
        result=dict(status='passed',checkpoint_roundtrip=True,batch_size=args.batch_size,padded_length=int(b['input_ids'].shape[1]),
                    seconds=time.time()-start,peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,
                    loss=float(loss.detach()),train_windows=len(train),validation_windows=len(dev),
                    note='One discarded optimizer-step feasibility check, not an experiment checkpoint')
        (args.output/'training_check.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
        return
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(manifest_hash=manifest_hash,model='FacebookAI/xlm-roberta-base',
                  method='v4 word endings only; repeated context ignored; joint switch/duration CE (-1); no LID input or punctuation fusion; switch probability>=0.5',
                  selection='Mean of validation switch positive F1 and duration three-class macro-F1',
                  lambda_switch=1.0,lambda_duration=args.duration_weight)
    (args.output/'config.json').write_text(json.dumps(config,indent=2))
    log=(args.output/'train.jsonl').open('a',encoding='utf-8')
    def emit(x):
        log.write(json.dumps(x)+'\n');log.flush();print(json.dumps(x),flush=True)
    try:
        for epoch in range(epoch0,args.epochs):
            order=torch.randperm(len(train),generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
            offset=batch0*args.batch_size if epoch==epoch0 else 0
            loader=DataLoader(Subset(train,order[offset:]),batch_size=args.batch_size,collate_fn=batcher,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(args.seed+epoch+10000))
            model.train();optimizer.zero_grad(set_to_none=True);start=time.time();running=0.;pending=0
            for local_i,b in enumerate(loader):
                i=local_i+offset//args.batch_size
                b=move(b)
                group_start=(i//args.accumulation)*args.accumulation
                divisor=min(args.accumulation,batches-group_start)
                with torch.autocast('cuda',dtype=torch.float16):
                    sw,dur=forward(model,b)
                    loss,sw_loss,dur_loss=model.loss(sw,dur,b['switch_labels'],b['duration_labels'],duration_weight=args.duration_weight)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                scaler.scale(loss/divisor).backward();running+=float(loss.detach());pending+=1
                if (i+1)%args.accumulation==0 or i+1==batches:
                    warmup=max(1,int(total_steps*.03))
                    rate=(step+1)/warmup if step<warmup else max(0.,(total_steps-step)/max(1,total_steps-warmup))
                    for group in optimizer.param_groups:group['lr']=args.lr*rate
                    scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    old_scale=scaler.get_scale();scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
                    step+=1
                    if step%25==0:
                        emit(dict(epoch=epoch+1,batch=i+1,batches=batches,step=step,total_steps=total_steps,
                                  mean_loss=running/max(1,pending),switch_loss=float(sw_loss.detach()),
                                  duration_loss=float(dur_loss.detach()),lr=optimizer.param_groups[0]['lr'],
                                  scaler_skipped_update=scaler.get_scale()<old_scale,elapsed_epoch_seconds=time.time()-start))
                        running=0.;pending=0
                    if step%args.save_every==0:save(epoch,i+1,latest)
                    if args.eval_every and step%args.eval_every==0 and i+1<batches:
                        validation=evaluate(model,DataLoader(dev,batch_size=args.batch_size,collate_fn=batcher,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(args.seed+20000)))
                        score=(validation['pooled']['switch']['class_f1'][1]+validation['pooled']['duration']['macro_f1'])/2
                        emit(dict(epoch=epoch+1,batch=i+1,step=step,validation=validation,selection_score=score))
                        if score>best:
                            best=score;save(epoch,i+1,args.output/'best.pt')
                        save(epoch,i+1,latest)
                        model.train()
            validation=evaluate(model,DataLoader(dev,batch_size=args.batch_size,collate_fn=batcher,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(args.seed+epoch+20000)))
            score=(validation['pooled']['switch']['class_f1'][1]+validation['pooled']['duration']['macro_f1'])/2
            emit(dict(epoch=epoch+1,step=step,validation=validation,selection_score=score))
            if score>best:
                best=score;save(epoch+1,0,args.output/'best.pt')
            save(epoch+1,0,latest);batch0=0
        emit(dict(status='training_complete',best_validation_score=best,test_evaluated=False))
    except KeyboardInterrupt:
        print('Stopped. Resume from last.pt; unfinished work after last checkpoint will be repeated.',flush=True)
    finally:log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=CURRENT_DATA)
    p.add_argument('--output',type=Path,default=HERE/'runs'/'xlmr_common12_word_v4')
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--accumulation',type=int,default=8)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--lr',type=float,default=2e-5)
    p.add_argument('--duration-weight',type=float,default=1.0)
    p.add_argument('--seed',type=int,default=6130)
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--check',action='store_true')
    args=p.parse_args()
    main(args)
