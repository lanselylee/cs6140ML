"""Discarded real-data feasibility checks, not formal training or test scoring."""
import os
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
import csv,gc,json,random,sys,time
from collections import Counter
from functools import partial
from pathlib import Path
import torch
from torch.utils.data import DataLoader,Subset
import pandas as pd
from transformers import AutoTokenizer
from model import CausalMultitask,local_snapshot
from word_data import CURRENT_DATA,WordWindows,collate_word,csv_rows,sha,validate_contract
from train import evaluate,move

HERE=Path(__file__).resolve().parent
OUT=HERE/'phase2_common12_v4'
sys.path.insert(0,str(HERE.parent/'shared_phase1/vendor'))
import cs_eval

def emit(s):print(s,flush=True)
def batch_checks(b):
    for row in range(b['input_ids'].shape[0]):
        mask=b['switch_labels'][row]>=0
        assert ((b['duration_labels'][row]>=0)<= (b['switch_labels'][row]==1)).all()
        assert (b['switch_labels'][row][b['attention_mask'][row]==0]==-1).all()
        for p in mask.nonzero().flatten().tolist():
            word=int(b['word_ids'][row,p]);assert word>=0
            assert p+1==b['word_ids'].shape[1] or int(b['word_ids'][row,p+1])!=word

def main():
    start=time.time();OUT.mkdir(exist_ok=True)
    manifest,delivery=validate_contract(CURRENT_DATA)
    expected_pairs=len(manifest['full_pairs'])
    assert expected_pairs==12
    random.seed(6130);torch.manual_seed(6130);torch.cuda.manual_seed_all(6130)
    assert torch.cuda.is_available()
    train=WordWindows(CURRENT_DATA,'train');dev=WordWindows(CURRENT_DATA,'val')
    emit(f'Loaded v4 only: {len(train)} train windows / {len(dev)} validation windows; test not loaded')
    tok=AutoTokenizer.from_pretrained(local_snapshot(),local_files_only=True,token=False)
    assert Path(local_snapshot()).name==manifest['tokenizer_revisions']['xlmr']
    collate=partial(collate_word,pad=tok.pad_token_id)
    by_pair={};long=[];overlap=None
    for i,r in enumerate(train.rows):
        if r['pair'] not in by_pair and 1 in r['sw_labels']:by_pair[r['pair']]=i
        if len(r['input_ids'])>=240 and len(long)<4:long.append(i)
        if r['owned_word_start']>r['word_start'] and overlap is None:overlap=i
    assert len(by_pair)==expected_pairs
    assert overlap is not None
    overlap_row=train[overlap]
    assert all(y==-1 for w,y in zip(overlap_row['word_ids'],overlap_row['switch_labels']) if w<overlap_row['owned_word_start'])
    emit('PASS: 12-pair real word-end alignment and repeated-context masking')
    model=CausalMultitask().cuda().eval()
    sample=train[by_pair['Chinese-English']]
    ids=torch.tensor([sample['input_ids']],device='cuda');attention=torch.ones_like(ids)
    candidates=[j for j,y in enumerate(sample['switch_labels']) if y==1 and j+2<ids.shape[1]]
    anchor=candidates[0];cut=anchor+1
    changed=ids.clone();changed[:,cut:]=torch.randint(100,20000,changed[:,cut:].shape,device='cuda')
    with torch.no_grad():
        original=model(ids,attention);alternative=model(changed,attention)
        prefix=model(ids[:,:cut],attention[:,:cut])
    future_delta=max(float((x[:,:cut]-y[:,:cut]).abs().max()) for x,y in zip(original,alternative))
    prefix_delta=max(float((x[:,:cut]-y).abs().max()) for x,y in zip(original,prefix))
    assert future_delta<1e-4 and prefix_delta<1e-4
    emit(f'PASS: real word-boundary causal check; future delta={future_delta}, prefix delta={prefix_delta}')
    del original,alternative,prefix,ids,changed

    # Four actual v4 windows; padding does not create labels. This is a
    # conservative length-256 memory check with real supervision.
    selected=long[:4]
    if len(selected)<4:selected=list(by_pair.values())[:4]
    b=collate([train[i] for i in selected]);batch_checks(b)
    for k in ['input_ids','attention_mask','switch_labels','duration_labels','word_ids']:
        value=tok.pad_token_id if k=='input_ids' else 0 if k=='attention_mask' else -1
        b[k]=torch.nn.functional.pad(b[k],(0,256-b[k].shape[1]),value=value)
    b=move(b);model.train();torch.cuda.reset_peak_memory_stats()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01,foreach=False)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    old_head=model.switch_head.weight.detach().cpu().clone()
    with torch.autocast('cuda',dtype=torch.float16):
        sw,dur=model(b['input_ids'],b['attention_mask'])
        sw.retain_grad();dur.retain_grad()
        loss,swloss,durloss=model.loss(sw,dur,b['switch_labels'],b['duration_labels'],duration_weight=.5)
        equal,_,_=model.loss(sw,dur,b['switch_labels'],b['duration_labels'],duration_weight=1.)
        assert torch.allclose(loss,swloss+.5*durloss)
        assert torch.allclose(equal,swloss+durloss)
        duration_gradient=torch.autograd.grad(durloss,dur,retain_graph=True)[0]
        weighted_gradient=torch.autograd.grad(loss,dur,retain_graph=True)[0]
        assert torch.allclose(weighted_gradient.float(),.5*duration_gradient.float(),atol=1e-5,rtol=1e-3)
    assert torch.isfinite(loss) and durloss>0
    scaler.scale(loss).backward()
    assert (sw.grad[b['switch_labels']<0]==0).all()
    assert (dur.grad[b['duration_labels']<0]==0).all()
    scaler.unscale_(optimizer)
    for name,module in [('backbone',model.backbone),('switch_head',model.switch_head),('duration_head',model.duration_head)]:
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert any(g.abs().sum()>0 for g in grads),name
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True)
    assert optimizer.state and not torch.equal(old_head,model.switch_head.weight.detach().cpu())
    _,_,zero_duration=model.loss(sw,dur,torch.zeros_like(b['switch_labels']),torch.full_like(b['duration_labels'],-1))
    assert zero_duration.item()==0
    emit('PASS: real batch4 x length256 backward, both heads/backbone, ignored gradients zero, AdamW step not skipped')
    peak=torch.cuda.max_memory_allocated()/1024**3
    # Real state serialization/restore. Discard this checkpoint and optimization
    # entirely: formal baseline must initialize from pretrained weights again.
    ckpt=OUT/'discarded_roundtrip.pt'
    saved=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
        rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),
        manifest_sha256=sha(CURRENT_DATA/'manifest.json'))
    torch.save(saved,ckpt);del saved
    expected=model.switch_head.weight.detach().cpu().clone()
    loaded=torch.load(ckpt,map_location='cpu',weights_only=False)
    with torch.no_grad():model.switch_head.weight.add_(1.)
    model.load_state_dict(loaded['model']);optimizer.load_state_dict(loaded['optimizer']);scaler.load_state_dict(loaded['scaler'])
    torch.set_rng_state(loaded['rng']);torch.cuda.set_rng_state_all(loaded['cuda_rng']);random.setstate(loaded['python_rng'])
    assert torch.equal(expected,model.switch_head.weight.detach().cpu())
    assert optimizer.state and all(int(s['step'])==1 for s in optimizer.state.values())
    del loaded;ckpt.unlink();optimizer.zero_grad(set_to_none=True)
    emit('PASS: model/optimizer/scaler/RNG checkpoint saved and restored; temporary checkpoint removed')
    del optimizer,sw,dur,b;gc.collect();torch.cuda.empty_cache()

    # Validation-only diagnostics: same real window predictions through our
    # accumulator and the team's unmodified cs_eval. No test access/results.
    val_pairs={}
    for i,r in enumerate(dev.rows):
        if r['pair'] not in val_pairs and 1 in r['sw_labels']:val_pairs[r['pair']]=i
        if len(val_pairs)==expected_pairs:break
    assert len(val_pairs)==expected_pairs
    loader=DataLoader(Subset(dev,list(val_pairs.values())),batch_size=4,collate_fn=collate)
    model.eval();csv_output=[]
    with torch.no_grad():
        for batch in loader:
            batch_checks(batch);gpu=move(batch)
            with torch.autocast('cuda',dtype=torch.float16):
                out_sw,out_dur=model(gpu['input_ids'],gpu['attention_mask'])
            csv_output.extend(csv_rows(batch,out_sw.float().softmax(-1)[...,1],out_dur.argmax(-1)))
    df=pd.DataFrame(csv_output)
    assert not df.duplicated(['id','word_idx']).any()
    team=cs_eval.evaluate(df,model_name='discarded_phase2_diagnostic',out_dir=None,verbose=False)
    ours=evaluate(model,loader)['official']
    for k in ['anticipatory_f1','duration_accuracy','macro_average_accuracy','sigma_universality']:
        assert abs(team[k]-ours[k])<1e-10,(k,team[k],ours[k])
    assert all(abs(team['f1_by_pair'][p]-v)<1e-10 for p,v in ours['f1_by_pair'].items())
    emit('PASS: word-level CSV unique IDs and all official metrics match unmodified team script')
    codes=['model.py','train.py','word_data.py','check_word_phase2.py']
    result=dict(status='passed',data_version=manifest['version'],data_root=str(CURRENT_DATA),
        manifest_sha256=sha(CURRENT_DATA/'manifest.json'),delivery_checks_sha256=sha(CURRENT_DATA/'delivery_checks.json'),
        code_sha256={p:sha(HERE/p) for p in codes},gpu=torch.cuda.get_device_name(0),
        backbone_parameters=sum(p.numel() for p in model.backbone.parameters()),
        total_parameters=sum(p.numel() for p in model.parameters()),batch_size=4,padded_length=256,
        peak_allocated_gib=peak,future_change_max_delta=future_delta,prefix_only_max_delta=prefix_delta,
        train_windows=len(train),validation_windows=len(dev),test_loaded=False,
        real_backward_and_optimizer_step=True,ignored_logits_gradients_zero=True,
        checked_duration_weights=[1.0,0.5],duration_gradient_weighting_checked=True,
        all_pairs_checked=True,pair_count=expected_pairs,checkpoint_roundtrip=True,temporary_checkpoint_removed=True,
        official_metric_parity=True,diagnostic_scored_word_positions=len(df),
        seconds=time.time()-start,formal_training_started=False,
        note='One real optimizer step for feasibility, discarded. No final/benchmark metrics; initialize fresh pretrained backbone for formal training.')
    (OUT/'preflight_pass.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    emit(json.dumps(result,indent=2))
    emit('PHASE2_COMMON12_V4_COMPLETE_NO_FORMAL_TRAINING')

if __name__=='__main__':main()
