"""Actual-GPU preflight for causal estimated prefix-LID only."""
from pathlib import Path
import json,gc
import torch
from model import CausalMultitask,local_snapshot
from lid_model import LIDCausalMultitask
from lid_features import LIDWindows,collate_lid,causal_ids,IDS
from word_data import CURRENT_DATA,sha
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent
OUT=HERE/'phase2_lid_v4'
def main():
    OUT.mkdir(exist_ok=True)
    for pair,words in [('French-English',['Je','suis','happy','today']),('Chinese-English',['今天','hello','world'])]:
        a=causal_ids(words,pair)
        for n in range(1,len(words)+1):assert causal_ids(words[:n],pair)==a[:n]
        assert causal_ids(words[:2]+['something','different'],pair)[:2]==a[:2]
    ds=LIDWindows(CURRENT_DATA,'train');pairs={}
    for i,r in enumerate(ds.rows):
        if r['pair'] not in pairs and 1 in r['sw_labels']:pairs[r['pair']]=i
    assert len(pairs)==12
    tok=AutoTokenizer.from_pretrained(local_snapshot(),token=False,local_files_only=True)
    sample=ds[pairs['Chinese-English']]
    b=collate_lid([sample],tok.pad_token_id)
    b={k:v.cuda() if torch.is_tensor(v) else v for k,v in b.items()}
    torch.manual_seed(6130);base=CausalMultitask(checkpointing=False).cuda().eval()
    torch.manual_seed(6130);model=LIDCausalMultitask().cuda().eval()
    with torch.no_grad():
        a=base(b['input_ids'],b['attention_mask']);c=model(b['input_ids'],b['attention_mask'],b['lid_ids'])
    zero_delta=max(float((x-y).abs().max()) for x,y in zip(a,c));assert zero_delta<1e-5
    del base,a,c;gc.collect();torch.cuda.empty_cache()
    with torch.no_grad():model.language_embedding.weight.normal_(0,.02);model.language_embedding.weight[0].zero_()
    cut=next(j+1 for j,y in enumerate(sample['switch_labels']) if y==1 and j+2<len(sample['input_ids']))
    changed=b['input_ids'].clone();changed[:,cut:]=torch.randint(100,20000,changed[:,cut:].shape,device='cuda')
    changed_lid=b['lid_ids'].clone();changed_lid[:,cut:]=torch.randint(1,len(IDS)+1,changed_lid[:,cut:].shape,device='cuda')
    with torch.no_grad():
        original=model(b['input_ids'],b['attention_mask'],b['lid_ids'])
        future=model(changed,b['attention_mask'],changed_lid)
        prefix=model(b['input_ids'][:,:cut],b['attention_mask'][:,:cut],b['lid_ids'][:,:cut])
    future_delta=max(float((x[:,:cut]-y[:,:cut]).abs().max()) for x,y in zip(original,future))
    prefix_delta=max(float((x[:,:cut]-y).abs().max()) for x,y in zip(original,prefix))
    assert future_delta<1e-4 and prefix_delta<1e-4
    examples=[]
    for index in pairs.values():
        r=ds[index];last={w:j for j,w in enumerate(r['word_ids']) if w>=0}
        assert all(x==0 or j==last[r['word_ids'][j]] for j,x in enumerate(r['lid_ids']))
        examples.append(r)
    selected=[r for r in examples if any(x>=0 for x in r['duration_labels'])][:4]
    batch=collate_lid(selected,tok.pad_token_id)
    for k,v in list(batch.items()):
        if torch.is_tensor(v):
            fill=tok.pad_token_id if k=='input_ids' else -1 if k in ['switch_labels','duration_labels','word_ids'] else 0
            batch[k]=torch.nn.functional.pad(v,(0,256-v.shape[1]),value=fill).cuda()
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,foreach=False)
    scaler=torch.amp.GradScaler('cuda',init_scale=128.)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda',dtype=torch.float16):
        sw,dur=model(batch['input_ids'],batch['attention_mask'],batch['lid_ids'])
        loss,_,_=model.loss(sw,dur,batch['switch_labels'],batch['duration_labels'])
    scaler.scale(loss).backward();scaler.unscale_(optimizer)
    for module in [model.backbone,model.switch_head,model.duration_head,model.language_embedding]:
        assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in module.parameters())
    assert (model.language_embedding.weight.grad[0]==0).all()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(optimizer);scaler.update();assert optimizer.state
    file=OUT/'discarded_roundtrip.pt'
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict()),file)
    state=torch.load(file,map_location='cpu',weights_only=False)
    expected=model.language_embedding.weight.detach().cpu().clone()
    with torch.no_grad():model.language_embedding.weight.add_(1.)
    model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler'])
    assert torch.equal(model.language_embedding.weight.detach().cpu(),expected)
    del state;file.unlink()
    result=dict(status='passed',manifest_sha256=sha(CURRENT_DATA/'manifest.json'),
        delivery_checks_sha256=sha(CURRENT_DATA/'delivery_checks.json'),checked_duration_weights=[1.0],
        code_sha256={p:sha(HERE/p) for p in ['train_lid.py','lid_features.py','lid_model.py','word_data.py','model.py','check_lid.py']},
        zero_embedding_baseline_delta=zero_delta,future_change_max_delta=future_delta,prefix_only_max_delta=prefix_delta,
        online_lid_prefix_stable=True,full_sentence_gold_lid_not_used=True,word_internal_features_zero=True,
        pair_count=12,language_embedding_gradient=True,checkpoint_roundtrip=True,batch_size=4,padded_length=256,
        peak_allocated_gib=torch.cuda.max_memory_allocated()/1024**3,test_loaded=False,
        note='Pair-aware online estimated LID from visible words only; no gold language labels. Test update discarded.')
    (OUT/'preflight_pass.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
