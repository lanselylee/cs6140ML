import gc,json,tempfile
from pathlib import Path
import torch
from single_task import SingleTaskModel,task_loss,target
from word_data import CURRENT_DATA,WordWindows,collate_word,sha,validate_contract

HERE=Path(__file__).resolve().parent
def main():
    validate_contract(CURRENT_DATA)
    ds=WordWindows(CURRENT_DATA,'train')
    samples=[]
    for i in range(len(ds)):
        r=ds[i]
        if any(y>=0 for y in r['duration_labels']):samples.append(r)
        if len(samples)==4:break
    b={k:v.cuda() if torch.is_tensor(v) else v for k,v in collate_word(samples,pad=1).items()}
    assert not ((b['duration_labels']>=0)&(b['switch_labels']!=1)).any()
    reports={}
    for task in ['switch','duration']:
        torch.manual_seed(6130);torch.cuda.manual_seed_all(6130)
        m=SingleTaskModel(task).cuda();m.eval()
        ids=b['input_ids'];mask=b['attention_mask'];cut=min(8,ids.shape[1]-2)
        with torch.no_grad():
            a=m(ids,mask);changed=ids.clone();changed[:,cut+1:]=42;c=m(changed,mask)
            delta=float((a[:,:cut+1]-c[:,:cut+1]).abs().max())
            assert delta<1e-4,delta
        del a,c,changed
        m.train();active=m.switch_head if task=='switch' else m.duration_head
        inactive=m.duration_head if task=='switch' else m.switch_head
        before=inactive.weight.detach().clone()
        params=[p for p in m.parameters() if p.requires_grad]
        assert not any(id(p) in {id(q) for q in inactive.parameters()} for p in params)
        opt=torch.optim.AdamW(params,lr=2e-5,foreach=False)
        scaler=torch.amp.GradScaler('cuda',init_scale=128.)
        with torch.autocast('cuda',dtype=torch.float16):
            z=m(ids,mask);z.retain_grad();loss=task_loss(z,target(b,task))
        scaler.scale(loss).backward();scaler.unscale_(opt)
        assert torch.isfinite(loss) and active.weight.grad.abs().sum()>0
        assert m.backbone.encoder.layer[0].attention.self.query.weight.grad.abs().sum()>0
        assert all(p.grad is None for p in inactive.parameters())
        assert (z.grad[target(b,task)<0]==0).all()
        torch.nn.utils.clip_grad_norm_(params,1.);scaler.step(opt);scaler.update()
        assert torch.equal(before,inactive.weight)
        empty=torch.randn(2,3,3,device='cuda',requires_grad=True)
        zero=task_loss(empty,torch.full((2,3),-1,device='cuda'));zero.backward()
        assert zero.item()==0 and (empty.grad==0).all()
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'check.pt';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict()),p)
            state=torch.load(p,map_location='cpu',weights_only=False)
            m.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);del state
        reports[task]=dict(loss=float(loss),causal_max_delta=delta,inactive_frozen=True,ignored_grad_zero=True,checkpoint_roundtrip=True)
        print(task,reports[task],flush=True)
        del m,opt,scaler,z,loss,active,inactive,params,before,empty,zero
        gc.collect();torch.cuda.empty_cache()
    out=HERE/'phase2_single_task';out.mkdir(exist_ok=True)
    result=dict(status='passed',checked_tasks=list(reports),reports=reports,manifest_sha256=sha(CURRENT_DATA/'manifest.json'),
        code_sha256={name:sha(HERE/name) for name in ['single_task.py','model.py','word_data.py','check_single_task.py']})
    (out/'preflight_pass.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
if __name__=='__main__':main()
