"""The active v4 word-level dataset adapter. No old subword datasets allowed."""
import hashlib,json
from pathlib import Path
import torch
from torch.utils.data import Dataset

HERE=Path(__file__).resolve().parent
CURRENT_DATA=HERE.parent/'outputs/Shared_Phase1_Common12_v4'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()

def validate_contract(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    checks=json.loads((root/'delivery_checks.json').read_text(encoding='utf-8'))
    checks['sha256']={k.replace('\\','/'):v for k,v in checks['sha256'].items()}
    assert manifest['version'].startswith('shared_word_v4'), 'Only corrected word-level v4 datasets are supported'
    assert manifest['complete'] and checks['status']=='passed'
    assert checks['cross_split_normalized_sentence_overlap']==0
    assert manifest['models']['xlmr']=='FacebookAI/xlm-roberta-base'
    return manifest,checks

class WordWindows(Dataset):
    """Read the shared .pt verbatim. Convert names only, never retokenize.

    Full train/validation metadata are loaded once. No test file is loaded unless
    the caller explicitly requests it (formal trainer only requests train/val).
    """
    def __init__(self,root,split):
        root=Path(root);self.root=root;self.split=split
        assert split in ['train','val','test']
        _,checks=validate_contract(root)
        path=root/'encoded/xlmr'/f'{split}.pt'
        assert sha(path)==checks['sha256'][f'encoded/xlmr/{split}.pt'],'Encoding file changed after validation'
        label_path=root/'labels'/f'{split}_labels.jsonl'
        assert sha(label_path)==checks['sha256'][f'labels/{split}_labels.jsonl'],'Labels changed after validation'
        self.rows=torch.load(path,map_location='cpu',weights_only=False)
        self.types={}
        for line in label_path.open(encoding='utf-8'):
            s=json.loads(line);self.types[s['id']]=s['cs_type'] or 'unknown'
        assert len(self.rows)==checks['results'][split]['models']['xlmr']['windows']
        self.source_encoding_sha256=checks['sha256'][f'encoded/xlmr/{split}.pt']

    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index]
        return dict(input_ids=r['input_ids'],attention_mask=r['attention_mask'],
            switch_labels=r['sw_labels'],duration_labels=r['dur_labels'],
            pair=r['pair'],id=r['id'],window_id=r['window_id'],word_ids=r['word_ids'],
            cs_type=self.types[r['id']],owned_word_start=r['owned_word_start'])

def collate_word(rows,pad):
    n=min(256,((max(len(r['input_ids']) for r in rows)+7)//8)*8)
    batch={k:torch.full((len(rows),n),value,dtype=torch.long) for k,value in
        [('input_ids',pad),('attention_mask',0),('switch_labels',-1),('duration_labels',-1),('word_ids',-1)]}
    for i,r in enumerate(rows):
        size=len(r['input_ids']);assert size<=n
        for k in batch:batch[k][i,:size]=torch.tensor(r[k],dtype=torch.long)
    for k in ['pair','id','window_id','cs_type']:batch[k]=[r[k] for r in rows]
    return batch

def csv_rows(batch,sw_prob,dur_pred):
    """Only owned, scored word endings become rows in the shared CSV format."""
    sw_prob=sw_prob.detach().cpu();dur_pred=dur_pred.detach().cpu()
    sw=batch['switch_labels'].detach().cpu();dur=batch['duration_labels'].detach().cpu()
    words=batch['word_ids'].detach().cpu()
    for i in range(sw.shape[0]):
        for j in (sw[i]>=0).nonzero().flatten().tolist():
            yield dict(id=batch['id'][i],word_idx=int(words[i,j]),pair=batch['pair'][i],
                cs_type=batch['cs_type'][i],sw_true=int(sw[i,j]),dur_true=int(dur[i,j]),
                sw_prob=float(sw_prob[i,j]),sw_pred=int(sw_prob[i,j]>=.5),dur_pred=int(dur_pred[i,j]))
