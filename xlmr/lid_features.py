"""Causal estimated LID: forward filtering only, no gold lang/sw/dur as features."""
import json,unicodedata
from functools import lru_cache
from pathlib import Path
import regex as re
from wordfreq import zipf_frequency
from word_data import WordWindows,collate_word

LANGUAGES=['en','ar','zh','fr','de','hi','it','ja','ko','ms','fil','ru','es']
IDS={x:i+1 for i,x in enumerate(LANGUAGES)}  # 0 = unknown/no completed word feature
PAIR={'Arabic-English':'ar','Chinese-English':'zh','French-English':'fr','German-English':'de',
 'Hindi-English':'hi','Italian-English':'it','Japanese-English':'ja','Korean-English':'ko',
 'Malay-English':'ms','Philippines-English':'fil','Russian-English':'ru','Spanish-English':'es'}
SCRIPTS={'ar':r'\p{Arabic}','zh':r'\p{Han}','hi':r'\p{Devanagari}',
 'ja':r'[\p{Han}\p{Hiragana}\p{Katakana}]','ko':r'\p{Hangul}','ru':r'\p{Cyrillic}'}
@lru_cache(maxsize=200000)
def freq(word,language):return zipf_frequency(word,language)

def causal_ids(words,pair):
    native=PAIR[pair];result=[];previous=0;state=[0.,0.]
    for i,word in enumerate(words):
        core=''.join(c for c in word if unicodedata.category(c)[0] in ['L','M'] or c in "'’")
        if not core or not any(c.isalpha() for c in core):result.append(previous);continue
        if native in SCRIPTS:
            if re.search(SCRIPTS[native],core):language=native
            elif re.search(r'\p{Latin}',core):language='en'
            else:result.append(previous);continue
        else:
            lower=core.casefold().replace('’',"'")
            emission=[freq(lower,native),freq(lower,'en')]
            if i>0 and core[0].isupper() and max(emission)<3.5:emission=[0.,0.]
            new=[max(state[k],state[1-k]-.75)+emission[k] for k in [0,1]]
            norm=max(new);state=[x-norm for x in new]
            language=native if state[0]>state[1] else 'en'
        previous=IDS[language];result.append(previous)
    return result

class LIDWindows(WordWindows):
    def __init__(self,root,split):
        super().__init__(root,split);self.estimated={}
        for line in (Path(root)/'labels'/f'{split}_labels.jsonl').open(encoding='utf-8'):
            s=json.loads(line)
            # Deliberately read words/pair only: never s['lang'], sw or dur.
            self.estimated[s['id']]=causal_ids(s['words'],s['pair'])
    def __getitem__(self,index):
        r=super().__getitem__(index);features=[0]*len(r['input_ids'])
        last={w:j for j,w in enumerate(r['word_ids']) if w>=0}
        for word,j in last.items():features[j]=self.estimated[r['id']][word]
        r['lid_ids']=features
        return r

def collate_lid(rows,pad):
    import torch
    b=collate_word(rows,pad);b['lid_ids']=torch.zeros_like(b['input_ids'])
    for i,r in enumerate(rows):b['lid_ids'][i,:len(r['lid_ids'])]=torch.tensor(r['lid_ids'])
    return b
