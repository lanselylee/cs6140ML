"""Causal language-ID embedding augmentation; baseline architecture otherwise intact."""
import torch
from torch import nn
from model import CausalMultitask
from lid_features import LANGUAGES

class LIDCausalMultitask(CausalMultitask):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with torch.random.fork_rng(devices=[]):
            self.language_embedding=nn.Embedding(len(LANGUAGES)+1,self.backbone.config.hidden_size,padding_idx=0)
        nn.init.zeros_(self.language_embedding.weight)
    def forward(self,input_ids,attention_mask,lid_ids):
        embedded=self.backbone.get_input_embeddings()(input_ids)+self.language_embedding(lid_ids)
        pad=self.backbone.config.pad_token_id
        mask=input_ids.ne(pad).long()
        positions=mask.cumsum(1)*mask+pad
        hidden=self.backbone(inputs_embeds=embedded,position_ids=positions,attention_mask=attention_mask,use_cache=False).last_hidden_state
        hidden=self.dropout(hidden)
        return self.switch_head(hidden),self.duration_head(hidden)
