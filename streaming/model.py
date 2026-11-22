"""Causal dual-head encoder for streaming code-switch prediction (spec Section III).

A bidirectional multilingual encoder (XLM-R or mBERT) is turned into a
left-to-right model by setting ``config.is_decoder = True``. Hugging Face then
builds a lower-triangular (look-ahead) attention mask, so the hidden state at
sub-word t only attends to sub-words <= t. ``tests/test_causal.py`` checks this.

Two parallel linear heads read the hidden state at each position:
  * switch head   -> 2 classes  (will word t+1 be in a different language?)
  * duration head -> 3 classes  (length bin of the upcoming segment: 1-2 / 3-6 / 7+ words)

Loss (Section III-B):  L = lambda_sw * CE(switch) + lambda_dur * CE(duration),
both with ignore_index=-1, so duration is learned only where a switch happens.
"""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

MODELS = {
    "xlmr": "FacebookAI/xlm-roberta-base",
    "mbert": "google-bert/bert-base-multilingual-cased",
}
IGNORE = -1


class CausalDualHead(nn.Module):
    def __init__(self, model_name=None, config=None, dropout=0.1):
        super().__init__()
        if config is None:
            config = AutoConfig.from_pretrained(model_name)
        config.is_decoder = True            # causal (look-ahead) self-attention mask
        config.add_cross_attention = False  # no encoder-decoder attention, just causal self-attention
        if model_name is not None:
            self.encoder = AutoModel.from_pretrained(model_name, config=config, add_pooling_layer=False)
        else:  # random init, used by the unit test
            self.encoder = AutoModel.from_config(config, add_pooling_layer=False)
        h = config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.switch_head = nn.Linear(h, 2)
        self.duration_head = nn.Linear(h, 3)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        hs = self.dropout(out.last_hidden_state)
        return self.switch_head(hs), self.duration_head(hs)


def multitask_loss(sw_logits, dur_logits, sw_labels, dur_labels,
                   lambda_sw=1.0, lambda_dur=1.0, sw_class_weight=None):
    ce_sw = nn.functional.cross_entropy(sw_logits.reshape(-1, 2).float(), sw_labels.reshape(-1),
                                        weight=sw_class_weight, ignore_index=IGNORE)
    if (dur_labels != IGNORE).any():
        ce_dur = nn.functional.cross_entropy(dur_logits.reshape(-1, 3).float(), dur_labels.reshape(-1),
                                             ignore_index=IGNORE)
    else:  # batch without any switch: duration loss is 0 but keep the graph connected
        ce_dur = dur_logits.sum() * 0.0
    return lambda_sw * ce_sw + lambda_dur * ce_dur, ce_sw.detach(), ce_dur.detach()
