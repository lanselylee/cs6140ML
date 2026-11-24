"""Trainable causal XLM-R backbone with next-switch and duration heads."""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel
from huggingface_hub import snapshot_download

MODEL_ID = 'FacebookAI/xlm-roberta-base'


def local_snapshot(model_id=MODEL_ID):
    # Load via a filesystem path to bypass tokenizer helper metadata requests.
    from pathlib import Path
    if Path(model_id).is_dir():
        return str(Path(model_id).resolve())
    return snapshot_download(model_id, local_files_only=True, token=False)


class CausalMultitask(nn.Module):
    def __init__(self, model_id=MODEL_ID, local_only=True, checkpointing=True):
        super().__init__()
        if local_only:
            model_id = local_snapshot(model_id)
        config = AutoConfig.from_pretrained(model_id, local_files_only=local_only, token=False)
        # Decoder-mode self-attention, WITHOUT adding cross-attention or an LM head.
        config.is_decoder = True
        config.add_cross_attention = False
        config.use_cache = False
        self.backbone = AutoModel.from_pretrained(
            model_id, config=config, add_pooling_layer=False,
            local_files_only=local_only, token=False, use_safetensors=True,
            attn_implementation='sdpa')
        if checkpointing:
            self.backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False})
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.switch_head = nn.Linear(config.hidden_size, 2)
        self.duration_head = nn.Linear(config.hidden_size, 3)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                               use_cache=False).last_hidden_state
        hidden = self.dropout(hidden)
        return self.switch_head(hidden), self.duration_head(hidden)

    @staticmethod
    def loss(switch_logits, duration_logits, switch_labels, duration_labels,
             duration_weight=1.0):
        """Labels align with position t but describe t+1. -1 marks ignored positions."""
        if ((duration_labels >= 0) & (switch_labels != 1)).any():
            raise ValueError('Duration targets must be masked outside true switches')
        def masked_ce(logits, labels):
            keep = labels >= 0
            # All-negative batches have zero duration loss, not NaN.
            return F.cross_entropy(logits[keep].float(), labels[keep]) if keep.any() else logits.sum() * 0.0
        sw = masked_ce(switch_logits, switch_labels)
        dur = masked_ce(duration_logits, duration_labels)
        return sw + duration_weight * dur, sw, dur
