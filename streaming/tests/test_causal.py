"""Leakage test for the causal mask (deliverable: causal masking implementation).

For a tiny randomly initialised XLM-R and mBERT, change every token AFTER
position t and check that the outputs at positions <= t do not change.
Also checks the opposite: with is_decoder=False (normal bidirectional
encoder) the same perturbation DOES change earlier outputs, so the test
would catch a mask that silently stopped working.

  python streaming/tests/test_causal.py
"""
import sys
from pathlib import Path

import torch
from transformers import BertConfig, XLMRobertaConfig, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model import CausalDualHead  # noqa: E402

TINY = dict(hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64,
            vocab_size=500, max_position_embeddings=80)


def configs():
    yield "xlmr", XLMRobertaConfig(pad_token_id=1, **TINY)
    yield "mbert", BertConfig(pad_token_id=0, **TINY)


def perturb_after(ids, t, gen):
    new = ids.clone()
    new[:, t + 1:] = torch.randint(5, 500, new[:, t + 1:].shape, generator=gen)
    return new


def check(name, cfg):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    model = CausalDualHead(config=cfg).eval()
    L = 40
    ids = torch.randint(5, 500, (3, L), generator=gen)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        base_sw, base_dur = model(ids, mask)
        for t in [0, 7, 20, L - 2]:
            sw, dur = model(perturb_after(ids, t, gen), mask)
            assert torch.allclose(sw[:, :t + 1], base_sw[:, :t + 1], atol=1e-5), f"{name}: leak at t={t}"
            assert torch.allclose(dur[:, :t + 1], base_dur[:, :t + 1], atol=1e-5), f"{name}: leak at t={t}"
        # right padding must not change real positions either
        padded = torch.cat([ids, torch.full((3, 5), cfg.pad_token_id)], 1)
        pmask = torch.cat([mask, torch.zeros(3, 5, dtype=torch.long)], 1)
        sw_p, _ = model(padded, pmask)
        assert torch.allclose(sw_p[:, :L], base_sw, atol=1e-5), f"{name}: padding changes outputs"

    # control: a bidirectional encoder must FAIL the same check
    cfg_bi = type(cfg)(**{**cfg.to_dict(), "is_decoder": False})
    enc = AutoModel.from_config(cfg_bi, add_pooling_layer=False).eval()
    with torch.no_grad():
        a = enc(input_ids=ids, attention_mask=mask).last_hidden_state
        b = enc(input_ids=perturb_after(ids, 7, gen), attention_mask=mask).last_hidden_state
    assert not torch.allclose(a[:, :8], b[:, :8], atol=1e-5), f"{name}: control failed (test is blind)"
    print(f"{name}: causal mask OK (no leak at any t; bidirectional control leaks as expected)")


if __name__ == "__main__":
    for n, c in configs():
        check(n, c)
    print("all causal-mask tests passed")
