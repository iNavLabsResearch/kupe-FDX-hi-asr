"""The riskiest GPU-side path: extending Nandi's factorized + tied embedding for our
new tokens. Validated here on TinyNandi (same structure) and, if reachable, real Nandi.

Run: python tests/test_token_extension.py   (or pytest tests -q)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import torch

from kupefdx.decoders import TinyNandi
from kupefdx.tokens import TinyTokenizer, extend_vocab
from kupefdx.vocab import build_special_tokens


def test_tiny_extension():
    tok = TinyTokenizer()
    dec = TinyNandi(vocab=len(tok), hidden=64, rank=16, layers=2)
    v0 = dec.get_input_embeddings().weight.shape[0]
    specials = build_special_tokens(n_codes=16)
    ids = extend_vocab(dec, tok, specials)

    # every new token has a unique id beyond the old vocab, and the table grew to match.
    assert len(ids) == len(specials)
    assert dec.get_input_embeddings().weight.shape[0] == len(tok)
    assert dec.get_input_embeddings().weight.shape[0] == v0 + len(specials)
    assert all(i >= v0 for i in ids.values())

    # a forward pass over the NEW ids works and the tied head covers them.
    new_ids = torch.tensor([[ids["<EOS_SPEECH>"], ids["<aud_0>"], ids["<aud_15>"]]])
    emb = dec.get_input_embeddings()(new_ids)
    assert emb.shape[-1] == dec.config.hidden_size          # composes to hidden, not rank
    out = dec(input_ids=new_ids)
    assert out.logits.shape[-1] == len(tok)                 # head width tracks vocab
    print("TinyNandi factorized+tied extension: OK (+%d tokens)" % len(specials))


def test_real_nandi_extension_if_available():
    """Best-effort: only runs if real Nandi + network are available. Never fails the
    suite when offline — the Tiny mirror already validates the logic."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        mid = "FrontiersMind/Nandi-Mini-150M"
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        dec = AutoModelForCausalLM.from_pretrained(mid, trust_remote_code=True)
    except Exception as e:
        print("skip real-Nandi test (offline / unavailable):", e)
        return
    v0 = dec.get_input_embeddings().weight.shape[0]
    specials = build_special_tokens(n_codes=8)
    ids = extend_vocab(dec, tok, specials)
    assert dec.get_input_embeddings().weight.shape[0] == v0 + len(specials)
    print("real Nandi extension: OK")


if __name__ == "__main__":
    test_tiny_extension()
    test_real_nandi_extension_if_available()
    print("ALL TOKEN-EXTENSION TESTS PASSED")
