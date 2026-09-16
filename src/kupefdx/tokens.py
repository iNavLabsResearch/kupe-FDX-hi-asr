"""Safe vocab extension for Nandi's factorized + tied embedding, and a TinyTokenizer
for the smoke path.

extend_vocab() adds our control + audio-code tokens and grows the model's embedding by
exactly that many rows. It works on BOTH real Nandi (HF tokenizer + model) and TinyNandi
(+ TinyTokenizer): both expose add_special_tokens / resize_token_embeddings / a [V, rank]
input-embedding weight, so the risky GPU-side path is exercised on the Mac unchanged.

New rows are initialised to the mean of existing rows + small noise (a well-known trick
that keeps a freshly added token from dominating early training).
"""
from __future__ import annotations

import torch

from .env import log
from .text import CharTokenizer


def extend_vocab(model, tokenizer, special_tokens: list[str]) -> dict:
    """Add `special_tokens`; grow embeddings; return {token: id} for the new tokens."""
    if not special_tokens:
        return {}
    before = len(tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    after = len(tokenizer)
    n_new = after - before
    if n_new == 0:
        return {t: tokenizer.convert_tokens_to_ids(t) for t in special_tokens}

    emb = model.get_input_embeddings()
    old_rows = emb.weight.shape[0]
    model.resize_token_embeddings(after)
    emb = model.get_input_embeddings()
    with torch.no_grad():
        w = emb.weight
        mean = w[:old_rows].mean(0, keepdim=True)
        std = float(w[:old_rows].std()) * 0.1 + 1e-4
        w[old_rows:] = mean + std * torch.randn_like(w[old_rows:])
    log.info("extended vocab %d -> %d (+%d tokens); embed table now %s",
             before, after, n_new, tuple(emb.weight.shape))
    return {t: tokenizer.convert_tokens_to_ids(t) for t in special_tokens}


class TinyTokenizer:
    """Char-level tokenizer over the Devanagari CTC charset for the smoke path.
    Mirrors the HF tokenizer surface KupeFDX relies on."""

    def __init__(self):
        chars = CharTokenizer().chars
        self.eos_token_id = 0
        self.bos_token_id = 1
        self.pad_token_id = 2
        self.eos_token, self.bos_token, self.pad_token = "</s>", "<s>", "<pad>"
        self._id2tok = {0: "</s>", 1: "<s>", 2: "<pad>"}
        self._tok2id = {v: k for k, v in self._id2tok.items()}
        for c in chars:
            i = len(self._id2tok)
            self._id2tok[i] = c
            self._tok2id[c] = i
        self.added: dict[str, int] = {}

    def __len__(self):
        return len(self._id2tok)

    def add_special_tokens(self, d: dict) -> int:
        n = 0
        for t in d.get("additional_special_tokens", []):
            if t in self._tok2id:
                continue
            i = len(self._id2tok)
            self._id2tok[i] = t
            self._tok2id[t] = i
            self.added[t] = i
            n += 1
        return n

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self._tok2id.get(tok, self.pad_token_id)

    def __call__(self, text: str, add_special_tokens: bool = False):
        # greedily match multi-char special tokens (e.g. "<EOS_SPEECH>") inline, else chars.
        longs = sorted((t for t in self._tok2id if len(t) > 1), key=len, reverse=True)
        ids, i = [], 0
        while i < len(text):
            for tk in longs:
                if text.startswith(tk, i):
                    ids.append(self._tok2id[tk])
                    i += len(tk)
                    break
            else:
                c = text[i]
                if c in self._tok2id:
                    ids.append(self._tok2id[c])
                i += 1
        return type("Enc", (), {"input_ids": ids})()

    def encode(self, text: str) -> list[int]:
        return self(text).input_ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        specials = {0, 1, 2} | set(self.added.values()) if skip_special_tokens else set()
        return "".join(self._id2tok.get(int(i), "") for i in ids if int(i) not in specials)

    def save_pretrained(self, out_dir: str):
        import json
        import os
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "tiny_tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump({"id2tok": self._id2tok, "added": self.added}, f, ensure_ascii=False)
