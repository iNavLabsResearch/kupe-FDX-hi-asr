"""Decoder SLM = Nandi-Mini-150M, plus TinyNandi (a faithful structural mirror).

Both expose the SAME minimal contract the KupeFDX wrapper needs:
  .config.hidden_size            int
  .get_input_embeddings()        module with .weight [vocab, rank] (factorized) and
                                 forward(ids)->[..,hidden]; resizable
  forward(inputs_embeds=, attention_mask=, labels=) -> obj with .loss and .logits
  .resize_token_embeddings(n)    extend vocab (factorized + tied safe)

Why TinyNandi mirrors factorized+tied+layer-sharing exactly: the riskiest GPU-side
code is extending Nandi's factorized/tied embedding for our new tokens. The smoke
test must exercise that same path on the Mac. tokens.py::extend_vocab therefore runs
identically on TinyNandi and real Nandi; a unit test also tries real Nandi if present.
"""
from __future__ import annotations

import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (NANDI_EMBED_RANK, NANDI_HIDDEN, NANDI_ID, NANDI_VOCAB)
from .env import hf_token, log

_DTYPES = {"float32": torch.float32, "fp32": torch.float32,
           "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
           "float16": torch.float16, "fp16": torch.float16}


# ----------------------------------------------------------------- factorized embed
class FactorizedEmbedding(nn.Module):
    """Mirror of Nandi's factorized embedding: a low-rank table [vocab, rank] and a
    shared [rank, hidden] projection. `.weight` exposes the table so generic
    resize/extension logic (and HF-style code) sees a [vocab, rank] parameter."""

    def __init__(self, vocab: int, rank: int, hidden: int):
        super().__init__()
        self.table = nn.Parameter(torch.empty(vocab, rank))
        self.proj = nn.Parameter(torch.empty(rank, hidden))
        nn.init.normal_(self.table, std=0.02)
        nn.init.normal_(self.proj, std=hidden ** -0.5)

    @property
    def weight(self) -> torch.Tensor:            # HF-compatible: the [vocab, rank] table
        return self.table

    @property
    def num_embeddings(self) -> int:
        return self.table.shape[0]

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.table) @ self.proj      # [.., hidden]


class TinyNandi(nn.Module):
    """Small causal LM with factorized tied embeddings + layer sharing, matching the
    real Nandi's structural knobs so wiring/token-extension is validated on CPU/MPS."""

    def __init__(self, vocab=512, hidden=64, rank=16, layers=2, heads=4,
                 layer_sharing_repeats=2, max_pos=2048):
        super().__init__()
        self.config = types.SimpleNamespace(
            hidden_size=hidden, vocab_size=vocab, embedding_rank=rank,
            num_hidden_layers=layers, layer_sharing_repeats=layer_sharing_repeats,
            max_position_embeddings=max_pos, factorized_embedding=True,
            tie_word_embeddings=True, model_type="tiny-nandi")
        self.embed = FactorizedEmbedding(vocab, rank, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_pos, hidden))
        nn.init.normal_(self.pos, std=0.02)
        layer = lambda: nn.TransformerEncoderLayer(hidden, heads, hidden * 4,
                                                   batch_first=True, activation="gelu",
                                                   norm_first=True)
        self.layers = nn.ModuleList([layer() for _ in range(layers)])
        self.repeats = int(layer_sharing_repeats)
        self.norm = nn.LayerNorm(hidden)
        self.gradient_checkpointing = False

    # ---- HF-compatible embedding accessors ----
    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, mod):
        self.embed = mod

    def resize_token_embeddings(self, new_num: int):
        old = self.embed.table
        v, r = old.shape
        if new_num == v:
            return self.embed
        new = nn.Parameter(torch.empty(new_num, r, dtype=old.dtype, device=old.device))
        with torch.no_grad():
            nn.init.normal_(new, std=0.02)
            n = min(v, new_num)
            new[:n] = old[:n]
        self.embed.table = new
        self.config.vocab_size = new_num
        return self.embed

    def gradient_checkpointing_enable(self, **kw):
        self.gradient_checkpointing = True

    def _causal_mask(self, L, device, dtype):
        m = torch.full((L, L), float("-inf"), device=device, dtype=dtype)
        return torch.triu(m, diagonal=1)

    def forward(self, inputs_embeds=None, attention_mask=None, labels=None, input_ids=None):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        x = inputs_embeds + self.pos[:, : inputs_embeds.shape[1]].to(inputs_embeds.dtype)
        L = x.shape[1]
        mask = self._causal_mask(L, x.device, x.dtype)
        kpm = (attention_mask == 0) if attention_mask is not None else None
        for _ in range(self.repeats):
            for layer in self.layers:
                x = layer(x, src_mask=mask, src_key_padding_mask=kpm)
        x = self.norm(x)
        # tied factorized head: hidden -> rank (proj^T) -> vocab (table^T)
        rank_h = x @ self.embed.proj.t()                        # [B,L,rank]
        logits = rank_h @ self.embed.table.t()                  # [B,L,vocab]
        loss = None
        if labels is not None:
            sl = logits[:, :-1, :].contiguous()
            tl = labels[:, 1:].contiguous()
            loss = F.cross_entropy(sl.view(-1, sl.shape[-1]), tl.view(-1),
                                   ignore_index=-100)
        return types.SimpleNamespace(loss=loss, logits=logits)


def load_tokenizer(model_id: str = NANDI_ID, trust_remote_code: bool = True):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code,
                                        token=hf_token())
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or tok.bos_token
    return tok


def load_decoder(cfg, dtype=torch.float32):
    """Return (decoder, tokenizer). backend=='real' loads Nandi; else TinyNandi + a
    tiny byte-level tokenizer built from the CTC charset is used by the caller."""
    if cfg.backend == "real":
        from transformers import AutoModelForCausalLM
        tok = load_tokenizer(cfg.base.decoder_id, cfg.base.trust_remote_code)
        try:
            dec = AutoModelForCausalLM.from_pretrained(
                cfg.base.decoder_id, trust_remote_code=cfg.base.trust_remote_code,
                dtype=dtype, token=hf_token())
        except TypeError:
            dec = AutoModelForCausalLM.from_pretrained(
                cfg.base.decoder_id, trust_remote_code=cfg.base.trust_remote_code,
                torch_dtype=dtype, token=hf_token())
        log.info("Nandi loaded | hidden=%d vocab=%d", dec.config.hidden_size,
                 dec.config.vocab_size)
        return dec, tok
    from .tokens import TinyTokenizer
    tok = TinyTokenizer()
    dec = TinyNandi(vocab=len(tok),
                    hidden=int(getattr(cfg.base, "tiny_hidden", 64)),
                    rank=int(getattr(cfg.base, "tiny_rank", 16)),
                    layers=int(getattr(cfg.base, "tiny_layers", 2))).to(dtype)
    return dec, tok
