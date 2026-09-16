"""The set of special tokens we add to Nandi's BPE vocab.

Two families:
  * control tokens (audio markers, history markers, floor-control signals) — a small
    fixed set, always added.
  * discrete audio-code tokens <aud_0>..<aud_{N-1}> — added only when the discrete
    audio-token branch is enabled (config: audio.n_codes > 0). These are what let
    Nandi "understand the omni encoder's audio tokens" in its own embedding space.

`build_special_tokens(n_codes)` returns the ordered list; `tokens.py` extends Nandi's
factorized+tied embedding by exactly this many rows.
"""
from __future__ import annotations

from .constants import (AUDIO_CODE_FMT, FC_TOKENS, TOK_AUDIO_BOS, TOK_AUDIO_EOS,
                        TOK_HIST_BOS, TOK_HIST_EOS)

CONTROL_TOKENS = [TOK_AUDIO_BOS, TOK_AUDIO_EOS, TOK_HIST_BOS, TOK_HIST_EOS] + FC_TOKENS


def audio_code_tokens(n_codes: int) -> list[str]:
    return [AUDIO_CODE_FMT.format(k=k) for k in range(int(n_codes))]


def build_special_tokens(n_codes: int = 0) -> list[str]:
    """Ordered, de-duplicated list of new tokens to append to Nandi's vocab."""
    toks = list(CONTROL_TOKENS)
    toks += audio_code_tokens(n_codes)
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def domain_tag(name: str) -> str:
    return f"<domain={name}>"
