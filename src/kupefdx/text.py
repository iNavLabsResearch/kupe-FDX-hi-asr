"""Hindi/Devanagari text normalization + the CTC character vocabulary.

Normalization is conservative and reversible-ish: NFC unicode, whitespace collapse,
strip zero-width joiners that don't change rendering, map fancy punctuation to plain,
and (optionally) drop Latin/emoji noise. We do NOT transliterate or fold nukta forms
away — that would change words. WER is computed on this normalized form for both ref
and hypothesis so scoring is fair.
"""
from __future__ import annotations

import re
import unicodedata

# Devanagari block + Hindi digits, common punctuation and space.
_DEVANAGARI = "".join(chr(c) for c in range(0x0900, 0x0980))
_PUNCT = " ।?!,.-‍"          # space, danda, latin punct, ZWJ (kept: it matters in conjuncts)
_ALLOWED = set(_DEVANAGARI + _PUNCT)

_ZW_STRIP = dict.fromkeys(map(ord, "​‎‏﻿"), None)  # zero-width non-joiners/marks
_PUNCT_MAP = {"“": '"', "”": '"', "’": "'", "‘": "'", "—": "-", "–": "-", "…": "...",
              "।": "।"}       # normalize danda variants


def normalize(text: str, *, keep_punct: bool = True) -> str:
    if text is None:
        return ""
    t = unicodedata.normalize("NFC", str(text))
    t = t.translate(_ZW_STRIP)
    for a, b in _PUNCT_MAP.items():
        t = t.replace(a, b)
    if not keep_punct:
        t = re.sub(r"[।?!,.\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def devanagari_ratio(text: str) -> float:
    """Fraction of letters that are Devanagari (vs Latin). 1.0 = pure Devanagari, low =
    romanized. Used to reject romanized Hindi like 'kaise ho' in data-quality checks."""
    dev = sum(1 for c in text if "ऀ" <= c <= "ॿ")
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    tot = dev + lat
    return dev / tot if tot else 1.0


def ctc_charset(extra: str = "") -> list[str]:
    """Ordered CTC symbol list; index 0 is reserved for the CTC blank in ctc_head.
    Includes Devanagari letters + digits + space + a few punctuation marks."""
    base = sorted({c for c in (_DEVANAGARI + " ।-" + extra)
                   if unicodedata.category(c)[0] in ("L", "M", "N") or c in " ।-"})
    return base


class CharTokenizer:
    """Char <-> id for the CTC head. id 0 == blank (owned by the CTC loss)."""

    def __init__(self, charset: list[str] | None = None):
        self.chars = charset or ctc_charset()
        self.blank = 0
        self.stoi = {c: i + 1 for i, c in enumerate(self.chars)}   # +1: 0 is blank
        self.itos = {i + 1: c for i, c in enumerate(self.chars)}
        self.vocab_size = len(self.chars) + 1

    def encode(self, text: str) -> list[int]:
        text = normalize(text)
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids if i != self.blank)
