#!/usr/bin/env python3
"""Test Nandi's tokenizer on Hindi + measure FERTILITY (subword tokens per word).

Fertility = mean tokens / word. Lower is better (a tokenizer that natively covers
Devanagari fragments Hindi less, which means shorter sequences, faster decoding, and more
of Nandi's 2048-token context left for audio + history). We also report chars/token and
bytes/token. Uses the real Nandi tokenizer when available, else the TinyTokenizer.

    python scripts/08_tokenizer_fertility.py --config configs/gpu.yaml
    python scripts/08_tokenizer_fertility.py --config configs/smoke.yaml   # TinyTokenizer
"""
import argparse
import statistics

import _bootstrap  # noqa: F401
from kupefdx.config import load_config
from kupefdx.env import log
from kupefdx.text import normalize

# a small representative Hindi set spanning domains (general, medical, technical).
SAMPLES = [
    "नमस्ते, आप कैसे हैं और आपका दिन कैसा जा रहा है?",
    "मरीज़ को उच्च रक्तचाप और मधुमेह की शिकायत है।",
    "कृपया सर्वर को पुनः प्रारंभ करें और लॉग फ़ाइल जाँचें।",
    "मुझे थोड़ी दवा चाहिए क्योंकि सिर में दर्द हो रहा है।",
    "आपका खाता सफलतापूर्वक अपडेट कर दिया गया है, धन्यवाद।",
    "इंजन का तापमान सामान्य से अधिक है, तुरंत रोकिए।",
]


def load_tok(cfg):
    if cfg.backend == "real":
        from kupefdx.decoders import load_tokenizer
        return load_tokenizer(cfg.base.decoder_id, cfg.base.trust_remote_code), "Nandi"
    from kupefdx.tokens import TinyTokenizer
    return TinyTokenizer(), "TinyTokenizer(char)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpu.yaml")
    a = ap.parse_args()
    cfg = load_config(a.config)
    tok, name = load_tok(cfg)
    log.info("tokenizer: %s | vocab=%d", name, len(tok))

    fert, cpt, bpt = [], [], []
    for s in SAMPLES:
        s = normalize(s)
        ids = tok(s, add_special_tokens=False).input_ids
        n_words = len(s.split())
        n_tok = len(ids)
        fert.append(n_tok / max(n_words, 1))
        cpt.append(len(s.replace(" ", "")) / max(n_tok, 1))
        bpt.append(len(s.encode("utf-8")) / max(n_tok, 1))
        log.info("  words=%2d tokens=%3d fertility=%.2f  | %s", n_words, n_tok,
                 n_tok / max(n_words, 1), s[:40])

    log.info("=== FERTILITY (tokens/word): mean=%.2f  min=%.2f  max=%.2f ===",
             statistics.mean(fert), min(fert), max(fert))
    log.info("chars/token mean=%.2f | bytes/token mean=%.2f", statistics.mean(cpt),
             statistics.mean(bpt))
    log.info("interpretation: lower fertility -> shorter target sequences, more of Nandi's "
             "2048-token budget free for audio soft-prompts + conversation context.")


if __name__ == "__main__":
    main()
