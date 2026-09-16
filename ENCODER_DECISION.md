# Encoder decision — Omnilingual (omniASR_W2V) vs FastConformer / IndicConformer

**Reality check (2026), measured facts separated from speculation. Verdict at the end.**

## 1. Newest measured real-world Hindi WER (spontaneous, not scripted)

**Vaani Benchmark V1.0** — 20.64 h spontaneous Hindi, 104 districts / 22 states
([arXiv 2606.21408](https://arxiv.org/abs/2606.21408)):

| Model | WER |
|---|---|
| Vaani FastConformer | **10.6** |
| IndicConformer-600m-multilingual | 14.2 |
| **OmniASR_LLM_1B** | **26.4** |
| Whisper-large-v3 | 27.1 |

**Voice of India** — large-scale real-world benchmark
([arXiv 2604.19151](https://arxiv.org/abs/2604.19151)):

| Model | Hindi WER |
|---|---|
| Sarvam Audio (commercial) | 5.0 |
| Gemini 3 Pro | 6.0 |
| Amazon Transcribe | 6.8 |
| ElevenLabs Scribe v2 | 7.7 |
| **IndicConformer** (open; lowest WER in 13/15 languages) | **8.2** |
| **OmniASR_LLM_7B** | **13.7** |
| **OmniASR_LLM_1B** | **14.9** |

→ **Measured:** across two independent 2026 benchmarks, Omnilingual's Hindi WER is roughly
**2× worse** than IndicConformer/FastConformer (13.7–26.4 vs 8.2–14.2).

## 2. Has fine-tuning/distillation narrowed the gap?
**No published evidence** of a fine-tuned omniASR SSL encoder beating FastConformer/
IndicConformer on Hindi. *(Speculation: fine-tuning our two-stage recipe could help, but there
is no measured result to lean on — so it is a bet, not a fact.)*

## 3. New Omni releases/checkpoints
omniASR-LLM-1B/7B checkpoints are mirrored on HF; the corpus adds 348 under-served languages
across Latin/Arabic/**Devanagari** scripts. **Measured:** no Hindi-specialized Omni checkpoint.

## 4. Latency / streaming
**Measured:** the Omni encoder is non-causal (offline); no official streaming mode — we would
retrofit causality (a known WER risk). FastConformer has **native cache-aware streaming** with
selectable latency (0 / 80 / 480 / 1040 ms) and ready Hindi streaming checkpoints
([salesken/Hindi-FastConformer-Streaming-ASR](https://huggingface.co/salesken/Hindi-FastConformer-Streaming-ASR),
NVIDIA NeMo). → FastConformer is decisively better for our streaming, floor-control use case.

## 5. Long-tail coverage only Omni offers
**Measured:** Omni uniquely covers 1,600+ languages incl. hundreds of ultra-low-resource ones
(some Devanagari-script). This is Omni's **only** clear edge — and it is irrelevant to a
Hindi-first product.

---

## Verdict
**Since the original plan (Sept 2026), the case for betting on Omnilingual for Indic production
has WORSENED:** two independent 2026 real-world benchmarks now show its Hindi WER is about twice
that of IndicConformer/FastConformer, and FastConformer additionally ships native Hindi streaming
that Omni lacks.

## What we do
Our architecture is **encoder-agnostic** (the projector adapts any encoder's dim, and our sibling
`kupe-asr-en` already runs FastConformer+Nandi), so this is a **config switch**, not a rewrite:
- **Default (recommended):** `base.encoder_type: fastconformer`,
  `encoder_id: ai4bharat/indic-conformer-600m-multilingual`.
- **Keep Omni as an option** (`encoder_type: omni_w2v`) only for the future long-tail-language
  expansion, where its 1,600-language coverage is the point.

Sources: [Vaani 2606.21408](https://arxiv.org/abs/2606.21408) ·
[Voice of India 2604.19151](https://arxiv.org/abs/2604.19151) ·
[Omnilingual ASR 2511.09690](https://arxiv.org/abs/2511.09690) ·
[Hindi FastConformer streaming](https://huggingface.co/salesken/Hindi-FastConformer-Streaming-ASR) ·
[IndicConformer](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual).
