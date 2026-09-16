"""KupeFDX-hi-asr — streaming Hindi ASR + floor-control on omniASR_W2V + Nandi-Mini-150M.

Package layout (mirrors kupe-asr-en conventions):
  config / env / ledger / hub     infra: config load, HF auth, resumable state, sync
  constants / vocab / text        static contracts: geometry, special tokens, Devanagari norm
  audio                           waveform IO + resample
  encoders                        omniASR_W2V (real) + TinyEncoder (smoke); causal chunk mask
  ctc_head                        Devanagari CTC head (kept for the whole project)
  frontend / quantizer            continuous projector + discrete audio-token quantizer
  decoders / tokens               Nandi loader (real) + TinyNandi (smoke); safe token extension
  model                           KupeFDXModel: encoder -> CTC + projector -> Nandi
  dataset / collate               manifest dataset + batch assembly
  train / evaluate / stream       training loop (resume), WER+FC eval, streaming inference
  smoke                           synthetic end-to-end test (runs on CPU/MPS in seconds)
"""
__version__ = "0.0.1"
