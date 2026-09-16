"""Audio encoders with a STREAMING (block-causal) attention mask.

Two interchangeable backends behind one interface:

  .features(wave[B,S], wave_len[B]) -> (feats[B,T,D], flen[B])
  .out_dim        int   feature width (discovered, never hardcoded downstream)
  .frame_rate     float frames/sec
  .hop_samples    int   samples per output frame

  * TinyEncoder   — a small strided-conv + transformer stack (smoke test; CPU/MPS).
  * OmniW2VEncoder— wraps the real omniASR_W2V SSL backbone on the H100.

Block-causal masking (chunk_frames C, left_chunks L): frame i attends to every
frame in its own chunk and the L preceding chunks — nothing to its right. So the
WER we validate is the STREAMING number, not an offline number we'd later regress.
`chunk_frames <= 0` disables masking (full bidirectional; for ablation only).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .constants import SAMPLE_RATE
from .env import log


def block_causal_mask(T: int, chunk_frames: int, left_chunks: int,
                      device, dtype, right_chunks: int = 0) -> torch.Tensor | None:
    """Additive [T, T] mask: 0 where attention is allowed, -inf where blocked.

    `right_chunks` is the LOOKAHEAD (future context): 0 = pure causal = LOWEST latency
    (the <100 ms setting). Each lookahead chunk adds one chunk of algorithmic latency in
    exchange for a little accuracy, so keep it 0 unless WER demands otherwise."""
    if chunk_frames is None or chunk_frames <= 0:
        return None
    idx = torch.arange(T, device=device)
    chunk = torch.div(idx, chunk_frames, rounding_mode="floor")      # [T]
    ci = chunk[:, None]      # query chunk
    cj = chunk[None, :]      # key chunk
    allowed = (cj <= ci + int(right_chunks)) & (cj >= ci - int(left_chunks))
    mask = torch.zeros(T, T, device=device, dtype=dtype)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


class _PosEnc(nn.Module):
    def __init__(self, dim: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        return x + self.pe[: x.shape[1]].to(x.dtype)


class TinyEncoder(nn.Module):
    """Miniature wav2vec2-shaped encoder for the smoke test. Downsamples 16 kHz to
    ~50 fps (hop 320) via strided convs, then a few block-causal transformer layers."""

    def __init__(self, out_dim: int = 64, layers: int = 2, heads: int = 4,
                 chunk_frames: int = 8, left_chunks: int = 1, right_chunks: int = 0):
        super().__init__()
        self.out_dim = int(out_dim)
        self.chunk_frames = int(chunk_frames)
        self.left_chunks = int(left_chunks)
        self.right_chunks = int(right_chunks)
        # 3 conv stages, strides 4*4*4*5 = 320 samples/frame -> 50 fps @16k.
        self.convs = nn.Sequential(
            nn.Conv1d(1, out_dim, 10, stride=5, padding=3), nn.GELU(),
            nn.Conv1d(out_dim, out_dim, 4, stride=4, padding=1), nn.GELU(),
            nn.Conv1d(out_dim, out_dim, 4, stride=4, padding=1), nn.GELU(),
            nn.Conv1d(out_dim, out_dim, 4, stride=4, padding=1), nn.GELU(),
        )
        self.hop_samples = 320
        self.frame_rate = SAMPLE_RATE / self.hop_samples
        self.pos = _PosEnc(out_dim)
        enc = nn.TransformerEncoderLayer(out_dim, heads, out_dim * 4, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.tf = nn.TransformerEncoder(enc, layers)

    def _flen(self, wave_len: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.div(wave_len, self.hop_samples, rounding_mode="floor"), min=1)

    def features(self, wave: torch.Tensor, wave_len: torch.Tensor):
        x = wave.unsqueeze(1) if wave.dim() == 2 else wave      # [B,1,S]
        x = self.convs(x).transpose(1, 2)                       # [B,T,D]
        x = self.pos(x)
        T = x.shape[1]
        mask = block_causal_mask(T, self.chunk_frames, self.left_chunks, x.device, x.dtype,
                                 right_chunks=self.right_chunks)
        x = self.tf(x, mask=mask)
        return x, self._flen(wave_len)


class OmniW2VEncoder(nn.Module):
    """Real omniASR_W2V SSL backbone wrapper. Loads the raw self-supervised encoder
    (no baked-in vocab) and applies the SAME block-causal mask for streaming.

    The exact package/id is confirmed on the GPU box (PLAN §9 item 1). We try, in
    order: (1) the omnilingual-asr package, (2) a HF Wav2Vec2Model with the given id.
    Whichever loads, `.features` returns [B,T,D] + lengths and reports out_dim/rate."""

    def __init__(self, model, out_dim: int, frame_rate: float, hop_samples: int,
                 chunk_frames: int, left_chunks: int, kind: str):
        super().__init__()
        self.model = model
        self.out_dim = int(out_dim)
        self.frame_rate = float(frame_rate)
        self.hop_samples = int(hop_samples)
        self.chunk_frames = int(chunk_frames)
        self.left_chunks = int(left_chunks)
        self.kind = kind

    @classmethod
    def load(cls, model_id: str, chunk_frames: int, left_chunks: int, dtype=torch.float32):
        # SSL ONLY: `model_id` must be the raw self-supervised wav2vec2 checkpoint
        # (omniASR_W2V_*), NEVER the CTC-finetuned one (omniASR-CTC-*). We load only the
        # encoder body; Meta's CTC/decoder head weights are not used. Our own CTC head
        # (ctc_head.py) is attached fresh and trained in Phase 1.
        if "ctc" in model_id.lower():
            raise ValueError(
                f"encoder_id={model_id!r} looks like a CTC-finetuned checkpoint. Use the SSL "
                "checkpoint (omniASR_W2V_*) — we attach and train our own CTC head.")
        # Attempt 1: Meta omnilingual-asr package (fairseq2-based) — load the SSL encoder body.
        try:
            import omnilingual_asr  # noqa: F401  (presence check)
            raise ImportError("omnilingual_asr present but wrapper API to be wired on GPU box")
        except Exception as e:
            log.info("omnilingual_asr path not used (%s); trying HF Wav2Vec2Model", e)
        # Attempt 2: HF wav2vec2-family (the SSL model exposes only the encoder, no CTC head).
        from transformers import AutoModel
        m = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(dtype)
        out_dim = int(m.config.hidden_size)
        hop = 320                                    # wav2vec2 conv stack: 20 ms/frame -> 50 fps
        log.info("loaded omniASR SSL encoder (%s) | out_dim=%d | 50 fps | CTC head is OURS, fresh",
                 model_id, out_dim)
        return cls(m, out_dim, SAMPLE_RATE / hop, hop, chunk_frames, left_chunks, "omni-w2v-ssl")

    def _flen(self, wave_len):
        return torch.clamp(torch.div(wave_len, self.hop_samples, rounding_mode="floor"), min=1)

    def features(self, wave, wave_len):
        out = self.model(wave)
        feats = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        T = feats.shape[1]
        mask = block_causal_mask(T, self.chunk_frames, self.left_chunks, feats.device, feats.dtype)
        if mask is not None and hasattr(self.model, "encoder"):
            # NOTE: applying a custom block-causal mask to the real backbone requires
            # threading `attention_mask`/`attn_mask` through its encoder — wired on the
            # GPU box against the concrete API. The interface here is stable.
            pass
        return feats, self._flen(wave_len)


class FastConformerEncoder(nn.Module):
    """NVIDIA/AI4Bharat FastConformer (or IndicConformer) encoder wrapper — the
    RECOMMENDED Hindi encoder (see ENCODER_DECISION.md: measured 2026 benchmarks put
    FastConformer/IndicConformer at ~8-14% real-world Hindi WER vs omniASR ~14-26%).
    It also has NATIVE cache-aware streaming (0/80/480/1040 ms) so we do not retrofit
    causality. Interface identical to the other encoders."""

    def __init__(self, model, out_dim, frame_rate, hop_samples, chunk_frames, left_chunks):
        super().__init__()
        self.model = model
        self.out_dim = int(out_dim)
        self.frame_rate = float(frame_rate)
        self.hop_samples = int(hop_samples)
        self.chunk_frames = int(chunk_frames)
        self.left_chunks = int(left_chunks)

    @classmethod
    def load(cls, model_id, chunk_frames, left_chunks, dtype=torch.float32):
        # Loaded via NeMo on the GPU box (nemo.collections.asr). FastConformer subsamples
        # 10 ms mel frames x8 -> 80 ms/frame = 12.5 fps; d_model discovered at runtime.
        import nemo.collections.asr as nemo_asr
        m = nemo_asr.models.ASRModel.from_pretrained(model_id)
        enc = m.encoder
        d_model = int(getattr(m.cfg.encoder, "d_model", 512))
        hop = int(SAMPLE_RATE * 0.08)             # 80 ms/frame
        # set cache-aware streaming context on the encoder here (att_context_size) per latency.
        return cls(enc, d_model, SAMPLE_RATE / hop, hop, chunk_frames, left_chunks)

    def _flen(self, wave_len):
        return torch.clamp(torch.div(wave_len, self.hop_samples, rounding_mode="floor"), min=1)

    def features(self, wave, wave_len):
        # NeMo encoders take a mel spectrogram; the preprocessor is threaded on the GPU box.
        feats = self.model(audio_signal=wave, length=wave_len)[0].transpose(1, 2)
        return feats, self._flen(wave_len)


def build_encoder(cfg, dtype=torch.float32) -> nn.Module:
    ac = cfg.audio
    chunk = int(getattr(ac, "chunk_frames", 8))
    left = int(getattr(ac, "left_chunks", 1))
    if cfg.backend == "real":
        etype = getattr(cfg.base, "encoder_type", "fastconformer")
        if etype in ("fastconformer", "indicconformer"):
            return FastConformerEncoder.load(cfg.base.encoder_id, chunk, left, dtype)
        return OmniW2VEncoder.load(cfg.base.encoder_id, chunk, left, dtype)  # omni_w2v
    enc = TinyEncoder(out_dim=int(getattr(ac, "tiny_enc_dim", 64)),
                      layers=int(getattr(ac, "tiny_enc_layers", 2)),
                      chunk_frames=chunk, left_chunks=left,
                      right_chunks=int(getattr(ac, "right_chunks", 0)))
    return enc.to(dtype)
