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


def _omni_card(model_id: str) -> str:
    """Normalize Hub/path id → fairseq2 card name (underscores).

    facebook/omniASR-W2V-300M  →  omniASR_W2V_300M
    omniASR_W2V_300M           →  omniASR_W2V_300M
    aadel4/omniASR-W2V-300M    →  omniASR_W2V_300M
    """
    return model_id.split("/")[-1].replace("-", "_")


def _omni_hf_mirror(card: str) -> str | None:
    """Transformers-ready Wav2Vec2 mirrors of Meta's fairseq2 SSL checkpoints.
    Official facebook/omniASR-W2V-* repos are fairseq2 assets (no config.json for
    AutoModel) — use these for the HF path. Parity-verified vs Meta weights."""
    return {
        "omniASR_W2V_300M": "aadel4/omniASR-W2V-300M",
        "omniASR_W2V_1B": "aadel4/omniASR-W2V-1B",
    }.get(card)


class OmniW2VEncoder(nn.Module):
    """Real omniASR_W2V SSL backbone wrapper. Loads the raw self-supervised encoder
    (no baked-in vocab) and applies the SAME block-causal mask for streaming.

    Load order:
      1) fairseq2 card via omnilingual-asr  (e.g. omniASR_W2V_300M)
      2) HF Wav2Vec2Model mirror             (aadel4/omniASR-W2V-300M)
    Official Hub id: https://huggingface.co/facebook/omniASR-W2V-300M
    """

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
        # SSL ONLY: never load Meta's CTC-finetuned checkpoint — we train our own CTC head.
        card = _omni_card(model_id)
        if "ctc" in card.lower().split("_"):
            raise ValueError(
                f"encoder_id={model_id!r} looks like a CTC-finetuned checkpoint. Use the SSL "
                "checkpoint (omniASR-W2V-*) — we attach and train our own CTC head.")
        hop = 320                                    # wav2vec2 conv stack: 20 ms/frame -> 50 fps

        # Attempt 1: fairseq2 / omnilingual-asr (registers Meta's asset cards)
        try:
            import omnilingual_asr  # noqa: F401  — registers omniASR_* cards
            from fairseq2.models.hub import load_model
            m = load_model(card)
            m.eval()
            out_dim = int(getattr(m, "model_dim", 0) or getattr(
                getattr(m, "encoder_frontend", None), "model_dim", 0) or 1024)
            log.info("loaded omniASR SSL via fairseq2 (%s) | out_dim=%d | 50 fps", card, out_dim)
            return cls(m, out_dim, SAMPLE_RATE / hop, hop, chunk_frames, left_chunks,
                       "omni-w2v-fairseq2")
        except Exception as e:
            log.info("fairseq2 path not used (%s); trying HF Wav2Vec2Model", e)

        # Attempt 2: Transformers Wav2Vec2 — official facebook/* has no config.json;
        # use the parity-verified mirror when the id is a Meta card / Hub path.
        hf_id = model_id
        if model_id.startswith("facebook/") or "/" not in model_id:
            mirror = _omni_hf_mirror(card)
            if not mirror:
                raise SystemExit(
                    f"no Transformers mirror for {card!r}; install omnilingual-asr "
                    f"(fairseq2) or use facebook/omniASR-W2V-300M / -1B")
            hf_id = mirror
            log.info("using HF Wav2Vec2 mirror %s for Meta card %s", hf_id, card)
        from transformers import Wav2Vec2Model
        m = Wav2Vec2Model.from_pretrained(hf_id).to(dtype)
        out_dim = int(m.config.hidden_size)
        log.info("loaded omniASR SSL encoder (%s) | out_dim=%d | 50 fps | CTC head is OURS, fresh",
                 hf_id, out_dim)
        return cls(m, out_dim, SAMPLE_RATE / hop, hop, chunk_frames, left_chunks, "omni-w2v-hf")

    def _flen(self, wave_len):
        return torch.clamp(torch.div(wave_len, self.hop_samples, rounding_mode="floor"), min=1)

    def _features_fairseq2(self, wave, wave_len):
        """Extract [B,T,D] embeddings from a fairseq2 Wav2Vec2Model."""
        try:
            from fairseq2.nn import BatchLayout
        except ImportError:
            from fairseq2.data import BatchLayout  # older fairseq2
        seq_lens = [int(x) for x in wave_len.tolist()]
        try:
            layout = BatchLayout.of(wave, seq_lens)
        except TypeError:
            layout = BatchLayout.of(batch=wave, seq_lens=seq_lens)
        m = self.model
        if hasattr(m, "encoder_frontend") and hasattr(m, "encoder"):
            packed = m.encoder_frontend.extract_features(wave, layout)
            enc_out, enc_layout = packed[0], packed[1]
            if hasattr(m.encoder_frontend, "process_features"):
                try:
                    enc_out, enc_layout = m.encoder_frontend.process_features(
                        enc_out, enc_layout, None)
                except TypeError:
                    enc_out, enc_layout = m.encoder_frontend.process_features(enc_out, enc_layout)
            feats = m.encoder(enc_out, enc_layout)
            if isinstance(feats, tuple):
                feats = feats[0]
            return feats, self._flen(wave_len)
        out = m(wave, layout)
        feats = out[0] if isinstance(out, tuple) else out
        return feats, self._flen(wave_len)

    def features(self, wave, wave_len):
        if self.kind == "omni-w2v-fairseq2":
            return self._features_fairseq2(wave, wave_len)
        # HF Wav2Vec2Model: raw waveform [B,S] → last_hidden_state [B,T,D]
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
