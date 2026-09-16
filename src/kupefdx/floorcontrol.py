"""Per-chunk floor-control head + stream decider — the STABILITY layer.

The inline AR flags (in target_sequence) give a turn-level transcript with signals. But a
streaming agent is asked, every 320-640 ms chunk, "emit something or not?" — and the answer
must be **NO most of the time**. So this head is a per-chunk classifier over encoder
features whose dominant, default class is NOTHING. It is trained with a huge NOTHING
majority (every ordinary ASR chunk is a NOTHING example), and rare classes are up-weighted
so they stay learnable without the model ever *drifting* into chatter.

At inference `StreamDecider` adds two hard stability guards on top of the softmax:
  * a confidence THRESHOLD (fire only if the class clearly beats NOTHING), and
  * HYSTERESIS (a minimum gap between fires; EOS latches until reset),
so the model cannot fire on every chunk even if logits wobble.

Classes: 0 NOTHING · 1 BACKCHANNEL · 2 THINK · 3 EOS_SPEECH · 4 SILENCE.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (FC_BACKCHANNEL, FC_EOS_SPEECH, FC_SILENCE, FC_THINK)

# class index <-> flag token
FC_CLASSES = ["NOTHING", FC_BACKCHANNEL, FC_THINK, FC_EOS_SPEECH, FC_SILENCE]
CLASS_OF_FLAG = {FC_BACKCHANNEL: 1, FC_THINK: 2, FC_EOS_SPEECH: 3, FC_SILENCE: 4}
FLAG_OF_CLASS = {v: k for k, v in CLASS_OF_FLAG.items()}
NOTHING = 0

# Rich Devanagari surface inventories — plain acknowledgments AND emotional expressions
# (laughter, sighs, surprise). In the data the agent picks a context-appropriate surface;
# in AR inference Nandi generates the word after the flag. These are the fast-path defaults
# and the allowed set the validator checks against.
SURFACES = {
    FC_BACKCHANNEL: ["हाँ", "हूँ", "जी", "जी हाँ", "अच्छा", "ठीक", "ठीक है", "बिलकुल",
                     "सही", "ओके", "ओह", "अरे", "अरे वाह", "वाह", "ओहो", "अहा",
                     "हाहाहा", "हे हे", "बाप रे", "सच में"],
    FC_THINK: ["हम्म", "हम्म्म", "उम्म", "आह", "ओह", "अच्छा", "देखिए", "ज़रा रुकिए",
               "एक मिनट", "सोचने दीजिए"],
}
# emotional-expression subset (used by the "expression" scenario in data generation).
EXPRESSIONS = ["हाहाहा", "हे हे", "उफ़", "आह", "ओह", "अरे", "बाप रे", "वाह", "ओहो", "इश"]

DEFAULT_SURFACE = {FC_BACKCHANNEL: "हाँ", FC_THINK: "हम्म"}


class FloorControlHead(nn.Module):
    def __init__(self, enc_dim: int, hidden: int = 128, n_classes: int = 5,
                 ctx_dim: int = 0):
        super().__init__()
        self.n_classes = n_classes
        self.net = nn.Sequential(
            nn.Linear(enc_dim + ctx_dim, hidden), nn.GELU(),
            nn.Linear(hidden, n_classes))

    def _chunk_pool(self, feats, flen, chunk_frames):
        """feats [B,T,D] -> list per sample of pooled chunk features [n_ch, D]."""
        cf = max(1, int(chunk_frames))
        pooled = []
        for b in range(feats.shape[0]):
            T = int(flen[b].item())
            f = feats[b, :T]
            n = max(1, math.ceil(T / cf))
            chunks = [f[i * cf:(i + 1) * cf].mean(0) for i in range(n)]
            pooled.append(torch.stack(chunks, 0))       # [n_ch, D]
        return pooled

    def forward(self, feats, flen, chunk_frames, ctx=None):
        pooled = self._chunk_pool(feats, flen, chunk_frames)
        logits = []
        for b, p in enumerate(pooled):
            if ctx is not None:
                p = torch.cat([p, ctx[b][None].expand(p.shape[0], -1)], dim=-1)
            logits.append(self.net(p))                  # [n_ch, C]
        return logits

    def loss(self, logits_list, events_list, class_weights, device):
        """events_list[b] = [(chunk_idx, class_id), ...]; every other chunk is NOTHING."""
        all_logits, all_labels = [], []
        for b, lg in enumerate(logits_list):
            n = lg.shape[0]
            lab = torch.zeros(n, dtype=torch.long, device=device)   # default NOTHING
            for idx, cls in events_list[b]:
                if 0 <= idx < n:
                    lab[idx] = int(cls)
            all_logits.append(lg)
            all_labels.append(lab)
        if not all_logits:
            return None
        L = torch.cat(all_logits, 0)
        Y = torch.cat(all_labels, 0)
        w = torch.tensor(class_weights, dtype=L.dtype, device=device)
        return F.cross_entropy(L, Y, weight=w)


def events_from_timeline(timeline, frame_rate, chunk_frames, eos_lead_ms=0):
    """[(chunk_idx, class_id)] for each flagged pause; NOTHING elsewhere (implicit).

    `eos_lead_ms` > 0 makes end-of-speech PREDICTIVE: the <EOS_SPEECH> label is shifted
    earlier by that many ms so the head learns to fire as the turn is finishing, instead of
    after a trailing silence. This is what enables sub-100 ms end-of-turn (no silence wait)."""
    cf = max(1, int(chunk_frames))
    lead_chunks = int(round((eos_lead_ms / 1000.0) * float(frame_rate) / cf))
    ev = []
    for seg in timeline or []:
        flag = seg.get("flag")
        if seg.get("kind") == "pause" and flag in CLASS_OF_FLAG:
            frame = float(seg.get("t_s", 0.0)) * float(frame_rate)
            idx = int(frame // cf)
            if flag == FC_EOS_SPEECH and lead_chunks > 0:
                idx = max(0, idx - lead_chunks)
            ev.append((idx, CLASS_OF_FLAG[flag]))
    return ev


class StreamControls:
    """Inference-time knobs for floor-control firing — the 'temperature' analog.

    Per class you set a probability threshold, a logit bias (eagerness: +bias fires more
    often, -bias suppresses), and a minimum chunk gap since the last fire. A global
    `temperature` sharpens (<1) or flattens (>1) the softmax. Defaults are sensible; an
    operator dials, e.g., `biases['<BC>']=+1.0` to make the agent backchannel more, or
    `thresholds['<EOS_SPEECH>']=0.7` to make it commit to end-of-turn more conservatively.
    """

    def __init__(self, thresholds=None, biases=None, min_gaps=None, temperature=1.0,
                 enabled=None, chunk_ms=80, endpoint_mode="predictive", emit_ctc_draft=False):
        # LOW-LATENCY knobs. chunk_ms sets the reaction granularity (smaller = lower latency).
        # endpoint_mode: "predictive" fires EOS from the head the moment the turn looks done
        # (no silence timeout) -> sub-100 ms end-of-turn; "silence" waits for a trailing gap
        # (safer, higher latency). emit_ctc_draft keeps the transcript = Nandi only when False.
        self.chunk_ms = int(chunk_ms)
        self.endpoint_mode = str(endpoint_mode)
        self.emit_ctc_draft = bool(emit_ctc_draft)
        self.temperature = float(temperature)
        self.thresholds = {FC_BACKCHANNEL: 0.40, FC_THINK: 0.50,
                           FC_EOS_SPEECH: 0.50, FC_SILENCE: 0.60, **(thresholds or {})}
        self.biases = {FC_BACKCHANNEL: 0.0, FC_THINK: 0.0,
                       FC_EOS_SPEECH: 0.0, FC_SILENCE: 0.0, **(biases or {})}
        self.min_gaps = {FC_BACKCHANNEL: 2, FC_THINK: 3,
                         FC_EOS_SPEECH: 0, FC_SILENCE: 4, **(min_gaps or {})}
        # a class can be turned off entirely at inference (e.g. disable THINK for a bot).
        self.enabled = {FC_BACKCHANNEL: True, FC_THINK: True,
                        FC_EOS_SPEECH: True, FC_SILENCE: True, **(enabled or {})}

    @classmethod
    def from_config(cls, cfg):
        s = getattr(cfg, "stream", None)
        if s is None:
            return cls()
        return cls(thresholds=getattr(s, "thresholds", None) and s.thresholds.to_dict(),
                   biases=getattr(s, "biases", None) and s.biases.to_dict(),
                   min_gaps=getattr(s, "min_gaps", None) and s.min_gaps.to_dict(),
                   temperature=float(getattr(s, "temperature", 1.0)),
                   chunk_ms=int(getattr(s, "chunk_ms", 80)),
                   endpoint_mode=str(getattr(s, "endpoint_mode", "predictive")),
                   emit_ctc_draft=bool(getattr(s, "emit_ctc_draft", False)))

    def _bias_vec(self, device, dtype, n):
        import torch
        b = torch.zeros(n, device=device, dtype=dtype)
        for flag, cid in CLASS_OF_FLAG.items():
            b[cid] = self.biases.get(flag, 0.0)
        return b


class StreamDecider:
    """Turns per-chunk class LOGITS into a STABLE, CONTROLLABLE fire/no-fire decision."""

    def __init__(self, controls: "StreamControls | None" = None, **legacy):
        # legacy kwargs (fire_threshold, min_gap_chunks, silence_threshold) still accepted.
        if controls is None:
            controls = StreamControls()
            if "fire_threshold" in legacy:
                for f in (FC_BACKCHANNEL, FC_THINK, FC_EOS_SPEECH):
                    controls.thresholds[f] = float(legacy["fire_threshold"])
            if "silence_threshold" in legacy:
                controls.thresholds[FC_SILENCE] = float(legacy["silence_threshold"])
            if "min_gap_chunks" in legacy:
                for f in (FC_BACKCHANNEL, FC_THINK):
                    controls.min_gaps[f] = int(legacy["min_gap_chunks"])
        self.c = controls
        self.last_fire = -10_000
        self.eos_latched = False

    def reset(self):
        self.last_fire = -10_000
        self.eos_latched = False

    def unlatch(self):
        """Resume after a premature end-of-speech: if the user keeps talking, we must not
        stay latched (that would drop the tail of the utterance). Called when speech resumes.
        This is what makes predictive endpointing SAFE — a wrong early EOS self-corrects."""
        self.eos_latched = False

    def decide(self, logits, chunk_idx) -> int:
        """logits: [n_classes] tensor -> class id (NOTHING if it should stay quiet).
        Applies temperature + per-class bias, then per-class threshold/gap/latch."""
        import torch
        z = logits / max(self.c.temperature, 1e-3) + self.c._bias_vec(
            logits.device, logits.dtype, logits.shape[-1])
        probs = torch.softmax(z, dim=-1)
        cls = int(probs.argmax().item())
        if self.eos_latched or cls == NOTHING:
            return NOTHING
        flag = FLAG_OF_CLASS[cls]
        if not self.c.enabled.get(flag, True):
            return NOTHING
        p = float(probs[cls].item())
        if p < self.c.thresholds.get(flag, 0.5) or p <= float(probs[NOTHING].item()):
            return NOTHING
        if (chunk_idx - self.last_fire) < self.c.min_gaps.get(flag, 0):
            return NOTHING
        self.last_fire = chunk_idx
        if cls == CLASS_OF_FLAG[FC_EOS_SPEECH]:
            self.eos_latched = True
        return cls
