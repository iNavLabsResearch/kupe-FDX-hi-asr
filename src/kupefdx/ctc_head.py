"""CTC head — a TRAINING + TIMING aid. It is NOT the transcript producer (Nandi is).

This is OUR OWN head (Meta's omniASR-CTC weights are never loaded). Its three roles:
  (1) STAGE A: the objective that fine-tunes the Omni SSL encoder on Hindi;
  (2) a monotonic-alignment anchor during training that speeds Nandi's convergence and
      curbs hallucination;
  (3) at inference, a free TIMING signal only — long CTC-blank runs mark silence /
      end-of-speech for the floor controller.
The authoritative transcript always comes from Nandi (model.generate). The greedy CTC
string is at most a throwaway low-latency DRAFT, never the final output.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCHead(nn.Module):
    def __init__(self, enc_dim: int, vocab_size: int, blank: int = 0):
        super().__init__()
        self.blank = int(blank)
        self.proj = nn.Linear(int(enc_dim), int(vocab_size))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats [B,T,D] -> log-probs [B,T,V]."""
        return F.log_softmax(self.proj(feats), dim=-1)

    def loss(self, feats, flen, targets, target_lens):
        logp = self(feats).transpose(0, 1)          # [T,B,V] for ctc_loss
        # aten::_ctc_loss is unimplemented on MPS — fall back to CPU for this op only.
        if logp.device.type == "mps":
            l = F.ctc_loss(logp.cpu(), targets.cpu(), flen.cpu(), target_lens.cpu(),
                           blank=self.blank, zero_infinity=True)
            return l.to(feats.device)
        return F.ctc_loss(logp, targets, flen, target_lens, blank=self.blank,
                          zero_infinity=True)

    @torch.no_grad()
    def greedy(self, feats: torch.Tensor):
        """Greedy CTC decode -> (collapsed_id_seqs, per-frame argmax). Collapsing
        removes repeats then blanks."""
        argmax = self(feats).argmax(-1)             # [B,T]
        outs = []
        for row in argmax:
            prev, seq = None, []
            for t in row.tolist():
                if t != prev and t != self.blank:
                    seq.append(t)
                prev = t
            outs.append(seq)
        return outs, argmax

    @torch.no_grad()
    def blank_runs(self, feats: torch.Tensor):
        """Per-frame boolean: is this frame CTC-blank? Used for VAD / end-of-speech
        timing (a long trailing blank run => turn likely finished)."""
        return self(feats).argmax(-1) == self.blank
