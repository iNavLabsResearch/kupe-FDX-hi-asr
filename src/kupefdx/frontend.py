"""Projector — maps continuous encoder features into Nandi's *input embedding* width
(discovered at runtime: for the factorized tied embedding this is the embedding dim,
not hidden_size). Owns the audio_bos/audio_eos markers and train-time SpecAugment.

This is the continuous branch of "teach Nandi the omni audio tokens". The discrete
branch (quantizer -> <aud_k> tokens in Nandi's vocab) is optional and additive.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureFrontend(nn.Module):
    def __init__(self, enc_dim: int, embed_dim: int, projector: str = "mlp",
                 dropout: float = 0.0):
        super().__init__()
        self.enc_dim = int(enc_dim)
        self.embed_dim = int(embed_dim)
        self.norm = nn.LayerNorm(self.enc_dim)
        if projector == "linear":
            self.proj = nn.Linear(self.enc_dim, self.embed_dim)
        else:
            self.proj = nn.Sequential(
                nn.Linear(self.enc_dim, self.embed_dim * 2), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(self.embed_dim * 2, self.embed_dim))
        self.audio_bos = nn.Parameter(torch.zeros(self.embed_dim))
        self.audio_eos = nn.Parameter(torch.zeros(self.embed_dim))
        self.spec_time_mask = 0
        self.spec_time_blocks = 1
        std = self.embed_dim ** -0.5
        nn.init.normal_(self.audio_bos, std=std)
        nn.init.normal_(self.audio_eos, std=std)

    def _time_mask(self, x):
        if not self.training or self.spec_time_mask <= 0:
            return x
        b, t = x.shape[0], x.shape[1]
        for i in range(b):
            for _ in range(int(self.spec_time_blocks)):
                w = int(torch.randint(1, self.spec_time_mask + 1, (1,)).item())
                if w >= t:
                    continue
                s = int(torch.randint(0, t - w, (1,)).item())
                x[i, s:s + w] = 0.0
        return x

    def audio_len(self, num_frames):
        return num_frames

    def forward(self, feats):
        x = self.norm(feats)
        x = self._time_mask(x)
        return self.proj(x)
