"""OPTIONAL, OFF BY DEFAULT (n_codes=0) — NOT RECOMMENDED for quality.

k-means discretization of encoder features into <aud_k> vocab tokens (HuBERT-style discrete
units). We DO NOT use this in the real recipe: the default path feeds CONTINUOUS encoder
features to Nandi via the projector (inputs_embeds) — no discretization, zero information loss,
exactly like frontier speech-LLMs (LLaVA / Qwen-Audio / SALMONN). Discretizing throws away
acoustic detail and hurts WER, so it is kept only as an explicit opt-in experiment
(audio.n_codes > 0), never on by default.
"""
from __future__ import annotations

import numpy as np
import torch


class KMeansQuantizer:
    def __init__(self, centroids: np.ndarray):
        self.centroids = np.asarray(centroids, dtype=np.float32)   # [K, D]
        self.n_codes = self.centroids.shape[0]
        self.dim = self.centroids.shape[1]

    @classmethod
    def fit(cls, feats: np.ndarray, n_codes: int, iters: int = 25, seed: int = 0,
            log_fn=None) -> "KMeansQuantizer":
        """Lloyd's algorithm on a [N, D] sample of features."""
        rng = np.random.default_rng(seed)
        feats = np.asarray(feats, dtype=np.float32)
        n = feats.shape[0]
        k = min(int(n_codes), n)
        cent = feats[rng.choice(n, size=k, replace=False)].copy()
        for it in range(int(iters)):
            d = ((feats[:, None, :] - cent[None, :, :]) ** 2).sum(-1)   # [N,K]
            assign = d.argmin(1)
            new = cent.copy()
            for j in range(k):
                m = assign == j
                if m.any():
                    new[j] = feats[m].mean(0)
                else:                                   # re-seed a dead centroid
                    new[j] = feats[rng.integers(0, n)]
            shift = float(((new - cent) ** 2).sum(1).mean())
            cent = new
            if log_fn:
                log_fn(it, shift)
            if shift < 1e-6:
                break
        return cls(cent)

    def encode(self, feats: np.ndarray) -> np.ndarray:
        """[T, D] (or [B,T,D]) -> code ids [T] (or [B,T])."""
        f = np.asarray(feats, dtype=np.float32)
        flat = f.reshape(-1, f.shape[-1])
        d = ((flat[:, None, :] - self.centroids[None, :, :]) ** 2).sum(-1)
        codes = d.argmin(1)
        return codes.reshape(f.shape[:-1])

    def save(self, path: str) -> None:
        torch.save({"centroids": self.centroids}, path)

    @classmethod
    def load(cls, path: str) -> "KMeansQuantizer":
        return cls(torch.load(path, map_location="cpu")["centroids"])
