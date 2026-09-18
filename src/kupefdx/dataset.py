"""Manifest-backed dataset. A manifest is a JSONL file, one clip per line:

  {"id","audio","text","dur","domain","split","feats"?,"codes"?,"fc"?}

  audio  path to 16 kHz wav (raw mode)         feats  path to cached [T,D] .npy (feats mode)
  codes  path to [T] .npy audio-code ids       fc     list[str] floor-control target tokens
                                                      appended to the text target sequence

Raw mode runs the encoder live (Phase 1/3). Feats mode reads cached features (Phase 2,
frozen encoder) for a big speedup.
"""
from __future__ import annotations

import numpy as np
from torch.utils.data import Dataset

from .audio import load_wav
from .constants import SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
from .jsonl import read_manifest, write_manifest  # re-export


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict], mode: str = "raw",
                 max_dur: float = 30.0, split: str | None = None):
        if split:
            rows = [r for r in rows if r.get("split", SPLIT_TRAIN) == split]
        rows = [r for r in rows if float(r.get("dur", 1.0)) <= max_dur]
        # feats mode never reads raw audio: drop rows lacking a cached-feats path so nothing
        # falls back to load_wav (which crashes on a feats-only box).
        if mode == "feats":
            rows = [r for r in rows if r.get("feats")]
        self.rows = rows
        self.mode = mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        # `text` is the plain transcript (CTC + WER ref); `target` (if present) is the
        # rendered sequence with inline floor-control flags that the LM decodes.
        item = {"text": r.get("text") or r.get("transcript", ""),
                "target": r.get("target_sequence"),
                "fc": r.get("fc", r.get("flags", [])),
                "timeline": r.get("timeline"),          # per-chunk floor-control events
                "context": r.get("context"),            # prior conversation turns
                "domain": r.get("domain", "general")}
        if self.mode == "feats" and r.get("feats"):
            item["feats"] = _load_array(r["feats"], r.get("feats_key")).astype(np.float32)
        else:
            item["wave"] = load_wav(r["audio"]).astype(np.float32)
        if r.get("codes"):
            item["codes"] = _load_array(r["codes"], r.get("codes_key")).astype(np.int64)
        return item


def _load_array(path: str, key: str | None = None) -> np.ndarray:
    """Load a .npy, or one array from a packed .npz (key = clip id)."""
    if key and (path.endswith(".npz") or path.endswith(".npz.npz")):
        with np.load(path, allow_pickle=False) as z:
            return z[key]
    return np.load(path)


def split_rows(rows):
    tr = [r for r in rows if r.get("split") == SPLIT_TRAIN]
    va = [r for r in rows if r.get("split") == SPLIT_VAL]
    te = [r for r in rows if r.get("split") == SPLIT_TEST]
    return tr, va, te
