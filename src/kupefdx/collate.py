"""Collator: manifest items -> padded tensors for KupeFDXModel.forward.

Emits (all optional keys present only when relevant to the phase):
  wave/wave_len  or  feats/num_frames     audio, raw or cached
  text_ids [B,M] text_lengths [B] labels [B,M]   supervised transcript (+ fc tokens + eos)
  ctc_targets [sumL] ctc_target_lens [B]   concatenated CTC char ids
  code_ids [B,Tc]                          discrete audio-code token ids (if n_codes>0)
"""
from __future__ import annotations

import numpy as np
import torch


class Collator:
    def __init__(self, tokenizer, char_tok, *, bos_id, eos_id, pad_id, special_ids,
                 max_audio_frames=2000, max_text_tokens=256, n_codes=0, mode="raw",
                 frame_rate=50.0, chunk_frames=32, eos_lead_ms=0):
        self.tok = tokenizer
        self.char = char_tok
        self.bos_id, self.eos_id, self.pad_id = int(bos_id), int(eos_id), int(pad_id)
        self.special_ids = special_ids
        self.max_audio_frames = int(max_audio_frames)
        self.max_text_tokens = int(max_text_tokens)
        self.n_codes = int(n_codes)
        self.mode = mode
        self.frame_rate = float(frame_rate)
        self.chunk_frames = int(chunk_frames)
        self.eos_lead_ms = int(eos_lead_ms)          # >0 = predictive end-of-turn labels
        self.max_ctx_tokens = 96

    def _ctx_ids(self, item):
        """Optional prefix: domain tag + prior conversation turns, wrapped in <hist>.
        Returns a 1D LongTensor or None. Kept short (max_ctx_tokens)."""
        from .constants import TOK_HIST_BOS, TOK_HIST_EOS
        dom = item.get("domain")
        ctx = item.get("context") or []
        if not dom and not ctx:
            return None
        body = (f"{dom} : " if dom else "") + " | ".join(ctx)
        s = f"{TOK_HIST_BOS} {body} {TOK_HIST_EOS}"
        ids = self.tok(s, add_special_tokens=False).input_ids[: self.max_ctx_tokens]
        return torch.tensor(ids, dtype=torch.long) if ids else None

    def _target_ids(self, item):
        # prefer the rendered target_sequence (inline flags); else plain transcript, with
        # any legacy `fc` flag list appended before eos.
        seq = item.get("target") or item["text"]
        ids = self.tok(seq, add_special_tokens=False).input_ids
        if ids and ids[0] == self.bos_id:
            ids = ids[1:]
        ids = ids[: self.max_text_tokens]
        if not item.get("target"):
            for t in (item.get("fc") or []):
                if t in self.special_ids:
                    ids.append(self.special_ids[t])
        return ids + [self.eos_id]

    def __call__(self, items):
        B = len(items)
        batch = {}

        # ---- audio ----
        if self.mode == "feats" and "feats" in items[0]:
            feats = [np.asarray(it["feats"], np.float32)[: self.max_audio_frames] for it in items]
            T = max(f.shape[0] for f in feats)
            D = feats[0].shape[1]
            arr = np.zeros((B, T, D), np.float32)
            nf = np.zeros(B, np.int64)
            for i, f in enumerate(feats):
                arr[i, : f.shape[0]] = f
                nf[i] = f.shape[0]
            batch["feats"] = torch.from_numpy(arr)
            batch["num_frames"] = torch.from_numpy(nf)
        else:
            waves = [np.asarray(it["wave"], np.float32) for it in items]
            S = max(len(w) for w in waves)
            arr = np.zeros((B, S), np.float32)
            wl = np.zeros(B, np.int64)
            for i, w in enumerate(waves):
                arr[i, : len(w)] = w
                wl[i] = len(w)
            batch["wave"] = torch.from_numpy(arr)
            batch["wave_len"] = torch.from_numpy(wl)

        # ---- text targets ----
        text_lists = [self._target_ids(it) for it in items]
        M = max(len(t) for t in text_lists)
        text_ids = np.full((B, M), self.pad_id, np.int64)
        tl = np.zeros(B, np.int64)
        for i, t in enumerate(text_lists):
            text_ids[i, : len(t)] = t
            tl[i] = len(t)
        text_ids_t = torch.from_numpy(text_ids)
        batch["text_ids"] = text_ids_t
        batch["text_lengths"] = torch.from_numpy(tl)
        batch["labels"] = text_ids_t.clone()

        # ---- ctc targets (concatenated) ----
        ctc_lists = [self.char.encode(it["text"]) or [self.char.blank + 1] for it in items]
        batch["ctc_targets"] = torch.tensor([c for lst in ctc_lists for c in lst], dtype=torch.long)
        batch["ctc_target_lens"] = torch.tensor([len(c) for c in ctc_lists], dtype=torch.long)

        # ---- per-chunk floor-control events (empty list == all NOTHING) ----
        from .floorcontrol import events_from_timeline
        batch["fc_events"] = [events_from_timeline(it.get("timeline"), self.frame_rate,
                                                   self.chunk_frames, self.eos_lead_ms)
                              for it in items]

        # ---- optional context prefix (domain tag + prior turns) ----
        ctx = [self._ctx_ids(it) for it in items]
        batch["ctx_ids"] = ctx if any(c is not None for c in ctx) else None

        # ---- discrete audio codes ----
        if self.n_codes > 0 and "codes" in items[0]:
            codes = [np.asarray(it["codes"], np.int64) for it in items]
            Tc = max(len(c) for c in codes)
            arr = np.zeros((B, Tc), np.int64)
            for i, c in enumerate(codes):
                arr[i, : len(c)] = c
            batch["code_ids"] = torch.from_numpy(arr)
        return batch
