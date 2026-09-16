"""Evaluation: AR-decode WER/CER, CTC-greedy WER (the fast path), and floor-control
precision/recall/F1 + false-fire rate when fc labels are present."""
from __future__ import annotations

import torch

from .constants import FC_TOKENS
from .dataset import ManifestDataset
from .metrics import corpus_wer, fc_prf


@torch.no_grad()
def run_eval(model, rows, collator, device, *, max_new_tokens=128, batch_size=8,
             mode="raw") -> dict:
    model.eval()
    ds = ManifestDataset(rows, mode=mode, max_dur=1e9)
    refs, hyps, ctc_hyps = [], [], []
    ref_fc, hyp_fc = [], []
    inv = {v: k for k, v in model.special_ids.items()}
    for i in range(0, len(ds), batch_size):
        items = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
        batch = collator(items)
        gen_ids = model.generate(
            wave=batch.get("wave"), wave_len=batch.get("wave_len"),
            feats=batch.get("feats"), num_frames=batch.get("num_frames"),
            code_ids=batch.get("code_ids"), max_new_tokens=max_new_tokens)
        hyps += model.transcribe(gen_ids)
        ctc_hyps += model.ctc_transcribe(
            wave=batch.get("wave"), wave_len=batch.get("wave_len"),
            feats=batch.get("feats"), num_frames=batch.get("num_frames"))
        for it, ids in zip(items, gen_ids):
            refs.append(it["text"])
            ref_fc.append(set(it.get("fc", [])))
            hyp_fc.append({inv[t] for t in ids if t in inv and inv[t] in FC_TOKENS})

    rep = corpus_wer(refs, hyps)
    rep["ctc_wer"] = corpus_wer(refs, ctc_hyps)["wer"]
    if any(ref_fc):
        rep["fc"] = fc_prf(ref_fc, hyp_fc, FC_TOKENS)
    rep["samples"] = [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyps))[:5]]
    return rep


def save_report(rep: dict, out_path: str, title: str = "") -> None:
    import json
    import os
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"title": title, **rep}, f, indent=2, ensure_ascii=False)
