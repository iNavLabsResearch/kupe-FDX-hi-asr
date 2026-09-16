"""WER / CER via Levenshtein, plus floor-control precision/recall/F1. No jiwer dep."""
from __future__ import annotations

from .text import normalize


def _edit_distance(a: list, b: list) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wer(ref: str, hyp: str) -> float:
    r, h = normalize(ref).split(), normalize(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


def cer(ref: str, hyp: str) -> float:
    r, h = list(normalize(ref)), list(normalize(hyp))
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


def corpus_wer(refs: list[str], hyps: list[str]) -> dict:
    rw = sum(len(normalize(r).split()) for r in refs)
    rc = sum(len(normalize(r)) for r in refs)
    ew = sum(_edit_distance(normalize(r).split(), normalize(h).split()) for r, h in zip(refs, hyps))
    ec = sum(_edit_distance(list(normalize(r)), list(normalize(h))) for r, h in zip(refs, hyps))
    return {"wer": ew / max(rw, 1), "cer": ec / max(rc, 1), "n": len(refs)}


def fc_prf(ref_tokens: list[set], hyp_tokens: list[set], labels: list[str]) -> dict:
    """Per-signal precision/recall/F1 + overall false-fire rate on empty-ref examples."""
    out = {}
    for lab in labels:
        tp = sum(1 for r, h in zip(ref_tokens, hyp_tokens) if lab in r and lab in h)
        fp = sum(1 for r, h in zip(ref_tokens, hyp_tokens) if lab not in r and lab in h)
        fn = sum(1 for r, h in zip(ref_tokens, hyp_tokens) if lab in r and lab not in h)
        p = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        out[lab] = {"p": p, "r": rc, "f1": 2 * p * rc / max(p + rc, 1e-9)}
    empties = [h for r, h in zip(ref_tokens, hyp_tokens) if not (r & set(labels))]
    fired = sum(1 for h in empties if h & set(labels))
    out["false_fire_rate"] = fired / max(len(empties), 1)
    return out
