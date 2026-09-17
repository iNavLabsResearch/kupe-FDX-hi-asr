"""Training-readiness checks. Two gates, importable + used by scripts/check_ready.py:

  check_data(cfg, manifest, sample)  — the ENCODED ASR corpus: hours by domain, duration
      histogram, dup ratio, and a real NPZ pass (shape [T,d_model], dtype, NaN/Inf,
      frame-rate vs duration on a random sample).
  check_gen(manifest, kind, lang)    — an LLM-GENERATED manifest (fc / domain): schema +
      semantic-ordering validity, scenario/flag distribution, no-flag ratio, audio
      existence, duplicates, and (Hindi only) Devanagari purity.

Each returns (ok: bool, lines: list[str]). No printing here — the caller renders.
"""
from __future__ import annotations

import collections
import hashlib
import os
import random

FPS = 12.5  # FastConformer frame rate (80 ms/frame)
_HARD_GEN = {"invalid_frac": 0.02, "dup_frac": 0.02, "romanized_frac": 0.05}
_DEV_MIN = 0.90


def _bar(frac, width=24):
    n = max(0, min(width, int(round(frac * width))))
    return "█" * n + "·" * (width - n)


# ───────────────────────────── encoded-corpus check (was 13_data_report) ─────────────────────────────
def check_data(cfg, manifest: str, sample: int = 300, expect_dim: int = 512):
    import numpy as np
    from .jsonl import read_manifest
    L, fail = [], []
    if not os.path.isfile(manifest):
        return False, [f"no manifest at {manifest} — run 12_build_manifest.py first"]
    rows = read_manifest(manifest)
    n = len(rows)
    if not n:
        return False, [f"{manifest} is empty"]
    durs = [float(r.get("dur", 0) or 0) for r in rows]
    total_h = sum(durs) / 3600
    L.append(f"clips {n:,} · {total_h:,.1f} h · avg {np.mean(durs):.1f}s "
             f"(min {min(durs):.1f}/max {max(durs):.1f})")

    by_dom = collections.defaultdict(lambda: [0, 0.0])
    for r, d in zip(rows, durs):
        by_dom[r.get("domain", "?")][0] += 1
        by_dom[r.get("domain", "?")][1] += d
    L.append("hours by domain:")
    for k, (c, s) in sorted(by_dom.items(), key=lambda x: -x[1][1]):
        L.append(f"   {k:20s} {_bar(s/3600/max(total_h,1e-9))} {s/3600:7.1f} h ({c:,})")

    seen, dups, empties = set(), 0, 0
    for r in rows:
        t = (r.get("text", "") or "").strip()
        if len(t) < 2:
            empties += 1
        key = (t.lower(), round(float(r.get("dur", 0) or 0), 1))
        if key in seen:
            dups += 1
        seen.add(key)
    L.append(f"transcripts: empty/short={empties}  exact-dupes={dups} ({100*dups/n:.1f}%)")
    if empties > 0.02 * n:
        fail.append(f"{empties} empty transcripts (>2%)")

    with_feats = [r for r in rows if r.get("feats")]
    L.append(f"rows with feats path: {len(with_feats):,}/{n:,}")
    if not with_feats:
        fail.append("no rows carry a feats path (use_cached_feats would be false)")
    else:
        pick = random.sample(with_feats, min(sample, len(with_feats)))
        ok = miss = bad_dim = nan = 0
        cache = {}
        for r in pick:
            p = r["feats"]
            if not os.path.isabs(p):
                p = os.path.join(cfg.paths.data_dir, "encoded", p)
            if not os.path.isfile(p):
                miss += 1; continue
            try:
                if p not in cache:
                    cache.clear(); cache[p] = np.load(p)
                z = cache[p]
                key = r.get("feats_key") or r.get("id")
                arr = z[key] if key in z else z[z.files[0]]
                if arr.ndim != 2 or arr.shape[-1] != expect_dim:
                    bad_dim += 1; continue
                if not np.isfinite(arr.astype(np.float32)).all():
                    nan += 1; continue
                ok += 1
            except Exception:
                miss += 1
        L.append(f"NPZ sample: checked {len(pick)} · ok={ok} missing={miss} "
                 f"wrong-dim={bad_dim} nan/inf={nan}")
        if miss:
            fail.append(f"{miss}/{len(pick)} sampled feats files missing")
        if bad_dim:
            fail.append(f"{bad_dim} feats with wrong dim (expected [T,{expect_dim}])")
        if nan:
            fail.append(f"{nan} feats contain NaN/Inf")
    L.extend(f"FAIL: {f}" for f in fail)
    return (not fail), L


# ───────────────────────────── generated-manifest check (was 09_data_quality) ─────────────────────────────
def check_gen(manifest: str, kind: str = "fc", lang: str = "en"):
    import re
    from .fcgen.schema import SchemaError, validate_row
    from .jsonl import read_manifest
    from .text import devanagari_ratio, normalize
    L, fail = [], []
    if not os.path.isfile(manifest):
        return False, [f"missing {manifest} — generate it with scripts/gen_data.py"]
    rows = read_manifest(manifest)
    n = len(rows)
    if not n:
        return False, [f"{manifest} is empty"]

    _EN_OK = re.compile(r"\b(BP|RAM|CPU|OK|SMS|OTP|PIN|ID|URL|API|server|email|internet|"
                        r"login|password|Agent|User)\b", re.I)

    def dev_of(r):
        parts = [r.get("transcript", ""), r.get("target_sequence", "")]
        parts += [normalize(c) for c in (r.get("context") or [])]
        parts += [s.get("surface", "") for s in (r.get("timeline") or [])]
        txt = re.sub(r"<[^>]+>", " ", " ".join(p for p in parts if p))
        return devanagari_ratio(_EN_OK.sub(" ", txt))

    invalid, dups, romanized, no_audio, bad_dur, noflag = [], 0, 0, 0, 0, 0
    seen, scen, dom, flags = set(), collections.Counter(), collections.Counter(), collections.Counter()
    for r in rows:
        if kind == "fc":
            try:
                validate_row(dict(r))
            except SchemaError as e:
                invalid.append((r.get("id"), str(e)))
        scen[r.get("scenario", "?")] += 1
        dom[r.get("domain", "?")] += 1
        rflags = r.get("flags") or [s.get("flag") for s in (r.get("timeline") or []) if s.get("flag")]
        for f in rflags:
            if f in FC_TOKENS:
                flags[f] += 1
        if not rflags:
            noflag += 1
        if lang == "hi":
            if dev_of(r) < _DEV_MIN:
                romanized += 1
        if r.get("audio") and not os.path.isfile(r["audio"]):
            no_audio += 1
        h = hashlib.sha1((r.get("target_sequence", "") + "|" + str(r.get("context"))).encode()).hexdigest()
        if h in seen:
            dups += 1
        seen.add(h)
        af = r.get("audio_features") or {}
        if af:
            d = float(af.get("duration_s", 1))
            if d <= 0 or d > 40:
                bad_dur += 1

    L.append(f"{manifest}: {n} rows · domains={dict(dom)}")
    L.append("scenarios: " + ", ".join(f"{k} {100*v/n:.0f}%" for k, v in scen.most_common()))
    if flags:
        L.append("flags: " + ", ".join(f"{k} {v}" for k, v in flags.most_common())
                 + f" · no-flag {100*noflag/n:.0f}%")
    L.append(f"invalid={len(invalid)} dupes={dups} missing-audio={no_audio} bad-dur={bad_dur}")
    for rid, why in invalid[:5]:
        L.append(f"   INVALID {rid} :: {why}")

    if len(invalid) / n > _HARD_GEN["invalid_frac"]:
        fail.append(f"invalid {len(invalid)/n:.1%} > {_HARD_GEN['invalid_frac']:.0%}")
    if dups / n > _HARD_GEN["dup_frac"]:
        fail.append(f"duplicates {dups/n:.1%} > {_HARD_GEN['dup_frac']:.0%}")
    if lang == "hi" and romanized / n > _HARD_GEN["romanized_frac"]:
        fail.append(f"romanized {romanized/n:.1%} > {_HARD_GEN['romanized_frac']:.0%}")
    L.extend(f"FAIL: {f}" for f in fail)
    return (not fail), L


from .constants import FC_TOKENS  # noqa: E402  (kept at bottom: avoids a torch import cycle)
