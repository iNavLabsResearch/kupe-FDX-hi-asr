#!/usr/bin/env python3
"""Stage 9 — STRICT data-quality check for a floor-control or domain manifest.

Reports and gates on: schema validity + semantic-ordering, per-FLAG distribution,
per-SCENARIO distribution, per-DOMAIN counts, no-flag ratio, Devanagari purity (rejects
romanized Hindi), audio existence, duration/pause sanity, and duplicates. Exits non-zero
if any hard threshold fails, so it can gate a training run in CI.

    python scripts/09_data_quality.py --manifest data/manifests/fc.jsonl
    python scripts/09_data_quality.py --manifest data/manifests/domain.jsonl --kind domain
"""
import argparse
import hashlib
import os
import re
from collections import Counter

import _bootstrap  # noqa: F401
from kupefdx.constants import FC_TOKENS
from kupefdx.dataset import read_manifest
from kupefdx.env import log
from kupefdx.fcgen.schema import SchemaError, validate_row
from kupefdx.text import devanagari_ratio, normalize

DEV_MIN = 0.90          # min Devanagari ratio for a Hindi field (English terms tolerated)
HARD = {"invalid_frac": 0.02, "dup_frac": 0.02, "romanized_frac": 0.05}


# strip markup that is legitimately non-Devanagari before the ratio: flag/audio tokens,
# role labels, and a small whitelist of genuine English terms.
_ENGLISH_OK = re.compile(r"\b(BP|RAM|CPU|OK|SMS|OTP|PIN|ID|URL|API|server|email|"
                         r"internet|login|password|Agent|User)\b", re.I)


def _strip_markup(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)           # <BC>, <EOS_SPEECH>, <aud_k> ...
    s = re.sub(r"\b(Agent|User)\b\s*:", " ", s)
    s = _ENGLISH_OK.sub(" ", s)              # allowed English tech terms
    return s


def _dev_of_row(r) -> float:
    parts = [r.get("transcript", ""), r.get("target_sequence", "")]
    parts += [normalize(c) for c in (r.get("context") or [])]
    for seg in (r.get("timeline") or []):
        parts.append(seg.get("surface", ""))
    txt = _strip_markup(" ".join(p for p in parts if p))
    return devanagari_ratio(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--kind", default="fc", choices=["fc", "domain"])
    ap.add_argument("--lang", default="en", help="en skips the Devanagari-purity check")
    a = ap.parse_args()
    rows = read_manifest(a.manifest)
    n = len(rows)
    if n == 0:
        raise SystemExit("empty manifest")

    invalid, dups, romanized, no_audio, bad_dur = [], 0, 0, 0, 0
    seen_hash = set()
    scen, dom, flags = Counter(), Counter(), Counter()
    noflag = 0
    dev_ratios = []

    for r in rows:
        # schema + semantic ordering (FC rows carry a timeline; domain rows do not)
        if a.kind == "fc":
            try:
                validate_row(dict(r))
            except SchemaError as e:
                invalid.append((r.get("id"), str(e)))
        # distributions
        scen[r.get("scenario", "?")] += 1
        dom[r.get("domain", "?")] += 1
        rflags = r.get("flags") or [s.get("flag") for s in (r.get("timeline") or []) if s.get("flag")]
        for f in rflags:
            if f in FC_TOKENS:
                flags[f] += 1
        if not rflags:
            noflag += 1
        # devanagari purity (Hindi only; English is Latin so skip)
        if a.lang == "hi":
            dr = _dev_of_row(r)
            dev_ratios.append(dr)
            if dr < DEV_MIN:
                romanized += 1
        # audio + duplicates
        if r.get("audio") and not os.path.isfile(r["audio"]):
            no_audio += 1
        h = hashlib.sha1((r.get("target_sequence", "") + "|" + str(r.get("context"))).encode()).hexdigest()
        if h in seen_hash:
            dups += 1
        seen_hash.add(h)
        # duration/pause sanity
        af = r.get("audio_features") or {}
        if af:
            d = float(af.get("duration_s", 1))
            if d <= 0 or d > 40:
                bad_dur += 1
            for p in af.get("pauses", []):
                if p.get("end_s", 0) > d + 0.05 or p.get("dur_s", 0) < 0:
                    bad_dur += 1
                    break

    # ---- report ----
    log.info("=== DATA QUALITY: %s (%d rows) ===", a.manifest, n)
    log.info("scenarios:")
    for k, v in scen.most_common():
        log.info("   %-22s %5d  %5.1f%%", k, v, 100 * v / n)
    log.info("FLAG distribution:")
    total_flags = sum(flags.values()) or 1
    for f in FC_TOKENS:
        c = flags.get(f, 0)
        log.info("   %-14s %5d   (%.1f%% of rows, %.1f%% of all flags)",
                 f, c, 100 * c / n, 100 * c / total_flags)
    log.info("   no-flag rows      %5d   (%.1f%%)", noflag, 100 * noflag / n)
    log.info("domains: %s", dict(dom))
    if a.lang == "hi":
        log.info("Devanagari ratio: mean=%.3f  rows<%.2f (romanized)=%d (%.1f%%)",
                 sum(dev_ratios) / max(len(dev_ratios), 1), DEV_MIN, romanized, 100 * romanized / n)
    log.info("invalid(schema/semantic)=%d  duplicates=%d  missing-audio=%d  bad-duration=%d",
             len(invalid), dups, no_audio, bad_dur)
    for rid, why in invalid[:8]:
        log.info("   INVALID %s :: %s", rid, why)

    # ---- gate ----
    fails = []
    if len(invalid) / n > HARD["invalid_frac"]:
        fails.append(f"invalid {len(invalid)/n:.1%} > {HARD['invalid_frac']:.0%}")
    if dups / n > HARD["dup_frac"]:
        fails.append(f"duplicates {dups/n:.1%} > {HARD['dup_frac']:.0%}")
    if romanized / n > HARD["romanized_frac"]:
        fails.append(f"romanized {romanized/n:.1%} > {HARD['romanized_frac']:.0%}")
    if fails:
        log.error("QUALITY GATE FAILED: %s", "; ".join(fails))
        raise SystemExit(1)
    log.info("QUALITY GATE PASSED ✅")


if __name__ == "__main__":
    main()
