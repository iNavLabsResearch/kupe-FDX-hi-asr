#!/usr/bin/env python3
"""Stage 13 — dataset distribution + NPZ sanity report for the MAIN ASR manifest.

Unlike 09_data_quality.py (which gates the floor-control / domain manifests), this
inspects the pretraining/align corpus: per-domain & per-split hours, duration histogram,
transcript stats, duplicate ratio, and a real NPZ sanity pass (loads a sample of
feats.npz, checks shape [T, d_model], dtype, NaN/Inf, and frame-rate vs duration).

    python scripts/13_data_report.py --config configs/en.yaml --manifest data/manifests/train.jsonl
    python scripts/13_data_report.py --config configs/en.yaml --manifest data/manifests/train.jsonl --sample 500

Exits non-zero if a hard problem is found (missing feats, wrong feature dim, NaN),
so it can gate a training run in CI.
"""
import argparse
import collections
import os
import random

import numpy as np

import _bootstrap  # noqa: F401
from kupefdx.config import load_config
from kupefdx.env import log
from kupefdx.jsonl import read_manifest

FPS = 12.5  # FastConformer: 12.5 frames/s (80 ms/frame)


def _hist(durs, edges=(0, 2, 5, 10, 15, 20, 25, 30, 1e9)):
    buckets = collections.OrderedDict()
    labels = [f"{int(edges[i])}-{int(edges[i+1])}s" if edges[i+1] < 1e9 else f">{int(edges[i])}s"
              for i in range(len(edges) - 1)]
    for lb in labels:
        buckets[lb] = 0
    for d in durs:
        for i in range(len(edges) - 1):
            if edges[i] <= d < edges[i + 1]:
                buckets[labels[i]] += 1
                break
    return buckets


def _bar(frac, width=28):
    n = int(round(frac * width))
    return "█" * n + "·" * (width - n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/en.yaml")
    ap.add_argument("--manifest", default="data/manifests/train.jsonl")
    ap.add_argument("--sample", type=int, default=300, help="clips to NPZ-sanity-check")
    ap.add_argument("--expect-dim", type=int, default=512, help="encoder d_model")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if not os.path.isfile(a.manifest):
        raise SystemExit(f"no manifest at {a.manifest} — run 12_build_manifest.py first")

    rows = read_manifest(a.manifest)
    n = len(rows)
    if not n:
        raise SystemExit("manifest is empty")
    durs = [float(r.get("dur", 0) or 0) for r in rows]
    total_h = sum(durs) / 3600
    hard_fail = []

    print("=" * 64)
    print(f"  DATASET REPORT — {a.manifest}")
    print("=" * 64)
    print(f"  clips: {n:,}   hours: {total_h:,.1f} h   avg dur: {np.mean(durs):.1f}s "
          f"(min {min(durs):.1f} / max {max(durs):.1f})")

    # ---- per-domain hours ----
    by_dom = collections.defaultdict(lambda: [0, 0.0])
    for r, d in zip(rows, durs):
        k = r.get("domain", "?")
        by_dom[k][0] += 1
        by_dom[k][1] += d
    print("\n  ── hours by domain ──")
    for k, (c, s) in sorted(by_dom.items(), key=lambda x: -x[1][1]):
        h = s / 3600
        print(f"   {k:22s} {_bar(h / total_h)} {h:8.1f} h  ({c:,} clips)")

    # ---- per-split ----
    by_split = collections.defaultdict(lambda: [0, 0.0])
    for r, d in zip(rows, durs):
        by_split[r.get("split", "?")][0] += 1
        by_split[r.get("split", "?")][1] += d
    print("\n  ── split ──")
    for k, (c, s) in sorted(by_split.items()):
        print(f"   {k:22s} {s/3600:8.1f} h  ({c:,} clips)")

    # ---- duration histogram ----
    print("\n  ── duration histogram ──")
    hist = _hist(durs)
    for lb, c in hist.items():
        print(f"   {lb:22s} {_bar(c / n)} {c:,}")

    # ---- transcript stats + dedup ----
    seen, dups, empties, nchars, nwords = set(), 0, 0, [], []
    for r in rows:
        t = (r.get("text", "") or "").strip()
        if len(t) < 2:
            empties += 1
        nchars.append(len(t))
        nwords.append(len(t.split()))
        key = (t.lower(), round(float(r.get("dur", 0) or 0), 1))
        if key in seen:
            dups += 1
        seen.add(key)
    print("\n  ── transcripts ──")
    print(f"   avg {np.mean(nwords):.1f} words / {np.mean(nchars):.0f} chars per clip")
    print(f"   empty/too-short: {empties}   exact-content duplicates: {dups} "
          f"({100*dups/n:.1f}%)")
    if empties > 0.02 * n:
        hard_fail.append(f"{empties} empty transcripts (>2%)")

    # ---- NPZ sanity (sample) ----
    with_feats = [r for r in rows if r.get("feats")]
    print("\n  ── NPZ sanity ──")
    print(f"   rows with feats path: {len(with_feats):,} / {n:,}")
    if not with_feats:
        hard_fail.append("no rows carry a feats path (use_cached_feats will be false)")
    else:
        sample = random.sample(with_feats, min(a.sample, len(with_feats)))
        ok, miss, bad_dim, nan, cache = 0, 0, 0, 0, {}
        fps_err = []
        for r in sample:
            p = r["feats"]
            if not os.path.isabs(p):
                p = os.path.join(cfg.paths.data_dir, "encoded", p)
            if not os.path.isfile(p):
                miss += 1
                continue
            try:
                if p not in cache:
                    cache.clear()  # keep memory bounded — one npz open at a time
                    cache[p] = np.load(p)
                z = cache[p]
                key = r.get("feats_key") or r.get("id")
                arr = z[key] if key in z else z[z.files[0]]
                if arr.ndim != 2 or arr.shape[-1] != a.expect_dim:
                    bad_dim += 1
                    continue
                if not np.isfinite(arr.astype(np.float32)).all():
                    nan += 1
                    continue
                exp = float(r.get("dur", 0) or 0) * FPS
                if exp > 0 and abs(arr.shape[0] - exp) / exp > 0.25:
                    fps_err.append((r.get("id"), arr.shape[0], round(exp)))
                ok += 1
            except Exception as e:
                miss += 1
                log.warning("npz read fail %s: %s", p, e)
        print(f"   checked {len(sample)}: ok={ok} missing={miss} wrong-dim={bad_dim} nan/inf={nan}")
        if fps_err[:3]:
            print(f"   frame-rate outliers (T vs ~{FPS}·dur): "
                  + ", ".join(f"{i}:{t}f/exp{e}" for i, t, e in fps_err[:3]))
        if miss:
            hard_fail.append(f"{miss}/{len(sample)} sampled feats files missing")
        if bad_dim:
            hard_fail.append(f"{bad_dim} feats with wrong dim (expected [T,{a.expect_dim}])")
        if nan:
            hard_fail.append(f"{nan} feats contain NaN/Inf")

    print("\n" + "=" * 64)
    if hard_fail:
        print("  RESULT: ❌ FAIL")
        for f in hard_fail:
            print(f"    - {f}")
        raise SystemExit(1)
    print(f"  RESULT: ✅ OK — {total_h:,.1f} h across {len(by_dom)} domains, feats healthy")
    print("=" * 64)


if __name__ == "__main__":
    main()
