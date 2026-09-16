#!/usr/bin/env python3
"""Consolidate per-shard manifests (from `11_shard_pipeline.py --no-flush`) into ONE local
training manifest with absolute local paths. Use this for single-box local training so you
don't round-trip audio/features through the Hub.

    # after: python scripts/11_shard_pipeline.py --config configs/en.yaml ... --no-flush
    python scripts/12_build_manifest.py --config configs/en.yaml --out data/manifests/train.jsonl
"""
import argparse
import glob
import json
import os

import _bootstrap  # noqa: F401
from kupefdx.config import load_config
from kupefdx.dataset import read_manifest, write_manifest
from kupefdx.env import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/en.yaml")
    ap.add_argument("--out", default="data/manifests/train.jsonl")
    a = ap.parse_args()
    cfg = load_config(a.config)
    enc_dir = os.path.join(cfg.paths.data_dir, "encoded")

    shard_manifests = sorted(glob.glob(os.path.join(enc_dir, "shards", "*", "manifest.jsonl")))
    if not shard_manifests:
        raise SystemExit(f"no shard manifests under {enc_dir}/shards/*/manifest.jsonl "
                         "(run 11_shard_pipeline.py with --no-flush first)")
    rows, n_feats = [], 0
    for mp in shard_manifests:
        for r in read_manifest(mp):
            for key in ("feats", "codes"):            # relative -> absolute local paths
                if r.get(key) and not os.path.isabs(r[key]):
                    r[key] = os.path.join(enc_dir, r[key])
            if r.get("feats") and os.path.isfile(r["feats"]):
                n_feats += 1
            rows.append(r)
    write_manifest(a.out, rows)
    hrs = sum(r.get("dur", 0) for r in rows) / 3600
    log.info("consolidated %d shard manifests -> %s | %d clips | %.1f h | %d with cached feats",
             len(shard_manifests), a.out, len(rows), hrs, n_feats)
    log.info("set data.manifest: %s and data.use_cached_feats: %s in the config",
             a.out, "true" if n_feats else "false")


if __name__ == "__main__":
    main()
