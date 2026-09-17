#!/usr/bin/env python3
"""Consolidate per-shard manifests into ONE local training manifest.

Looks under data/encoded/ recursively so both layouts work:
  shards/<sid>/_hub/manifest.jsonl   (local --no-flush gather)
  encoded/<sid>/manifest.jsonl       (Hub download)

Torch-free on purpose: a 100%-full box can still import this after you free a
few hundred MB. If nothing is on disk, --from-hub pulls ONLY the tiny
manifest.jsonl files (not feats.npz).

    python scripts/12_build_manifest.py --config configs/en.yaml --out data/manifests/train.jsonl
    python scripts/12_build_manifest.py --config configs/en.yaml --out data/manifests/train.jsonl --from-hub
"""
import argparse
import glob
import os

import _bootstrap  # noqa: F401
from kupefdx.config import load_config
from kupefdx.env import log
from kupefdx.jsonl import read_manifest, write_manifest


def _pull_hub_manifests(cfg):
    from kupefdx.env import hf_login, require_token
    hf_login()
    from huggingface_hub import snapshot_download
    log.info("pulling encoded/**/manifest.jsonl from %s (feats.npz NOT downloaded)", cfg.repos.data)
    snapshot_download(
        cfg.repos.data, repo_type="dataset", local_dir=cfg.paths.data_dir,
        allow_patterns=["encoded/**/manifest.jsonl", "ledger/**"],
        token=require_token(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/en.yaml")
    ap.add_argument("--out", default="data/manifests/train.jsonl")
    ap.add_argument("--from-hub", action="store_true",
                    help="download encoded/**/manifest.jsonl from the data repo first")
    a = ap.parse_args()
    cfg = load_config(a.config)
    enc_dir = os.path.join(cfg.paths.data_dir, "encoded")
    if a.from_hub:
        _pull_hub_manifests(cfg)

    # feats.npz + manifest.jsonl are written side-by-side (in shards/<sid>/_hub/ locally,
    # or encoded/<sid>/ when pulled from the Hub) — search recursively and resolve feats
    # relative to each manifest's OWN directory, so both layouts work.
    shard_manifests = sorted(set(
        glob.glob(os.path.join(enc_dir, "**", "manifest.jsonl"), recursive=True)))
    if not shard_manifests:
        raise SystemExit(
            f"no manifest.jsonl found under {enc_dir}/\n"
            "  disk full?  rm -rf ~/.cache/huggingface/datasets data/raw data/hubbatch_* /tmp/*\n"
            "  then either git pull (local --no-flush shards live in shards/<sid>/_hub/)\n"
            "  or: python scripts/12_build_manifest.py --config configs/en.yaml --from-hub"
        )
    rows, n_feats, seen, dups = [], 0, set(), 0
    for mp in shard_manifests:
        base = os.path.dirname(mp)
        for r in read_manifest(mp):
            for key in ("feats", "codes"):            # resolve beside the manifest
                if r.get(key):
                    r[key] = os.path.join(base, os.path.basename(r[key]))
            # dedup by content (transcript + rounded duration) so old-namespace shards that
            # overlap the re-encoded ones don't get trained on twice.
            key = (r.get("text", "").strip().lower(), round(float(r.get("dur", 0)), 1))
            if key in seen:
                dups += 1
                continue
            seen.add(key)
            if r.get("feats") and os.path.isfile(r["feats"]):
                n_feats += 1
            rows.append(r)
    log.info("dropped %d duplicate clips (content dedup)", dups)
    write_manifest(a.out, rows)
    hrs = sum(r.get("dur", 0) for r in rows) / 3600
    log.info("consolidated %d shard manifests -> %s | %d clips | %.1f h | %d with cached feats",
             len(shard_manifests), a.out, len(rows), hrs, n_feats)
    log.info("set data.manifest: %s and data.use_cached_feats: %s in the config",
             a.out, "true" if n_feats else "false")
    if rows and n_feats == 0:
        log.warning("manifest has 0 local feats.npz — pull them before training:\n"
                    "  HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download %s "
                    "--repo-type dataset --local-dir data --include 'encoded/**'",
                    cfg.repos.data)


if __name__ == "__main__":
    main()
