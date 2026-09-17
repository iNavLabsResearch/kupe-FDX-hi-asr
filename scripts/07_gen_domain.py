#!/usr/bin/env python3
"""Stage 7 — generate domain-correction training data (Phase 5).

Given ASR clips, produce records of {domain, omni_raw_transcript, context,
corrected_transcript, correction_spans, chunk_boundaries_ms}, converted to training rows
(text=raw, target_sequence=corrected, context conditions the decoder).

    python scripts/07_gen_domain.py --config configs/gpu.yaml --src data/manifests/train.jsonl \
        --out data/manifests/domain.jsonl --mock --limit 100
"""
import argparse
import os

import _bootstrap  # noqa: F401
from kupefdx.config import load_config
from kupefdx.dataset import read_manifest, write_manifest
from kupefdx.env import log
from kupefdx.fcgen.domain import generate_domain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpu.yaml")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--no-push", action="store_true", help="do NOT auto-sync the result to the Hub")
    a = ap.parse_args()
    cfg = load_config(a.config)

    src = read_manifest(a.src)[: a.limit or None]
    clips = [{"id": r["id"], "audio": r["audio"], "transcript": r["text"],
              "domain": r.get("domain", "general")} for r in src if os.path.isfile(r["audio"])]
    log.info("domain-correction gen from %d clips (mock=%s)", len(clips), a.mock)

    existing = read_manifest(a.out) if os.path.isfile(a.out) else []

    def _push(rows):
        nonlocal existing
        existing += rows
        write_manifest(a.out, existing)

    generate_domain(clips, concurrency=a.concurrency, mock=a.mock, push_cb=_push)
    for i, r in enumerate(existing):
        r["split"] = "train" if i < int(0.9 * len(existing)) else "val"
    write_manifest(a.out, existing)
    log.info("wrote %d domain-correction rows -> %s", len(existing), a.out)

    if not a.no_push and not a.mock and existing:
        try:
            from kupefdx.env import ensure_repo, hf_login, upload_file
            hf_login(); ensure_repo(cfg.repos.data, "dataset")
            dest = f"manifests/{os.path.basename(a.out)}"
            upload_file(a.out, cfg.repos.data, "dataset", dest, "sync domain data")
            log.info("auto-synced %d rows -> %s:%s", len(existing), cfg.repos.data, dest)
        except Exception as e:
            log.warning("auto-sync to Hub failed (data is safe locally): %s", e)


if __name__ == "__main__":
    main()
