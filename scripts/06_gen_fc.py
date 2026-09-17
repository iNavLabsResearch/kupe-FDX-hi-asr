#!/usr/bin/env python3
"""Stage 6 — generate floor-control training data from real ASR clips.

Reads a source ASR manifest (audio+transcript), reads each wav to extract pause/timing
features, and drives an LLM agent (concurrency, tqdm, ~20-25 rows/hit) to emit structured,
audio-grounded floor-control rows across scenarios with a controlled distribution. Output
is a training-ready JSONL manifest (audio + target_sequence with inline flags). Resumable
per batch; optionally pushed to the Hub.

    # offline dry-run (no keys) — proves the whole generator end to end
    python scripts/06_gen_fc.py --config configs/gpu.yaml --src data/manifests/train.jsonl \
        --out data/manifests/fc.jsonl --mock --limit 50

    # real run (OpenAI-compatible endpoint)
    export KUPE_LLM_BASE_URL=... KUPE_LLM_API_KEY=... KUPE_LLM_MODEL=...
    python scripts/06_gen_fc.py --config configs/gpu.yaml --src data/manifests/train.jsonl \
        --out data/manifests/fc.jsonl --concurrency 10 --rows-per-hit 22
"""
import argparse
import os
import time

import _bootstrap  # noqa: F401
from kupefdx.config import load_config
from kupefdx.constants import SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST
from kupefdx.dataset import read_manifest, write_manifest
from kupefdx.env import log
from kupefdx.fcgen.agent import generate
from kupefdx.fcgen.audio_probe import audio_card, probe
from kupefdx.fcgen.scenarios import distribution_table, rebalance, set_weights
from kupefdx.ledger import ShardLedger


def build_clips(src_rows, limit, cache_path=None):
    from concurrent.futures import ThreadPoolExecutor
    from kupefdx.jsonl import read_manifest as _rm, write_manifest as _wm
    rows = [r for r in src_rows[: limit or None] if os.path.isfile(r["audio"])]
    missing = len(src_rows[: limit or None]) - len(rows)
    n = len(rows)
    # reuse a previous probe so repeated restarts don't re-probe every wav
    if cache_path and os.path.isfile(cache_path):
        cached = {c["id"]: c for c in _rm(cache_path)}
        if all(r["id"] in cached for r in rows):
            log.info("loaded %d probed clips from cache %s (skip re-probe)", len(rows), cache_path)
            return [cached[r["id"]] for r in rows]
    workers = min(48, max(4, (os.cpu_count() or 8)))
    log.info("probing %d clips for pause/timing (%d workers)%s ...", n, workers,
             f"  [{missing} skipped: no local wav]" if missing else "")

    def _one(r):
        feats = probe(r["audio"])
        return {"id": r["id"], "audio": r["audio"], "transcript": r["text"],
                "domain": r.get("domain", "general"), "features": feats,
                "card": audio_card(feats, r["text"])}

    clips, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, c in enumerate(ex.map(_one, rows), 1):
            clips.append(c)
            if i % 500 == 0 or i == n:
                rate = i / max(1e-6, time.time() - t0)
                log.info("  probed %d/%d clips (%.0f/s)", i, n, rate)
    if cache_path:
        try:
            _wm(cache_path, clips)
            log.info("cached probed clips -> %s (reused on restart)", cache_path)
        except Exception as e:
            log.warning("probe-cache write failed: %s", e)
    return clips


def _split_of(i, n):
    if i < int(0.9 * n):
        return SPLIT_TRAIN
    return SPLIT_VAL if i < int(0.95 * n) else SPLIT_TEST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpu.yaml")
    ap.add_argument("--src", required=True, help="source ASR manifest (audio+text)")
    ap.add_argument("--out", required=True, help="output FC manifest JSONL")
    ap.add_argument("--mock", action="store_true", help="offline deterministic generation")
    ap.add_argument("--limit", type=int, default=0, help="cap source clips (0 = all)")
    ap.add_argument("--rows-per-hit", type=int, default=22)
    ap.add_argument("--clips-per-hit", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=0,
                    help="in-flight LLM calls (default 10; forced to 1 under --show-stream)")
    ap.add_argument("--show-stream", action="store_true",
                    help="print the model's live SSE token stream to stdout; runs one call at a "
                         "time so the output is readable (override with an explicit --concurrency)")
    ap.add_argument("--no-push", action="store_true", help="do NOT auto-sync the result to the Hub")
    # floor-control needs CONVERSATIONAL clips; lecture monologues (NPTEL) teach bad turn-taking.
    ap.add_argument("--exclude-domains", default="indian_english,read_us",
                    help="comma domains to skip for FC gen (default skips lectures + read speech)")
    ap.add_argument("--include-domains", default="",
                    help="comma domains to KEEP (overrides exclude); e.g. spontaneous,accented")
    a = ap.parse_args()
    # --show-stream is a "watch one call live" mode -> default to sequential so SSE tokens
    # from different hits don't interleave; an explicit --concurrency still wins.
    a.concurrency = a.concurrency or (1 if a.show_stream else 10)
    cfg = load_config(a.config)

    fc_cfg = getattr(cfg, "fc", None)
    if fc_cfg is not None and getattr(fc_cfg, "distribution", None) is not None:
        set_weights(fc_cfg.distribution.to_dict())
        log.info("scenario distribution overridden from config")

    log.info("target scenario distribution:")
    for name, pct, desc in distribution_table():
        log.info("  %-20s %4.1f%%  %s", name, pct, desc)

    src_rows = read_manifest(a.src)
    # keep only conversational clips for floor-control (exclude monotonous lectures/read speech)
    inc = {d.strip() for d in a.include_domains.split(",") if d.strip()}
    exc = {d.strip() for d in a.exclude_domains.split(",") if d.strip()}
    n0 = len(src_rows)
    if inc:
        src_rows = [r for r in src_rows if r.get("domain") in inc]
    else:
        src_rows = [r for r in src_rows if r.get("domain") not in exc]
    log.info("FC source filter: %d -> %d clips (include=%s exclude=%s)",
             n0, len(src_rows), inc or "-", exc or "-")
    if not src_rows:
        raise SystemExit("no conversational clips left after domain filter — check --include/--exclude-domains")
    clips = build_clips(src_rows, a.limit, cache_path=a.src + ".cards.jsonl")
    log.info("probed %d clips from %s", len(clips), a.src)

    led = ShardLedger(os.path.join(cfg.paths.ledger_dir, "fcgen.json"), "fcgen")
    existing = read_manifest(a.out) if os.path.isfile(a.out) else []

    def _push(rows):  # incremental append so a crash never loses generated rows
        nonlocal existing
        existing += rows
        write_manifest(a.out, existing)

    rows = generate(clips, rows_per_hit=a.rows_per_hit, clips_per_hit=a.clips_per_hit,
                    concurrency=a.concurrency, mock=a.mock, seen_ledger=led, push_cb=_push,
                    show_stream=a.show_stream)
    all_rows = existing
    all_rows = rebalance(all_rows)
    for i, r in enumerate(all_rows):
        r["split"] = _split_of(i, len(all_rows))
    write_manifest(a.out, all_rows)

    from collections import Counter
    c = Counter(r["scenario"] for r in all_rows)
    fc_cnt = Counter(f for r in all_rows for f in r.get("flags", []))
    log.info("wrote %d FC rows -> %s", len(all_rows), a.out)
    log.info("realized scenarios: %s", dict(c))
    log.info("flag counts: %s | no-flag rows: %d",
             dict(fc_cnt), sum(1 for r in all_rows if not r.get("flags")))

    if not a.no_push and not a.mock and all_rows:
        try:
            from kupefdx.env import ensure_repo, hf_login, upload_file
            hf_login(); ensure_repo(cfg.repos.data, "dataset")
            dest = f"manifests/{os.path.basename(a.out)}"
            upload_file(a.out, cfg.repos.data, "dataset", dest, "sync fc data")
            log.info("auto-synced %d rows -> %s:%s", len(all_rows), cfg.repos.data, dest)
        except Exception as e:
            log.warning("auto-sync to Hub failed (data is safe locally): %s", e)


if __name__ == "__main__":
    main()
