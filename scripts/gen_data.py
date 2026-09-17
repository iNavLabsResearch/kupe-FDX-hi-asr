#!/usr/bin/env python3
"""Generate ALL LLM training data in ONE command — floor-control (FC) + domain-correction.

Reads a source ASR manifest (audio + transcript), probes each clip for pause/timing, and
drives the LLM (Krutrim gemma-4-31b-it by default) to emit:
  * data/manifests/fc.jsonl      floor-control rows (audio-grounded flags/scenarios)
  * data/manifests/domain.jsonl  domain-correction rows (raw -> corrected transcript)

Every request prints a COLORED line in real time (green OK / red FAIL) with latency, rows
kept, and running token/cost totals. Output is streamed to disk per request and mirrored
to the Hub every --sync-every requests, so a crash or dropped SSH never loses work. Both
generators are resumable per batch (ledger), so re-running skips finished requests.

    # offline dry-run (no keys) — proves the whole generator end to end
    python scripts/gen_data.py --config configs/en.yaml --src data/manifests/train.jsonl --mock --limit 50

    # real run (uses KUPE_LLM_* from .env)
    python scripts/gen_data.py --config configs/en.yaml --src data/manifests/train.jsonl --concurrency 40

    # watch one call's live SSE token stream
    python scripts/gen_data.py --config configs/en.yaml --src data/manifests/train.jsonl \
        --only fc --limit 3 --show-stream --no-push
"""
import argparse
import os
import time

import _bootstrap  # noqa: F401
from kupefdx.config import load_config
from kupefdx.dataset import read_manifest, write_manifest
from kupefdx.fcgen.audio_probe import audio_card, probe
from kupefdx.fcgen.generate import C, cprint, generate_domain, generate_fc
from kupefdx.fcgen.scenarios import (distribution_table, rebalance, set_weights)
from kupefdx.ledger import ShardLedger


def _synth_features(dur: float) -> dict:
    """Duration-only feature card when the raw wav is not on this box (feats-only pull).
    No real pause grounding — the LLM places flags from transcript + duration."""
    return {"duration_s": round(float(dur), 2), "sample_rate": 16000, "n_pauses": 0,
            "pauses": [], "leading_silence_s": 0.0, "trailing_silence_s": 0.0,
            "speech_fraction": 1.0, "mean_energy_db": -20.0}


def _hub_raw_index(cfg):
    """Set of raw/* wav paths that exist on the data repo (empty if raw was never pushed)."""
    try:
        from huggingface_hub import HfApi
        from kupefdx.env import require_token
        api = HfApi()
        return {f for f in api.list_repo_files(cfg.repos.data, repo_type="dataset",
                                               token=require_token()) if f.startswith("raw/")}
    except Exception as e:
        cprint(C.WARN, f"could not check Hub for raw audio: {e}")
        return set()


def _hub_path(audio: str) -> str:
    """Local `data/raw/wavs/...wav` -> Hub `raw/wavs/...wav`."""
    return audio.split("data/", 1)[-1] if audio.startswith("data/") else audio


def _fetch_wav(cfg, audio: str, raw_set: set) -> bool:
    """Download one raw wav from the Hub into its manifest path if available. True on success."""
    hp = _hub_path(audio)
    if hp not in raw_set:
        return False
    try:
        from huggingface_hub import hf_hub_download
        from kupefdx.env import require_token
        hf_hub_download(cfg.repos.data, hp, repo_type="dataset",
                        local_dir=cfg.paths.data_dir, token=require_token())
        return os.path.isfile(audio)
    except Exception:
        return False


def probe_clips(rows, limit, cache_path, cfg=None, fetch=True):
    """Build one clip card per row (reused by FC and domain). If the raw wav is local we
    probe it (native-rate read + vectorized VAD, ~1 ms/clip). If not, we auto-fetch it from
    the Hub when raw audio is available there; otherwise we fall back to a duration-only card
    so generation still runs on a feats-only box. Cached to disk."""
    from concurrent.futures import ThreadPoolExecutor
    from kupefdx.jsonl import read_manifest as _rm, write_manifest as _wm
    rows = rows[: limit or None]
    n = len(rows)
    if cache_path and os.path.isfile(cache_path):
        cached = {c["id"]: c for c in _rm(cache_path)}
        if all(r["id"] in cached for r in rows):
            cprint(C.INFO, f"loaded {n} probed clips from cache (skip re-probe)")
            return [cached[r["id"]] for r in rows]

    # auto-fetch: only attempt if raw audio actually exists on the Hub (one list call, not N 404s)
    raw_set = set()
    have = sum(1 for r in rows if os.path.isfile(r["audio"]))
    if fetch and have < n and cfg is not None:
        raw_set = _hub_raw_index(cfg)
        if raw_set:
            cprint(C.INFO, f"auto-fetching missing wavs from {cfg.repos.data} (raw available) ...")
        else:
            cprint(C.WARN, f"{n - have}/{n} clips have no local wav and the Hub has NO raw audio "
                   "-> duration-only cards. For real pause grounding, generate on the box with "
                   "data/raw/, or push raw (PUSH_RAW=1 gather).")
    workers = min(64, max(4, (os.cpu_count() or 8) * 2))
    cprint(C.INFO, f"building {n} clip cards ({workers} workers) ...")

    def one(r):
        if not os.path.isfile(r["audio"]) and raw_set:
            _fetch_wav(cfg, r["audio"], raw_set)
        f = probe(r["audio"]) if os.path.isfile(r["audio"]) else _synth_features(r.get("dur", 0))
        return {"id": r["id"], "audio": r["audio"], "transcript": r["text"],
                "domain": r.get("domain", "general"), "features": f,
                "card": audio_card(f, r["text"])}

    clips, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, c in enumerate(ex.map(one, rows), 1):
            clips.append(c)
            if i % 2000 == 0 or i == n:
                cprint(C.DIM, f"  probed {i}/{n} ({i/max(1e-6, time.time()-t0):.0f}/s)")
    if cache_path:
        try:
            _wm(cache_path, clips)
        except Exception:
            pass
    return clips


def hub_syncer(cfg, path, no_push, mock, label):
    """Return a sync_cb(done, total) that uploads `path` to the data repo, or None."""
    if no_push or mock:
        return None
    from kupefdx.env import ensure_repo, hf_login, upload_file
    dest = f"manifests/{os.path.basename(path)}"
    ready = {"ok": False}

    def sync(done, total):
        try:
            if not ready["ok"]:
                hf_login(); ensure_repo(cfg.repos.data, "dataset"); ready["ok"] = True
            upload_file(path, cfg.repos.data, "dataset", dest, f"sync {label} ({done}/{total})")
            cprint(C.INFO, f"↑ synced {label} -> {cfg.repos.data}:{dest} ({done}/{total} requests)")
        except Exception as e:
            cprint(C.WARN, f"Hub sync failed (data safe locally): {e}")
    return sync


def run_fc(cfg, clips, a):
    if getattr(getattr(cfg, "fc", None), "distribution", None) is not None:
        set_weights(cfg.fc.distribution.to_dict())
    cprint(C.INFO + C.BOLD, "FC scenario distribution:")
    for name, pct, desc in distribution_table():
        cprint(C.DIM, f"  {name:20s} {pct:4.1f}%  {desc}")
    led = ShardLedger(os.path.join(cfg.paths.ledger_dir, "fcgen.json"), "fcgen")
    existing = read_manifest(a.fc_out) if os.path.isfile(a.fc_out) else []

    def push(rows):
        existing.extend(rows); write_manifest(a.fc_out, existing)

    generate_fc(clips, rows_per_hit=a.rows_per_hit, clips_per_hit=a.clips_per_hit,
                concurrency=a.concurrency, mock=a.mock, seen_ledger=led, push_cb=push,
                sync_cb=hub_syncer(cfg, a.fc_out, a.no_push, a.mock, "fc"),
                sync_every=a.sync_every, show_stream=a.show_stream)
    rows = rebalance(existing)
    for i, r in enumerate(rows):
        r["split"] = "train" if i < int(0.9 * len(rows)) else ("val" if i < int(0.95 * len(rows)) else "test")
    write_manifest(a.fc_out, rows)
    cprint(C.OK + C.BOLD, f"wrote {len(rows)} FC rows -> {a.fc_out}")
    return rows


def run_domain(cfg, clips, a):
    led = ShardLedger(os.path.join(cfg.paths.ledger_dir, "domaingen.json"), "domaingen")
    existing = read_manifest(a.domain_out) if os.path.isfile(a.domain_out) else []

    def push(rows):
        existing.extend(rows); write_manifest(a.domain_out, existing)

    generate_domain(clips, clips_per_hit=a.clips_per_hit + 1, concurrency=a.concurrency,
                    mock=a.mock, seen_ledger=led, push_cb=push,
                    sync_cb=hub_syncer(cfg, a.domain_out, a.no_push, a.mock, "domain"),
                    sync_every=a.sync_every, show_stream=a.show_stream)
    for i, r in enumerate(existing):
        r["split"] = "train" if i < int(0.9 * len(existing)) else "val"
    write_manifest(a.domain_out, existing)
    cprint(C.OK + C.BOLD, f"wrote {len(existing)} domain rows -> {a.domain_out}")
    return existing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/en.yaml")
    ap.add_argument("--src", required=True, help="source ASR manifest (audio+text)")
    ap.add_argument("--only", choices=["both", "fc", "domain"], default="both")
    ap.add_argument("--fc-out", default="data/manifests/fc.jsonl")
    ap.add_argument("--domain-out", default="data/manifests/domain.jsonl")
    ap.add_argument("--mock", action="store_true", help="offline deterministic generation")
    ap.add_argument("--limit", type=int, default=0, help="cap source clips (0 = all)")
    ap.add_argument("--rows-per-hit", type=int, default=22)
    ap.add_argument("--clips-per-hit", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=0,
                    help="in-flight LLM calls (default 10; forced to 1 under --show-stream)")
    ap.add_argument("--sync-every", type=int, default=25,
                    help="mirror the manifest to the Hub every N requests (0 = only at end)")
    ap.add_argument("--show-stream", action="store_true",
                    help="print the live SSE token stream (runs one call at a time)")
    ap.add_argument("--no-push", action="store_true", help="never sync to the Hub")
    ap.add_argument("--no-fetch-audio", dest="fetch_audio", action="store_false",
                    help="do NOT auto-download missing raw wavs from the Hub (use duration-only)")
    # FC needs CONVERSATIONAL clips; lecture monologues teach bad turn-taking.
    ap.add_argument("--exclude-domains", default="indian_english,read_us")
    ap.add_argument("--include-domains", default="")
    a = ap.parse_args()
    a.concurrency = a.concurrency or (1 if a.show_stream else 10)
    cfg = load_config(a.config)

    src = read_manifest(a.src)
    inc = {d.strip() for d in a.include_domains.split(",") if d.strip()}
    exc = {d.strip() for d in a.exclude_domains.split(",") if d.strip()}
    fc_src = ([r for r in src if r.get("domain") in inc] if inc
              else [r for r in src if r.get("domain") not in exc])
    cprint(C.INFO, f"source: {len(src)} clips · FC-eligible (conversational): {len(fc_src)}")

    if a.only in ("both", "fc"):
        if not fc_src:
            raise SystemExit("no conversational clips for FC — check --include/--exclude-domains")
        clips = probe_clips(fc_src, a.limit, a.src + ".cards.jsonl", cfg=cfg, fetch=a.fetch_audio)
        run_fc(cfg, clips, a)
    if a.only in ("both", "domain"):
        # domain correction only needs the transcript (audio path is kept as a reference),
        # so it runs on the whole corpus even on a feats-only box.
        dclips = [{"id": r["id"], "audio": r["audio"], "transcript": r["text"],
                   "domain": r.get("domain", "general")}
                  for r in src[: a.limit or None]]
        run_domain(cfg, dclips, a)


if __name__ == "__main__":
    main()
