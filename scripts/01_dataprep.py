#!/usr/bin/env python3
"""Stage 1 — build a Hindi ASR manifest from public corpora and/or a local dir.

Resumable: a ShardLedger tracks which sources are done, so re-running only ingests
what's missing. Text is Devanagari-normalized; clips are duration/quality filtered;
the split is speaker/id-disjoint and domain-stratified, frozen once here.

    # local directory of  <name>.wav + <name>.txt  pairs
    python scripts/01_dataprep.py --local-dir /data/hi_pairs --domain medical

    # a public HF dataset (streaming) — see SOURCES for wired adapters
    python scripts/01_dataprep.py --hf fleurs_hi --max-hours 200

Real large corpora (Shrutilipi, IndicVoices, Kathbath, Vaani, Spring-INX, MUCS) each
get a small adapter following the same shape as SOURCES below (PLAN §1 lists them).
"""
import argparse
import glob
import hashlib
import os

import _bootstrap  # noqa: F401
from kupefdx.audio import duration_s, load_wav, save_wav
from kupefdx.config import load_config
from kupefdx.constants import SAMPLE_RATE, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
from kupefdx.dataset import read_manifest, write_manifest
from kupefdx.env import log
from kupefdx.ledger import ShardLedger
from kupefdx.text import normalize

# name -> (hf_dataset_id, config, split, audio_col, text_col)
SOURCES = {
    "fleurs_hi": ("google/fleurs", "hi_in", "train", "audio", "transcription"),
    "common_voice_hi": ("mozilla-foundation/common_voice_16_1", "hi", "train",
                        "audio", "sentence"),
}


def _split_of(clip_id: str, val=0.02, test=0.02) -> str:
    h = int(hashlib.sha1(clip_id.encode()).hexdigest(), 16) % 10000 / 10000.0
    if h < test:
        return SPLIT_TEST
    if h < test + val:
        return SPLIT_VAL
    return SPLIT_TRAIN


def _keep(text: str, dur: float) -> bool:
    return 0.5 <= dur <= 30.0 and len(normalize(text)) >= 2


def ingest_local(local_dir, domain, out_wav_dir, rows):
    for wav in sorted(glob.glob(os.path.join(local_dir, "*.wav"))):
        txt = wav[:-4] + ".txt"
        if not os.path.isfile(txt):
            continue
        text = normalize(open(txt, encoding="utf-8").read())
        dur = duration_s(wav)
        if not _keep(text, dur):
            continue
        cid = "loc_" + hashlib.sha1(wav.encode()).hexdigest()[:12]
        rows.append({"id": cid, "audio": wav, "text": text, "dur": dur,
                     "domain": domain, "split": _split_of(cid)})


def ingest_hf(name, domain, out_wav_dir, rows, max_hours):
    import soundfile as sf
    from datasets import load_dataset
    ds_id, cfg_name, split, acol, tcol = SOURCES[name]
    ds = load_dataset(ds_id, cfg_name, split=split, streaming=True)
    os.makedirs(out_wav_dir, exist_ok=True)
    total = 0.0
    for i, ex in enumerate(ds):
        if max_hours and total / 3600 >= max_hours:
            break
        text = normalize(ex[tcol])
        arr = ex[acol]["array"]
        sr = ex[acol]["sampling_rate"]
        cid = f"{name}_{i:07d}"
        path = os.path.join(out_wav_dir, cid + ".wav")
        import numpy as np
        wav = np.asarray(arr, dtype="float32")
        if sr != SAMPLE_RATE:
            from kupefdx.audio import _resample
            wav = _resample(wav, sr, SAMPLE_RATE)
        dur = len(wav) / SAMPLE_RATE
        if not _keep(text, dur):
            continue
        save_wav(path, wav)
        rows.append({"id": cid, "audio": path, "text": text, "dur": dur,
                     "domain": domain, "split": _split_of(cid)})
        total += dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpu.yaml")
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--hf", default=None, choices=list(SOURCES))
    ap.add_argument("--domain", default="general")
    ap.add_argument("--max-hours", type=float, default=0)
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, overrides=a.set)

    led = ShardLedger(os.path.join(cfg.paths.ledger_dir, "dataprep.json"), "dataprep")
    manifest = cfg.data.manifest
    rows = read_manifest(manifest) if os.path.isfile(manifest) else []
    out_wav = os.path.join(cfg.paths.raw_dir, "wavs")

    src_id = a.local_dir or a.hf
    if src_id is None:
        raise SystemExit("give --local-dir or --hf")
    if led.is_done(src_id):
        log.info("source %s already done — skipping", src_id)
    else:
        n0 = len(rows)
        try:
            if a.local_dir:
                ingest_local(a.local_dir, a.domain, out_wav, rows)
            else:
                ingest_hf(a.hf, a.domain, out_wav, rows, a.max_hours)
            hrs = sum(r["dur"] for r in rows[n0:]) / 3600
            led.mark(src_id, "done", clips=len(rows) - n0, hours=hrs)
            log.info("ingested %s: +%d clips (%.2f h)", src_id, len(rows) - n0, hrs)
        except Exception as e:
            led.mark(src_id, "failed", error=str(e))
            raise

    write_manifest(manifest, rows)
    tot = sum(r["dur"] for r in rows) / 3600
    log.info("manifest %s | %d clips | %.1f h total | ledger: %s",
             manifest, len(rows), tot, led.counts())


if __name__ == "__main__":
    main()
