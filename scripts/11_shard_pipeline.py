#!/usr/bin/env python3
"""Stage 1+2 fused, SHARDED and streaming — the correct disk-light flow:

  for each shard:  download  ->  encode (feats + codes)  ->  push RAW + ENCODED to HF  ->  FLUSH local

So raw and encoded land on the Hub continuously and local disk never fills. Resumable per
shard (a finished shard is skipped). Run several in parallel across GPUs/processes with
--shard-start / --stride (e.g. two boxes: start 0 stride 2, and start 1 stride 2) so the
"next shards" are processed meanwhile.

    # one box, all shards, 500 clips/shard:
    python scripts/11_shard_pipeline.py --config configs/gpu.yaml --hf fleurs_hi --shard-size 500

    # two GPUs in parallel (interleaved shards):
    CUDA_VISIBLE_DEVICES=0 python scripts/11_shard_pipeline.py --config configs/gpu.yaml --hf shrutilipi_hi --shard-size 500 --shard-start 0 --stride 2 &
    CUDA_VISIBLE_DEVICES=1 python scripts/11_shard_pipeline.py --config configs/gpu.yaml --hf shrutilipi_hi --shard-size 500 --shard-start 1 --stride 2 &

    # local dir of <name>.wav + <name>.txt pairs:
    python scripts/11_shard_pipeline.py --config configs/gpu.yaml --local-dir /data/hi --domain medical --shard-size 500
"""
import argparse
import glob
import hashlib
import itertools
import os
import shutil

import numpy as np
import torch

import _bootstrap  # noqa: F401
from kupefdx.audio import _resample, duration_s, load_wav, save_wav
from kupefdx.config import load_config
from kupefdx.constants import SAMPLE_RATE, SPLIT_TEST, SPLIT_TRAIN, SPLIT_VAL
from kupefdx.dataset import write_manifest
from kupefdx.encoders import build_encoder
from kupefdx.env import device_auto, ensure_repo, hf_login, log, require_token, upload_file, upload_folder
from kupefdx.ledger import ShardLedger
from kupefdx.quantizer import KMeansQuantizer
from kupefdx.text import normalize

SOURCES = {
    "fleurs_hi": ("google/fleurs", "hi_in", "train", "audio", "transcription"),
    "common_voice_hi": ("mozilla-foundation/common_voice_17_0", "hi", "train", "audio", "sentence"),
    # add Shrutilipi / IndicVoices / Kathbath adapters here (same 5-tuple shape).
}


def _split_of(cid, val=0.02, test=0.02):
    h = int(hashlib.sha1(cid.encode()).hexdigest(), 16) % 10000 / 10000.0
    return SPLIT_TEST if h < test else SPLIT_VAL if h < test + val else SPLIT_TRAIN


def _keep(text, dur):
    return 0.5 <= dur <= 30.0 and len(normalize(text)) >= 2


def shard_stream(a):
    """Yield (shard_idx, [rows]) where rows have local wav paths already written."""
    raw_dir = os.path.join(a_cfg.paths.raw_dir, "wavs")
    os.makedirs(raw_dir, exist_ok=True)
    if a.local_dir:
        wavs = sorted(glob.glob(os.path.join(a.local_dir, "*.wav")))
        for si, i in enumerate(range(0, len(wavs), a.shard_size)):
            rows = []
            for wav in wavs[i:i + a.shard_size]:
                txt = wav[:-4] + ".txt"
                if not os.path.isfile(txt):
                    continue
                text = normalize(open(txt, encoding="utf-8").read())
                dur = duration_s(wav)
                if _keep(text, dur):
                    cid = "loc_" + hashlib.sha1(wav.encode()).hexdigest()[:12]
                    rows.append({"id": cid, "audio": wav, "text": text, "dur": dur,
                                 "domain": a.domain, "split": _split_of(cid)})
            yield si, rows
    else:
        from datasets import load_dataset
        ds_id, cfg_name, split, acol, tcol = SOURCES[a.hf]
        ds = load_dataset(ds_id, cfg_name, split=split, streaming=True)
        it = iter(ds)
        si = 0
        while True:
            batch = list(itertools.islice(it, a.shard_size))
            if not batch:
                break
            rows = []
            for j, ex in enumerate(batch):
                text = normalize(ex[tcol])
                arr = np.asarray(ex[acol]["array"], dtype="float32")
                sr = ex[acol]["sampling_rate"]
                if sr != SAMPLE_RATE:
                    arr = _resample(arr, sr, SAMPLE_RATE)
                dur = len(arr) / SAMPLE_RATE
                if not _keep(text, dur):
                    continue
                cid = f"{a.hf}_{si:04d}_{j:04d}"
                p = os.path.join(raw_dir, cid + ".wav")
                save_wav(p, arr)
                rows.append({"id": cid, "audio": p, "text": text, "dur": dur,
                             "domain": a.domain, "split": _split_of(cid)})
            yield si, rows
            si += 1


def main():
    global a_cfg
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpu.yaml")
    ap.add_argument("--hf", default=None, choices=list(SOURCES))
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--domain", default="general")
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--shard-start", type=int, default=0, help="process shards where idx%%stride==shard-start")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-flush", action="store_true", help="keep local shard files (debug)")
    ap.add_argument("--raw-only", action="store_true",
                    help="download + push RAW audio only (skip the encoder); encode later")
    a = ap.parse_args()
    cfg = a_cfg = load_config(a.config)
    if not a.hf and not a.local_dir:
        raise SystemExit("give --hf or --local-dir")
    dev = device_auto()
    hf_login()
    ensure_repo(cfg.repos.data, "dataset")

    enc = None if a.raw_only else build_encoder(cfg).to(dev).eval()
    n_codes = 0 if a.raw_only else int(getattr(cfg.audio, "n_codes", 0))
    qpath = os.path.join(cfg.paths.data_dir, "encoded", "quantizer.pt")
    os.makedirs(os.path.dirname(qpath), exist_ok=True)
    quant = KMeansQuantizer.load(qpath) if (n_codes > 0 and os.path.isfile(qpath)) else None

    led = ShardLedger(os.path.join(cfg.paths.ledger_dir, "shardpipe.json"), "shardpipe",
                      repo_id=cfg.repos.data)
    src = a.hf or a.local_dir
    total_h = led.total_meta("hours")

    @torch.no_grad()
    def feats_of(path):
        w = load_wav(path)
        f, fl = enc.features(torch.from_numpy(w)[None].to(dev), torch.tensor([len(w)], device=dev))
        return f[0, : int(fl[0])].float().cpu().numpy()

    for si, rows in shard_stream(a):
        if (si % a.stride) != a.shard_start:
            continue
        sid = f"{src}_shard_{si:05d}"
        if led.is_done(sid):
            log.info("shard %s already done — skip", sid)
            continue
        if not rows:
            continue
        work = os.path.join(cfg.paths.data_dir, "encoded", "shards", sid)
        os.makedirs(work, exist_ok=True)
        try:
            # 1) encode feats (+ fit quantizer on the very first shard if codes enabled)
            if not a.raw_only:
              for r in rows:
                f = feats_of(r["audio"])
                np.save(os.path.join(work, r["id"] + ".feats.npy"), f.astype(np.float16))
                r["feats"] = f"shards/{sid}/{r['id']}.feats.npy"
            if not a.raw_only and n_codes > 0 and quant is None:
                pool = np.concatenate([np.load(os.path.join(work, r["id"] + ".feats.npy")).astype(np.float32)
                                       for r in rows[:64]], 0)
                quant = KMeansQuantizer.fit(pool, n_codes, iters=25)
                quant.save(qpath)
                upload_file(qpath, cfg.repos.data, "dataset", "encoded/quantizer.pt", "quantizer")
                log.info("fitted + pushed quantizer (%d codes)", n_codes)
            if not a.raw_only and n_codes > 0:
                for r in rows:
                    f = np.load(os.path.join(work, r["id"] + ".feats.npy")).astype(np.float32)
                    np.save(os.path.join(work, r["id"] + ".codes.npy"), quant.encode(f).astype(np.int64))
                    r["codes"] = f"shards/{sid}/{r['id']}.codes.npy"
            # 2) shard manifest
            write_manifest(os.path.join(work, "manifest.jsonl"), rows)
            # 3) push RAW (always) + ENCODED (unless raw-only)
            for r in rows:                          # raw wavs
                upload_file(r["audio"], cfg.repos.data, "dataset",
                            f"raw/{sid}/{os.path.basename(r['audio'])}", f"raw {sid}")
            sub = "manifests" if a.raw_only else "encoded"
            upload_folder(work, cfg.repos.data, "dataset", path_in_repo=f"{sub}/{sid}",
                          commit_message=f"{sub} {sid}")
            # 4) ledger + hours
            sh = sum(r["dur"] for r in rows) / 3600
            total_h += sh
            led.mark(sid, "done", clips=len(rows), hours=sh)
            led.push(f"shardpipe {sid} done ({total_h:.1f} h total)")
            log.info("shard %s: %d clips, %.2f h | cumulative %.1f h", sid, len(rows), sh, total_h)
        finally:
            # 5) FLUSH local (raw + encoded) to keep disk light
            if not a.no_flush:
                shutil.rmtree(work, ignore_errors=True)
                for p in [r.get("audio") for r in rows]:
                    if p and os.path.isfile(p):
                        os.remove(p)

    log.info("DONE. total pushed: %.1f h | ledger: %s", total_h, led.counts())


if __name__ == "__main__":
    main()
