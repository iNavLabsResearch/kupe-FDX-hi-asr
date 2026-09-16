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
    "common_voice_hi": ("mozilla-foundation/common_voice_16_1", "hi", "train", "audio", "sentence"),
    # add Shrutilipi / IndicVoices / Kathbath adapters here (same 5-tuple shape).
}


def _split_of(cid, val=0.02, test=0.02):
    h = int(hashlib.sha1(cid.encode()).hexdigest(), 16) % 10000 / 10000.0
    return SPLIT_TEST if h < test else SPLIT_VAL if h < test + val else SPLIT_TRAIN


def _keep(text, dur):
    return 0.5 <= dur <= 30.0 and len(normalize(text)) >= 2


def _detect_cols(ex, acol, tcol):
    """Auto-find the audio + text columns from one example, so any ASR dataset works
    without passing column names. Audio = a dict with array/sampling_rate/path; text =
    a preferred name, else the first string field."""
    if acol not in ex or not isinstance(ex.get(acol), dict):
        for k, v in ex.items():
            if isinstance(v, dict) and ({"array", "sampling_rate", "path"} & set(v)):
                acol = k
                break
    if tcol not in ex or not isinstance(ex.get(tcol), str):
        pref = ["transcript", "text", "sentence", "transcription", "normalized_text",
                "raw_text", "clean_text", "transcription_v1", "transcription_v2"]
        cand = [k for k in pref if isinstance(ex.get(k), str)] or \
               [k for k, v in ex.items() if isinstance(v, str) and k != acol and len(v) > 1]
        if cand:
            tcol = cand[0]
    if acol not in ex or tcol not in ex:
        raise SystemExit(f"could not auto-detect audio/text columns; keys={list(ex.keys())}")
    return acol, tcol


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
        if a.hf_id:
            ds_id, cfg_name, split, acol, tcol = a.hf_id, a.hf_config, a.split, a.audio_col, a.text_col
        else:
            ds_id, cfg_name, split, acol, tcol = SOURCES[a.hf]
        try:
            ds = load_dataset(ds_id, cfg_name, split=split, streaming=True, trust_remote_code=True)
        except TypeError:
            ds = load_dataset(ds_id, cfg_name, split=split, streaming=True)
        from tqdm import tqdm
        try:                                       # total clips for the bar (done/left/ETA)
            from datasets import load_dataset_builder
            total = load_dataset_builder(ds_id, cfg_name).info.splits[split].num_examples
        except Exception:
            total = None
        name = (a.hf or a.hf_id).replace("/", "_")
        log.info("streaming %s (first shard downloads the archive — hold on)...", name)
        # peek one example → auto-detect columns → stitch it back so no clip is lost
        it = iter(ds)
        try:
            first = next(it)
        except StopIteration:
            log.warning("%s is empty — nothing to stream", name)
            return
        acol, tcol = _detect_cols(first, acol, tcol)
        log.info("columns: audio=%s text=%s", acol, tcol)
        stream = itertools.chain([first], it)
        bar = tqdm(total=total, unit="clip", desc=name, dynamic_ncols=True)
        rows, si, j, kept_h = [], 0, 0, 0.0
        for ex in stream:                          # incremental: save each clip as it arrives
            bar.update(1); j += 1
            text = normalize(str(ex[tcol]))
            aud = ex[acol]
            if not isinstance(aud, dict) or "array" not in aud:
                continue
            arr = np.asarray(aud["array"], dtype="float32")
            sr = aud["sampling_rate"]
            if sr != SAMPLE_RATE:
                arr = _resample(arr, sr, SAMPLE_RATE)
            dur = len(arr) / SAMPLE_RATE
            if not _keep(text, dur):
                continue
            cid = f"{name}_{si:04d}_{j:06d}"
            p = os.path.join(raw_dir, cid + ".wav")
            save_wav(p, arr)
            rows.append({"id": cid, "audio": p, "text": text, "dur": dur,
                         "domain": a.domain, "split": _split_of(cid)})
            kept_h += dur / 3600
            bar.set_postfix(shard=si, in_shard=len(rows), kept_h=f"{kept_h:.1f}")
            if len(rows) >= a.shard_size:
                yield si, rows
                rows, si = [], si + 1
        if rows:
            yield si, rows
        bar.close()


def main():
    global a_cfg
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpu.yaml")
    ap.add_argument("--hf", default=None, choices=list(SOURCES))
    # generic HF source (any dataset): overrides --hf shortcuts
    ap.add_argument("--hf-id", default=None, help="any HF dataset id, e.g. ai4bharat/indicvoices")
    ap.add_argument("--hf-config", default=None, help="dataset config/subset, e.g. hi")
    ap.add_argument("--split", default="train")
    ap.add_argument("--audio-col", default="audio")
    ap.add_argument("--text-col", default="sentence")
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--domain", default="general")
    ap.add_argument("--shard-size", type=int, default=500)
    ap.add_argument("--shard-start", type=int, default=0, help="process shards where idx%%stride==shard-start")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-flush", action="store_true", help="keep local shard files (debug)")
    ap.add_argument("--raw-only", action="store_true",
                    help="download + push RAW audio only (skip the encoder); encode later")
    ap.add_argument("--peek", action="store_true",
                    help="print one example's column names (find --audio-col/--text-col) and exit")
    a = ap.parse_args()
    cfg = a_cfg = load_config(a.config)
    if not a.hf and not a.hf_id and not a.local_dir:
        raise SystemExit("give --hf or --local-dir")

    if a.peek:
        from datasets import load_dataset
        hf_login()
        ds_id = a.hf_id or SOURCES[a.hf][0]
        cfg_name = a.hf_config if a.hf_id else (SOURCES[a.hf][1] if a.hf else None)
        ds = load_dataset(ds_id, cfg_name, split=a.split, streaming=True)
        ex = next(iter(ds))
        log.info("COLUMNS for %s/%s:", ds_id, cfg_name)
        for k, v in ex.items():
            extra = f" -> keys {list(v.keys())}" if isinstance(v, dict) else f" ({type(v).__name__})"
            log.info("   %-20s%s", k, extra)
        log.info("use --audio-col <the audio key> --text-col <the text key>")
        return
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
    src = (a.hf or a.hf_id or a.local_dir).replace("/", "_")
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
            # 3) push RAW (always) + ENCODED (unless raw-only) — ONE folder commit each (fast)
            raw_stage = os.path.join(work, "raw")
            os.makedirs(raw_stage, exist_ok=True)
            for r in rows:
                shutil.copy2(r["audio"], os.path.join(raw_stage, os.path.basename(r["audio"])))
            log.info("shard %s: %d clips downloaded -> uploading raw...", sid, len(rows))
            upload_folder(raw_stage, cfg.repos.data, "dataset", path_in_repo=f"raw/{sid}",
                          commit_message=f"raw {sid}")
            shutil.rmtree(raw_stage, ignore_errors=True)
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
