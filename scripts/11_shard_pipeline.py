#!/usr/bin/env python3
"""Stage 1+2 fused — continuous pipeline (download ‖ encode ‖ batched Hub push):

  download → shard queue → GPU encode → upload batch queue → Hub (few commits)

Resume-safe: Hub-synced done shards are never re-encoded. Hub rate-limit (128
commits/hour) handled by batching + 429 retry — encode keeps going.

    ONLY=shrutilipi_hi bash scripts/gather_all.sh
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import itertools
import os
import shutil
import sys
import tempfile
import threading
import time
from queue import Queue

import numpy as np
import torch
from tqdm import tqdm

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
}

SENTINEL = object()


def _split_of(cid, val=0.02, test=0.02):
    h = int(hashlib.sha1(cid.encode()).hexdigest(), 16) % 10000 / 10000.0
    return SPLIT_TEST if h < test else SPLIT_VAL if h < test + val else SPLIT_TRAIN


def _keep(text, dur):
    return 0.5 <= dur <= 30.0 and len(normalize(text)) >= 2


def _keep_light(text, ex) -> bool:
    """Keep-check without decoding audio (for fast-skip of done shards)."""
    if len(normalize(text)) < 2:
        return False
    for k in ("duration", "duration_seconds", "duration_ms", "audio_duration"):
        if k in ex and ex[k] is not None:
            d = float(ex[k])
            if k.endswith("_ms"):
                d /= 1000.0
            return 0.5 <= d <= 30.0
    return True  # assume ok when skipping already-done region


def _detect_cols(ex, acol, tcol):
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


def _estimate_total(a):
    if a.local_dir:
        return len(glob.glob(os.path.join(a.local_dir, "*.wav"))) or None
    try:
        from datasets import load_dataset_builder
        ds_id = a.hf_id or SOURCES[a.hf][0]
        cfg_name = a.hf_config if a.hf_id else SOURCES[a.hf][1]
        return load_dataset_builder(ds_id, cfg_name).info.splits[a.split].num_examples
    except Exception:
        return None


def _sid(src_name, shard_size, si):
    return f"{src_name}_n{shard_size}_shard_{si:05d}"


def sync_done_from_hub(led: ShardLedger, repo_id: str, src_name: str, shard_size: int) -> int:
    """Mark any shard that already has feats.npz (or legacy .npy) on Hub as done."""
    try:
        from huggingface_hub import HfApi
        files = HfApi().list_repo_files(repo_id, repo_type="dataset", token=require_token())
    except Exception as e:
        tqdm.write(f"hub sync skipped: {e}")
        return 0
    prefix = f"encoded/{src_name}_n{shard_size}_shard_"
    # also older naming without n{size}
    prefix_old = f"encoded/{src_name}_shard_"
    found = set()
    for f in files:
        if "/feats.npz" in f or f.endswith(".feats.npy"):
            # encoded/<sid>/feats.npz  OR encoded/<sid>/<id>.feats.npy
            parts = f.split("/")
            if len(parts) >= 2 and parts[0] == "encoded":
                sid = parts[1]
                if sid.startswith(src_name):
                    found.add(sid)
    n = 0
    for sid in sorted(found):
        if not led.is_done(sid):
            led.mark(sid, "done", clips=0, hours=0, source="hub_sync")
            n += 1
    if n:
        led.save()
        tqdm.write(f"hub sync: marked {n} already-uploaded shards done (won't redo)")
    # count how many consecutive from 0 are done
    si = 0
    while led.is_done(_sid(src_name, shard_size, si)) or led.is_done(f"{src_name}_shard_{si:05d}"):
        si += 1
    if si:
        tqdm.write(f"resume after shard index {si - 1} → next todo = {_sid(src_name, shard_size, si)}")
    return si


def _upload_retry(folder, repo_id, path_in_repo, msg, max_sleep=600):
    """Upload with 429 backoff — never burn a GPU hour on a hard fail."""
    sleep_s = 60
    while True:
        try:
            upload_folder(folder, repo_id, "dataset", path_in_repo=path_in_repo,
                          commit_message=msg)
            return
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate limit" in err:
                tqdm.write(f"Hub 429 rate limit — sleep {sleep_s}s then retry ({path_in_repo})")
                time.sleep(sleep_s)
                sleep_s = min(max_sleep, int(sleep_s * 1.5))
                continue
            raise


def download_worker(a, cfg, src_name, led, dl_q: Queue, dl_bar: tqdm, err_box: list,
                    first_todo_si: int):
    """Stream HF → write wav shards. Skips done shards without decoding audio."""
    try:
        raw_root = os.path.join(cfg.paths.raw_dir, "wavs", src_name)
        os.makedirs(raw_root, exist_ok=True)

        def should_skip(si: int) -> bool:
            if (si % a.stride) != a.shard_start:
                return True
            return led.is_done(_sid(src_name, a.shard_size, si)) or \
                led.is_done(f"{src_name}_shard_{si:05d}")

        def emit(si, rows, end_j):
            sid = _sid(src_name, a.shard_size, si)
            if should_skip(si):
                for r in rows:
                    p = r.get("audio")
                    if p and os.path.isfile(p):
                        os.remove(p)
                return
            hours = sum(r["dur"] for r in rows) / 3600
            dl_q.put((si, sid, rows, hours, end_j))
            dl_bar.set_postfix(q=dl_q.qsize(), shard=si, kept_h=f"{hours:.1f}")

        if a.local_dir:
            wavs = sorted(glob.glob(os.path.join(a.local_dir, "*.wav")))
            for si, i in enumerate(range(0, len(wavs), a.shard_size)):
                if should_skip(si):
                    dl_bar.update(min(a.shard_size, len(wavs) - i))
                    continue
                rows = []
                for wav in wavs[i:i + a.shard_size]:
                    dl_bar.update(1)
                    txt = wav[:-4] + ".txt"
                    if not os.path.isfile(txt):
                        continue
                    text = normalize(open(txt, encoding="utf-8").read())
                    dur = duration_s(wav)
                    if _keep(text, dur):
                        cid = "loc_" + hashlib.sha1(wav.encode()).hexdigest()[:12]
                        rows.append({"id": cid, "audio": wav, "text": text, "dur": dur,
                                     "domain": a.domain, "split": _split_of(cid)})
                if rows:
                    emit(si, rows, i + len(wavs[i:i + a.shard_size]))
            return

        from datasets import load_dataset
        if a.hf_id:
            ds_id, cfg_name, split, acol, tcol = (
                a.hf_id, a.hf_config, a.split, a.audio_col, a.text_col)
        else:
            ds_id, cfg_name, split, acol, tcol = SOURCES[a.hf]
        try:
            ds = load_dataset(ds_id, cfg_name, split=split, streaming=True,
                              trust_remote_code=True)
        except TypeError:
            ds = load_dataset(ds_id, cfg_name, split=split, streaming=True)

        # Jump past done prefix using stored end_j when available.
        # Hub-synced shards often lack end_j — use shard_size * n as a safe lower bound
        # (never overshoots; remainder is light-skipped without audio decode).
        skip_until_j = 0
        if first_todo_si > 0:
            prev = _sid(src_name, a.shard_size, first_todo_si - 1)
            meta = led.d.get("meta", {}).get(prev, {})
            if not meta:
                meta = led.d.get("meta", {}).get(f"{src_name}_shard_{first_todo_si - 1:05d}", {})
            skip_until_j = int(meta.get("end_j") or 0)
            if not skip_until_j:
                skip_until_j = first_todo_si * a.shard_size
                tqdm.write(f"approx fast-forward to j>{skip_until_j} "
                           f"(hub-synced; light-skip any remainder of done 0..{first_todo_si - 1})")
            else:
                tqdm.write(f"fast-forward stream to j>{skip_until_j} "
                           f"(skip re-download of done shards 0..{first_todo_si - 1})")

        tqdm.write(f"streaming {src_name}...")
        it = iter(ds)
        try:
            first = next(it)
        except StopIteration:
            tqdm.write(f"{src_name} empty")
            return
        acol, tcol = _detect_cols(first, acol, tcol)
        tqdm.write(f"columns: audio={acol} text={tcol}")

        rows, si, j, n_kept = [], 0, 0, 0
        skip = should_skip(0)
        for ex in itertools.chain([first], it):
            j += 1
            dl_bar.update(1)

            # Cheap skip of already-finished prefix (no audio decode)
            if skip_until_j and j <= skip_until_j:
                if j == skip_until_j:
                    si = first_todo_si
                    skip = should_skip(si)
                    n_kept = 0
                    rows = []
                    dl_bar.set_postfix_str(f"resumed @ shard {si}")
                continue

            text = normalize(str(ex[tcol]))

            if skip:
                if not _keep_light(text, ex):
                    continue
                n_kept += 1
                if n_kept >= a.shard_size:
                    dl_bar.set_postfix_str(f"skip shard {si}")
                    for cand in (_sid(src_name, a.shard_size, si),
                                 f"{src_name}_shard_{si:05d}"):
                        if led.is_done(cand):
                            led.mark(cand, "done", end_j=j)
                            break
                    si += 1
                    n_kept = 0
                    skip = should_skip(si)
                continue

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
            cid = f"{src_name}_{si:04d}_{j:06d}"
            p = os.path.join(raw_root, cid + ".wav")
            save_wav(p, arr)
            rows.append({"id": cid, "audio": p, "text": text, "dur": dur,
                         "domain": a.domain, "split": _split_of(cid)})
            if len(rows) >= a.shard_size:
                emit(si, rows, j)
                rows, si = [], si + 1
                skip = should_skip(si)
        if rows and not skip:
            emit(si, rows, j)
    except Exception as e:
        err_box.append(e)
        tqdm.write(f"DOWNLOAD ERROR: {e}")
    finally:
        dl_q.put(SENTINEL)


def upload_worker(up_q: Queue, cfg, led, total_h_box: list, enc_bar: tqdm,
                  upload_every: int):
    """Batch several shards into ONE Hub commit; retry 429; never kill encode."""
    batch = []  # list of (payload_parent, sid, n_clips, hours, end_j)
    # After moving payloads into a tree, keep (tree, sids) until Hub accepts —
    # so a failed flush can be retried without dropping work.
    pending = None  # (tree_path, [(sid, n_clips, hours, end_j), ...])

    def flush():
        nonlocal batch, pending
        if pending is None:
            if not batch:
                return
            tree = tempfile.mkdtemp(prefix="hubbatch_", dir=cfg.paths.data_dir)
            sids = []
            for parent, sid, n_clips, hours, end_j in batch:
                payload = os.path.join(parent, "payload")
                dst = os.path.join(tree, "encoded", sid)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(payload, dst)
                sids.append((sid, n_clips, hours, end_j))
                shutil.rmtree(parent, ignore_errors=True)
            batch = []
            pending = (tree, sids)

        tree, sids = pending
        msg = f"encoded {sids[0][0]}..{sids[-1][0]} ({len(sids)} shards)"
        _upload_retry(tree, cfg.repos.data, "", msg)
        for sid, n_clips, hours, end_j in sids:
            total_h_box[0] += hours
            meta = {"clips": n_clips, "hours": hours}
            if end_j:
                meta["end_j"] = end_j
            led.mark(sid, "done", **meta)
        led.save()
        enc_bar.set_postfix(hub_q=up_q.qsize(), pushed_h=f"{total_h_box[0]:.1f}",
                            batch=len(sids))
        tqdm.write(f"Hub ok: {len(sids)} shards in 1 commit | "
                   f"cumulative {total_h_box[0]:.1f} h")
        shutil.rmtree(tree, ignore_errors=True)
        pending = None

    try:
        while True:
            item = up_q.get()
            if item is SENTINEL:
                while pending is not None or batch:
                    try:
                        flush()
                    except Exception as e:
                        tqdm.write(f"final flush retry in 60s: {e}")
                        time.sleep(60)
                try:
                    led.push(f"shardpipe checkpoint {total_h_box[0]:.1f}h")
                except Exception:
                    pass
                break
            parent, sid, n_clips, hours, end_j = item
            batch.append((parent, sid, n_clips, hours, end_j))
            # Retry pending first; only open a new batch commit when full.
            while pending is not None or len(batch) >= max(1, upload_every):
                try:
                    flush()
                except Exception as e:
                    tqdm.write(f"upload batch deferred (encode continues): {e}")
                    time.sleep(60)
                    break
    except Exception as e:
        tqdm.write(f"upload worker exit: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpu.yaml")
    ap.add_argument("--hf", default=None, choices=list(SOURCES))
    ap.add_argument("--hf-id", default=None)
    ap.add_argument("--hf-config", default=None)
    ap.add_argument("--split", default="train")
    ap.add_argument("--audio-col", default="audio")
    ap.add_argument("--text-col", default="sentence")
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--domain", default="general")
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--encode-batch", type=int, default=16)
    ap.add_argument("--max-batch-sec", type=float, default=48.0)
    ap.add_argument("--prefetch", type=int, default=6)
    ap.add_argument("--upload-every", type=int, default=8,
                    help="shards per Hub commit (HF limit ≈128 commits/hour)")
    ap.add_argument("--shard-start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-flush", action="store_true")
    ap.add_argument("--raw-only", action="store_true")
    ap.add_argument("--push-raw", action="store_true")
    ap.add_argument("--peek", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    if not a.hf and not a.hf_id and not a.local_dir:
        raise SystemExit("give --hf / --hf-id / --local-dir")

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
        return

    import logging
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    dev = device_auto()
    hf_login()
    ensure_repo(cfg.repos.data, "dataset")

    enc_dtype = torch.bfloat16 if (dev == "cuda" and str(getattr(cfg.base, "dtype", "")) == "bfloat16") \
        else torch.float32
    enc = None if a.raw_only else build_encoder(cfg, dtype=enc_dtype).to(dev).eval()
    n_codes = 0 if a.raw_only else int(getattr(cfg.audio, "n_codes", 0))
    qpath = os.path.join(cfg.paths.data_dir, "encoded", "quantizer.pt")
    os.makedirs(os.path.dirname(qpath), exist_ok=True)
    quant_box = [KMeansQuantizer.load(qpath) if (n_codes > 0 and os.path.isfile(qpath)) else None]

    led = ShardLedger(os.path.join(cfg.paths.ledger_dir, "shardpipe.json"), "shardpipe",
                      repo_id=cfg.repos.data)
    src_name = (a.hf or a.hf_id or a.local_dir).replace("/", "_")
    first_todo = sync_done_from_hub(led, cfg.repos.data, src_name, a.shard_size)
    total_h_box = [led.total_meta("hours")]
    total = _estimate_total(a)

    dl_q: Queue = Queue(maxsize=max(1, a.prefetch))
    up_q: Queue = Queue(maxsize=max(4, a.prefetch * 2))
    err_box: list = []

    dl_bar = tqdm(total=total, unit="ex", desc="download", position=0,
                  dynamic_ncols=True, file=sys.stderr, leave=True, smoothing=0.05)
    enc_bar = tqdm(total=None, unit="clip", desc="encode  ", position=1,
                   dynamic_ncols=True, file=sys.stderr, leave=True, smoothing=0.05)

    t_dl = threading.Thread(
        target=download_worker,
        args=(a, cfg, src_name, led, dl_q, dl_bar, err_box, first_todo),
        name="download", daemon=True,
    )
    t_up = threading.Thread(
        target=upload_worker,
        args=(up_q, cfg, led, total_h_box, enc_bar, a.upload_every),
        name="upload", daemon=True,
    )
    t_dl.start()
    t_up.start()

    max_batch_samples = int(float(a.max_batch_sec) * SAMPLE_RATE)
    max_clips = max(1, int(a.encode_batch))

    @torch.no_grad()
    def _forward_once(wavs):
        lens = [len(w) for w in wavs]
        maxlen = max(lens)
        batch = np.zeros((len(wavs), maxlen), dtype=np.float32)
        for j, w in enumerate(wavs):
            batch[j, : lens[j]] = w
        wave = torch.from_numpy(batch).to(dev, non_blocking=True)
        wave_len = torch.tensor(lens, device=dev)
        f, fl = enc.features(wave, wave_len)
        out = [f[j, : int(fl[j])].float().cpu().numpy() for j in range(len(wavs))]
        del wave, wave_len, f, fl, batch
        return out

    def _forward_safe(wavs):
        try:
            return _forward_once(wavs)
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if len(wavs) == 1:
                raise
            mid = max(1, len(wavs) // 2)
            tqdm.write(f"OOM at batch={len(wavs)} — split → {mid}+{len(wavs) - mid}")
            return _forward_safe(wavs[:mid]) + _forward_safe(wavs[mid:])

    @torch.no_grad()
    def feats_batch(rows):
        items = [(r["audio"], float(r["dur"])) for r in rows]
        items.sort(key=lambda x: x[1])
        out_by_path = {}
        i = 0
        while i < len(items):
            chunk_paths, chunk_wavs = [], []
            max_len = 0
            while i < len(items) and len(chunk_paths) < max_clips:
                path, dur = items[i]
                L = max(1, int(dur * SAMPLE_RATE))
                new_max = max(max_len, L)
                if chunk_paths and (len(chunk_paths) + 1) * new_max > max_batch_samples:
                    break
                w = load_wav(path)
                chunk_paths.append(path)
                chunk_wavs.append(w)
                max_len = max(max_len, len(w))
                i += 1
            if not chunk_wavs:
                path, _ = items[i]
                chunk_paths, chunk_wavs = [path], [load_wav(path)]
                i += 1
            for p, f in zip(chunk_paths, _forward_safe(chunk_wavs)):
                out_by_path[p] = f
            del chunk_wavs
        return [out_by_path[r["audio"]] for r in rows]

    try:
        while True:
            item = dl_q.get()
            if item is SENTINEL:
                break
            if err_box:
                raise err_box[0]
            si, sid, rows, hours, end_j = item
            if not rows:
                continue
            # Belt-and-suspenders: never re-encode a done shard
            if led.is_done(sid):
                for r in rows:
                    p = r.get("audio")
                    if p and os.path.isfile(p):
                        os.remove(p)
                continue

            work = os.path.join(cfg.paths.data_dir, "encoded", "shards", sid)
            hub = os.path.join(work, "_hub")
            os.makedirs(hub, exist_ok=True)
            try:
                if not a.raw_only:
                    feats = feats_batch(rows)
                    pack = {r["id"]: f.astype(np.float16) for r, f in zip(rows, feats)}
                    np.savez(os.path.join(hub, "feats.npz"), **pack)
                    for r in rows:
                        r["feats"] = f"encoded/{sid}/feats.npz"
                        r["feats_key"] = r["id"]
                    if n_codes > 0 and quant_box[0] is None:
                        pool = np.concatenate([f.astype(np.float32) for f in feats[:64]], 0)
                        quant_box[0] = KMeansQuantizer.fit(pool, n_codes, iters=25)
                        quant_box[0].save(qpath)
                        upload_file(qpath, cfg.repos.data, "dataset",
                                    "encoded/quantizer.pt", "quantizer")
                    if n_codes > 0:
                        qz = quant_box[0]
                        cpack = {r["id"]: qz.encode(f.astype(np.float32)).astype(np.int64)
                                 for r, f in zip(rows, feats)}
                        np.savez(os.path.join(hub, "codes.npz"), **cpack)
                        for r in rows:
                            r["codes"] = f"encoded/{sid}/codes.npz"
                            r["codes_key"] = r["id"]
                    del feats, pack
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                write_manifest(os.path.join(hub, "manifest.jsonl"), rows)

                stage = tempfile.mkdtemp(prefix="hubup_", dir=cfg.paths.data_dir)
                payload = os.path.join(stage, "payload")
                shutil.copytree(hub, payload)
                up_q.put((stage, sid, len(rows), hours, end_j))

                enc_bar.update(len(rows))
                enc_bar.set_postfix(q_dl=dl_q.qsize(), hub_q=up_q.qsize(),
                                    shard=si, h=f"{hours:.1f}")
            finally:
                if not a.no_flush:
                    shutil.rmtree(work, ignore_errors=True)
                    for r in rows:
                        p = r.get("audio")
                        if p and os.path.isfile(p):
                            os.remove(p)
    finally:
        up_q.put(SENTINEL)
        t_dl.join(timeout=5)
        t_up.join(timeout=7200)
        dl_bar.close()
        enc_bar.close()

    if err_box:
        raise err_box[0]
    log.info("DONE. total pushed: %.1f h | ledger: %s", total_h_box[0], led.counts())


if __name__ == "__main__":
    main()
