#!/usr/bin/env python3
"""Stage 2 — cache encoder features, fit the k-means quantizer, dump discrete codes.

Resumable per clip via a ShardLedger. Feature caching (frozen encoder) is a big
Phase-2 speedup; discrete codes feed the audio-token branch. Runs on any single GPU
(H100/L4/4090) or CPU.

    python scripts/02_encode.py --config configs/gpu.yaml --feats --fit-quantizer --codes
"""
import argparse
import os

import numpy as np
import torch

import _bootstrap  # noqa: F401
from kupefdx.audio import load_wav
from kupefdx.config import load_config
from kupefdx.dataset import read_manifest, write_manifest
from kupefdx.encoders import build_encoder
from kupefdx.env import device_auto, log
from kupefdx.ledger import ShardLedger
from kupefdx.quantizer import KMeansQuantizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpu.yaml")
    ap.add_argument("--feats", action="store_true", help="dump cached encoder features")
    ap.add_argument("--fit-quantizer", action="store_true")
    ap.add_argument("--codes", action="store_true", help="dump discrete audio-code ids")
    ap.add_argument("--sample-frames", type=int, default=200000, help="frames for k-means fit")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, overrides=a.set)
    dev = device_auto()

    rows = read_manifest(cfg.data.manifest)
    enc = build_encoder(cfg).to(dev).eval()
    feat_dir = cfg.paths.feats_dir
    code_dir = os.path.join(cfg.paths.data_dir, "encoded", "codes")
    os.makedirs(feat_dir, exist_ok=True)
    os.makedirs(code_dir, exist_ok=True)
    led = ShardLedger(os.path.join(cfg.paths.ledger_dir, "encode.json"), "encode")

    @torch.no_grad()
    def feats_of(r):
        w = load_wav(r["audio"])
        f, fl = enc.features(torch.from_numpy(w)[None].to(dev), torch.tensor([len(w)], device=dev))
        return f[0, : int(fl[0])].float().cpu().numpy()

    # ---- feature dump ----
    if a.feats:
        for r in rows:
            if led.state(r["id"]) == "done" and r.get("feats"):
                continue
            fp = os.path.join(feat_dir, r["id"] + ".npy")
            np.save(fp, feats_of(r))
            r["feats"] = fp
            led.mark(r["id"], "done", frames=int(np.load(fp).shape[0]))
        write_manifest(cfg.data.manifest, rows)
        log.info("feature dump done: %s", led.counts())

    # ---- quantizer fit ----
    qpath = os.path.join(cfg.paths.data_dir, "encoded", "quantizer.pt")
    if a.fit_quantizer:
        pool, budget = [], a.sample_frames
        for r in rows:
            f = np.load(r["feats"]) if r.get("feats") else feats_of(r)
            pool.append(f)
            if sum(len(x) for x in pool) >= budget:
                break
        pool = np.concatenate(pool, 0)[:budget]
        q = KMeansQuantizer.fit(pool, int(cfg.audio.n_codes), iters=25,
                                log_fn=lambda i, s: log.info("kmeans it %d shift=%.4f", i, s))
        q.save(qpath)
        log.info("quantizer fit (%d codes) -> %s", cfg.audio.n_codes, qpath)

    # ---- discrete codes ----
    if a.codes:
        q = KMeansQuantizer.load(qpath)
        for r in rows:
            f = np.load(r["feats"]) if r.get("feats") else feats_of(r)
            cp = os.path.join(code_dir, r["id"] + ".npy")
            np.save(cp, q.encode(f).astype(np.int64))
            r["codes"] = cp
        write_manifest(cfg.data.manifest, rows)
        log.info("codes dumped for %d clips", len(rows))


if __name__ == "__main__":
    main()
