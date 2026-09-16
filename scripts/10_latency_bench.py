#!/usr/bin/env python3
"""Stage 10 — measure REAL per-chunk streaming latency, so the <100 ms claim is backed by
numbers (not hope). Reports p50/p95 for: encoder on one chunk, floor-control head, and one
Nandi decode step; then the end-of-turn budget = chunk_ms + compute.

    python scripts/10_latency_bench.py --config configs/smoke.yaml            # tiny model (shape)
    python scripts/10_latency_bench.py --config configs/gpu.yaml --chunk-ms 80 # real, on H100
"""
import argparse
import statistics
import time

import numpy as np
import torch

import _bootstrap  # noqa: F401
from kupefdx.audio import synth_speechish
from kupefdx.config import load_config
from kupefdx.env import device_auto, log
from kupefdx.model import KupeFDXModel


def _timed(fn, n, warmup=3, cuda=False):
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return statistics.median(ts), (sorted(ts)[int(0.95 * len(ts)) - 1] if len(ts) > 1 else ts[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--chunk-ms", type=int, default=None)
    ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args()
    cfg = load_config(a.config)
    dev = device_auto()
    cuda = dev == "cuda"
    chunk_ms = a.chunk_ms or int(getattr(getattr(cfg, "stream", object()), "chunk_ms", 80))

    model = KupeFDXModel.build(cfg).to(dev).eval()
    from kupefdx.constants import SAMPLE_RATE
    # a rolling buffer ~2 s (realistic mid-utterance) and one new chunk on top.
    buf = synth_speechish(2.0, seed=1)
    wave = torch.from_numpy(buf)[None].to(dev)
    wl = torch.tensor([len(buf)], device=dev)

    with torch.no_grad():
        enc_p50, enc_p95 = _timed(lambda: model._encode(wave, wl, None, None), a.n, cuda=cuda)
        feats, flen = model._encode(wave, wl, None, None)
        fc_p50, fc_p95 = _timed(lambda: model.fc_head(feats.float(), flen, model.chunk_frames),
                                a.n, cuda=cuda)
        # one Nandi decode step (KV-cache streaming would amortize this per token)
        pre, _ = model._audio_prefix_embeds(feats, flen)
        H = pre[0][None].to(model._dtype)
        attn = torch.ones(1, H.shape[1], dtype=torch.long, device=dev)
        dec_p50, dec_p95 = _timed(lambda: model.lm(inputs_embeds=H, attention_mask=attn), a.n, cuda=cuda)

    log.info("=== latency on %s | chunk_ms=%d | %s ===", dev, chunk_ms, cfg.backend)
    log.info("encoder / chunk : p50=%.1f ms  p95=%.1f ms", enc_p50, enc_p95)
    log.info("floor-control   : p50=%.2f ms  p95=%.2f ms", fc_p50, fc_p95)
    log.info("one decode step : p50=%.1f ms  p95=%.1f ms", dec_p50, dec_p95)
    compute = enc_p50 + fc_p50 + dec_p50
    log.info("per-chunk compute (encode+FC+1 decode step) ~ %.1f ms", compute)
    log.info("END-OF-TURN (predictive endpoint) ~ chunk_ms(%d) + compute(%.0f) = %.0f ms",
             chunk_ms, compute, chunk_ms + compute)
    note = "note: TINY model — real H100 numbers come from configs/gpu.yaml with real weights." \
        if cfg.backend != "real" else \
        "note: for the real transcript, KV-cache streaming decode amortizes to ~1 step/token."
    log.info(note)


if __name__ == "__main__":
    main()
