"""End-to-end smoke test — runs the WHOLE pipeline on synthetic data in seconds on
CPU/MPS, so the H100 run hits zero wiring surprises. Exercises: synthetic manifest ->
feature dump -> k-means quantizer -> discrete codes -> train (phase 1 CTC + phase 3
joint) -> resume-from-checkpoint -> eval (WER/CER/CTC/FC) -> AR inference -> streaming.

Nothing here needs the network or any token. It uses TinyEncoder + TinyNandi (a faithful
factorized/tied/layer-sharing mirror of Nandi), so the risky token-extension path is the
same code that runs on the H100 with backend: real.
"""
from __future__ import annotations

import os

import numpy as np

from .audio import save_wav, synth_speechish
from .dataset import read_manifest, write_manifest
from .env import log
from .model import KupeFDXModel
from .quantizer import KMeansQuantizer
from .stream import StreamingSession

HINDI = ["नमस्ते आप कैसे हैं", "मुझे दवा चाहिए", "यह तकनीकी समस्या है",
         "धन्यवाद आपका दिन शुभ हो", "कृपया थोड़ी मदद करें", "मेरा नाम राहुल है"]


def make_synthetic(root: str, n_train=12, n_val=4, n_test=4, seconds=1.2) -> str:
    wav_dir = os.path.join(root, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    rows, idx = [], 0
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        for _ in range(n):
            text = HINDI[idx % len(HINDI)]
            wav = synth_speechish(seconds, seed=idx)
            path = os.path.join(wav_dir, f"{split}_{idx:04d}.wav")
            save_wav(path, wav)
            fc = ["<EOS_SPEECH>"] if idx % 3 == 0 else ["<NOP>"]
            rows.append({"id": f"{split}_{idx:04d}", "audio": path, "text": text,
                         "dur": seconds, "domain": "general", "split": split, "fc": fc})
            idx += 1
    mpath = os.path.join(root, "manifest.jsonl")
    write_manifest(mpath, rows)
    log.info("synthetic manifest: %d clips -> %s", len(rows), mpath)
    return mpath


def encode_codes(cfg, manifest: str) -> None:
    """Dump TinyEncoder feats, fit a small k-means quantizer, write per-clip code ids,
    and add `codes` paths to the manifest — exercising the discrete audio-token branch."""
    if int(getattr(cfg.audio, "n_codes", 0)) <= 0:
        return
    import torch

    from .encoders import build_encoder
    from .audio import load_wav
    enc = build_encoder(cfg).eval()
    rows = read_manifest(manifest)
    root = os.path.dirname(manifest)
    code_dir = os.path.join(root, "codes")
    os.makedirs(code_dir, exist_ok=True)

    feats_all, per_clip = [], []
    with torch.no_grad():
        for r in rows:
            w = load_wav(r["audio"])
            wt = torch.from_numpy(w)[None]
            f, fl = enc.features(wt, torch.tensor([len(w)]))
            f = f[0, : int(fl[0])].float().numpy()
            per_clip.append((r, f))
            feats_all.append(f)
    pool = np.concatenate(feats_all, 0)
    q = KMeansQuantizer.fit(pool, int(cfg.audio.n_codes), iters=10,
                            log_fn=lambda i, s: log.info("kmeans it %d shift=%.4f", i, s))
    q.save(os.path.join(root, "quantizer.pt"))
    for r, f in per_clip:
        codes = q.encode(f).astype(np.int64)
        cp = os.path.join(code_dir, r["id"] + ".npy")
        np.save(cp, codes)
        r["codes"] = cp
    write_manifest(manifest, rows)
    log.info("wrote discrete codes for %d clips (n_codes=%d)", len(rows), cfg.audio.n_codes)


def _gen_fc(cfg, manifest: str) -> str:
    """Mock floor-control generation from the synthetic ASR clips -> fc manifest, and
    print one sample row so the format is visible."""
    import json

    from .fcgen.agent import generate
    from .fcgen.audio_probe import audio_card, probe
    from .fcgen.scenarios import rebalance
    rows = read_manifest(manifest)
    clips = [{"id": r["id"], "audio": r["audio"], "transcript": r["text"],
              "domain": r["domain"], "features": (f := probe(r["audio"])),
              "card": audio_card(f, r["text"])} for r in rows]
    fc = rebalance(generate(clips, rows_per_hit=12, clips_per_hit=4, concurrency=4, mock=True))
    for i, r in enumerate(fc):
        r["split"] = "train" if i < int(0.9 * len(fc)) else "val"
    out = os.path.join(os.path.dirname(manifest), "fc.jsonl")
    write_manifest(out, fc)
    log.info("generated %d FC rows -> %s", len(fc), out)
    if fc:
        log.info("sample FC row:\n%s", json.dumps(fc[0], ensure_ascii=False, indent=2))
    return out


def run_smoke(cfg) -> bool:
    from .constants import PHASE_CTC, PHASE_JOINT
    from .train import train

    root = os.path.join(cfg.paths.data_dir, "smoke")
    manifest = make_synthetic(root)
    cfg.data.manifest = manifest
    encode_codes(cfg, manifest)

    log.info("===== PHASE 1 (CTC encoder warmup) =====")
    train(cfg, PHASE_CTC)

    log.info("===== PHASE 3 (joint) — run A =====")
    run_a = train(cfg, PHASE_JOINT)
    run_name = os.path.basename(run_a)
    log.info("===== PHASE 3 — run B (resume auto, proves checkpoint restore) =====")
    train(cfg, PHASE_JOINT, resume=run_name)

    log.info("===== FLOOR-CONTROL DATA GENERATION (mock agent) + PHASE 4 =====")
    fc_manifest = _gen_fc(cfg, manifest)
    from .constants import PHASE_FC
    cfg.data.manifest = fc_manifest
    train(cfg, PHASE_FC)
    cfg.data.manifest = manifest

    log.info("===== INFERENCE + STREAMING on a fresh model =====")
    import torch
    model = KupeFDXModel.build(cfg).to("cpu").eval()
    rows = read_manifest(manifest)
    from .audio import load_wav
    w = load_wav(rows[0]["audio"])
    wt = torch.from_numpy(w)[None]
    gen = model.generate(wave=wt, wave_len=torch.tensor([len(w)]), max_new_tokens=16)
    log.info("AR transcript (random-init tiny, content irrelevant): %r", model.transcribe(gen)[0])
    sess = StreamingSession(model, chunk_ms=480, correct_every=4)
    records = sess.run_file(w)
    fired = sum(1 for r in records if r["backchannel"] or r["thinking_sound"]
                or r["eos_flag"] or r["silence_flag"])
    log.info("streamed %d chunks | %d emitted a signal, %d stayed quiet (NOTHING)",
             len(records), fired, len(records) - fired)
    import json
    log.info("sample per-chunk record (matches the output spec):\n%s",
             json.dumps(records[0], ensure_ascii=False, indent=2))

    log.info("✅ SMOKE PASS — full pipeline ran end to end (prep→encode→quantize→train→"
             "resume→eval→infer→stream). Flip backend: real on the H100.")
    return True
