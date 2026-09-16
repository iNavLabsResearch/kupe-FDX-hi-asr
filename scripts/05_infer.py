#!/usr/bin/env python3
"""Stage 5 — inference on a wav file: offline AR transcript, CTC fast-path, and a
streaming pass (partial transcripts + floor-control signals).

    python scripts/05_infer.py --config configs/gpu.yaml --ckpt <ckpt> --wav clip.wav
    python scripts/05_infer.py --config configs/gpu.yaml --ckpt <ckpt> --wav clip.wav --stream
"""
import argparse
import os

import torch

import _bootstrap  # noqa: F401
from kupefdx.audio import load_wav
from kupefdx.config import load_config
from kupefdx.env import device_auto, log
from kupefdx.floorcontrol import StreamControls
from kupefdx.model import KupeFDXModel
from kupefdx.stream import StreamingSession


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpu.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wav", required=True)
    ap.add_argument("--stream", action="store_true")
    # inference-time floor-control controls (the "temperature" analog)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--bc-bias", type=float, default=None, help="backchannel eagerness (+ = more)")
    ap.add_argument("--think-bias", type=float, default=None)
    ap.add_argument("--no-think", action="store_true", help="disable thinking-sound at inference")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, overrides=a.set)
    dev = device_auto()

    model = KupeFDXModel.build(cfg).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.ckpt, "state.pt"), map_location=dev)["model"])
    model.eval()

    w = load_wav(a.wav)
    wt = torch.from_numpy(w)[None].to(dev)
    wl = torch.tensor([len(w)], device=dev)
    print("CTC fast-path :", model.ctc_transcribe(wave=wt, wave_len=wl)[0])
    gen = model.generate(wave=wt, wave_len=wl, max_new_tokens=int(cfg.eval.max_new_tokens))
    print("AR transcript :", model.transcribe(gen)[0])

    if a.stream:
        import json
        controls = StreamControls.from_config(cfg)
        if a.temperature is not None:
            controls.temperature = a.temperature
        if a.bc_bias is not None:
            controls.biases["<BC>"] = a.bc_bias
        if a.think_bias is not None:
            controls.biases["<THINK>"] = a.think_bias
        if a.no_think:
            controls.enabled["<THINK>"] = False
        sess = StreamingSession(model, controls=controls)
        for rec in sess.run_file(w):
            sig = rec["backchannel"] or rec["thinking_sound"] or \
                ("EOS" if rec["eos_flag"] else "") or ("SIL" if rec["silence_flag"] else "") or "-"
            print(f"  chunk {rec['chunk_id']:3d} @ {rec['timestamp_ms']:5d}ms | "
                  f"signal={sig:6s} | raw={rec['raw_ctc_transcript']!r}")
        print("last record:", json.dumps(sess.run_file(w)[-1], ensure_ascii=False))


if __name__ == "__main__":
    main()
