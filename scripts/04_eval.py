#!/usr/bin/env python3
"""Stage 4 — evaluate a saved run on val/test: WER/CER, CTC-greedy WER, floor-control.

    python scripts/04_eval.py --config configs/gpu.yaml --ckpt checkpoints/<run>/checkpoint-XXXX --split test
"""
import argparse
import os

import torch

import _bootstrap  # noqa: F401
from kupefdx.collate import Collator
from kupefdx.config import load_config
from kupefdx.dataset import read_manifest
from kupefdx.env import device_auto, log
from kupefdx.evaluate import run_eval, save_report
from kupefdx.model import KupeFDXModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpu.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test", "train"])
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, overrides=a.set)
    dev = device_auto()

    model = KupeFDXModel.build(cfg).to(dev)
    state = torch.load(os.path.join(a.ckpt, "state.pt"), map_location=dev)
    model.load_state_dict(state["model"])
    model.eval()

    rows = [r for r in read_manifest(cfg.data.manifest) if r.get("split") == a.split]
    mode = "feats" if cfg.data.use_cached_feats and rows and rows[0].get("feats") else "raw"
    coll = Collator(model.tok, model.char_tok, bos_id=model.bos_id, eos_id=model.eos_id,
                    pad_id=model.pad_id, special_ids=model.special_ids,
                    max_audio_frames=int(cfg.model.max_audio_frames),
                    max_text_tokens=int(cfg.model.max_text_tokens),
                    n_codes=model.n_codes, mode=mode)
    rep = run_eval(model, rows, coll, dev, max_new_tokens=int(cfg.eval.max_new_tokens),
                   batch_size=int(cfg.eval.batch_size), mode=mode)
    log.info("%s | WER=%.4f CER=%.4f ctcWER=%.4f (n=%d)", a.split, rep["wer"],
             rep["cer"], rep.get("ctc_wer", float("nan")), rep["n"])
    if "fc" in rep:
        log.info("floor-control: %s", rep["fc"])
    out = os.path.join(a.ckpt, f"eval_{a.split}.json")
    save_report(rep, out, title=os.path.basename(a.ckpt))
    log.info("report -> %s", out)


if __name__ == "__main__":
    main()
