"""Training — one plain PyTorch loop, phase-aware, fully resumable.

Resume: a checkpoint bundles model parts + optimizer + scheduler + global step + RNG +
wandb run id, so `--resume auto` continues bit-for-bit on the same or another box (after
`hf_pull`). Phases set which sub-modules train and the loss weights (see phase_setup).

Single H100 target (no DDP needed); the loop is torchrun-safe for later scale-out.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .collate import Collator
from .constants import (PHASE_ALIGN, PHASE_CTC, PHASE_DOMAIN, PHASE_FC, PHASE_JOINT,
                        SPLIT_TRAIN, SPLIT_VAL)
from .dataset import ManifestDataset, read_manifest
from .env import device_auto, init_wandb, log
from .evaluate import run_eval
from .ledger import Ledger, iso_now
from .model import KupeFDXModel


def phase_setup(phase: int, cfg) -> dict:
    """Return loss weights + which modules train + encoder LR scale for a phase."""
    t = cfg.train
    # fc = inline-flag CE weight (LM); fcc = per-chunk head weight; fch = fc_head trains.
    # RECIPE for a PRETRAINED ASR encoder (FastConformer): keep the encoder FROZEN and warm up
    # the projector first, then add Nandi, then unfreeze the encoder at a tiny LR at the very
    # end. This prevents random projector gradients from wrecking the good pretrained encoder.
    # lr_scale multiplies cfg.train.lr for the phase; enc_lr scales the encoder param-group LR.
    table = {
        # Stage A/warmup: projector-only (encoder+Nandi FROZEN). CTC head (linear on frozen
        # feats) also trains — cheap, gives blank-run timing. Projector LR ~1e-3.
        PHASE_CTC:    dict(lm=1.0, ctc=0.3, fc=0.0, fcc=0.0, fch=False, lr_scale=10.0,
                           enc=False, c=True,  proj=True,  dec=False, enc_lr=0.0),
        # Stage B align: encoder FROZEN; projector + Nandi + CTC + FC head. LR ~1e-4.
        PHASE_ALIGN:  dict(lm=1.0, ctc=0.3, fc=0.0, fcc=0.2, fch=True, lr_scale=1.0,
                           enc=False, c=True,  proj=True,  dec=True,  enc_lr=0.0),
        # Stage B joint / full fine-tune: unfreeze encoder at a TINY LR. Nandi ~2e-5, enc ~1e-5.
        PHASE_JOINT:  dict(lm=1.0, ctc=0.3, fc=0.0, fcc=0.2, fch=True, lr_scale=0.2,
                           enc=True,  c=True,  proj=True,  dec=True,  enc_lr=0.5),
        # Stage C floor-control: encoder frozen again; Nandi+projector+FC at low LR.
        PHASE_FC:     dict(lm=1.0, ctc=0.1, fc=float(getattr(t, "fc_weight", 3.0)),
                           fcc=float(getattr(t, "fc_chunk_weight", 1.0)), fch=True, lr_scale=0.3,
                           enc=False, c=False, proj=True, dec=True, enc_lr=0.0),
        PHASE_DOMAIN: dict(lm=1.0, ctc=0.1, fc=1.0, fcc=0.0, fch=False, lr_scale=0.3,
                           enc=False, c=False, proj=False, dec=True, enc_lr=0.0),
    }
    return table[int(phase)]


def _seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def _loaders(cfg, model, mode):
    rows = read_manifest(cfg.data.manifest)
    max_dur = float(getattr(cfg.data, "max_dur", 30.0))
    tr = ManifestDataset(rows, mode, max_dur, SPLIT_TRAIN)
    va = ManifestDataset(rows, mode, max_dur, SPLIT_VAL)
    coll = Collator(model.tok, model.char_tok, bos_id=model.bos_id, eos_id=model.eos_id,
                    pad_id=model.pad_id, special_ids=model.special_ids,
                    max_audio_frames=int(cfg.model.max_audio_frames),
                    max_text_tokens=int(cfg.model.max_text_tokens),
                    n_codes=model.n_codes, mode=mode,
                    frame_rate=float(model.encoder.frame_rate),
                    chunk_frames=int(model.chunk_frames),
                    eos_lead_ms=int(getattr(cfg.train, "eos_lead_ms", 160)))
    nw = int(getattr(cfg.train, "num_workers", 0))
    train_dl = DataLoader(tr, batch_size=int(cfg.train.batch_size), shuffle=True,
                          collate_fn=coll, num_workers=nw, drop_last=False)
    return train_dl, va, coll


def _optim(model, cfg, ps):
    lr = float(cfg.train.lr) * float(ps.get("lr_scale", 1.0))   # per-phase LR (research recipe)
    enc_params = [p for p in model.encoder.parameters() if p.requires_grad]
    other = [p for n, p in model.named_parameters()
             if p.requires_grad and not n.startswith("encoder.")]
    groups = [{"params": other, "lr": lr}]
    if enc_params and ps["enc_lr"] > 0:
        groups.append({"params": enc_params, "lr": lr * ps["enc_lr"]})
    return torch.optim.AdamW(groups, lr=lr, weight_decay=float(cfg.train.weight_decay))


def _sched(opt, cfg, total_steps):
    warm = int(float(getattr(cfg.train, "warmup_ratio", 0.03)) * total_steps)

    def fn(step):
        if step < warm:
            return (step + 1) / max(warm, 1)
        prog = (step - warm) / max(total_steps - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * min(prog, 1.0)))
    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def _ckpt_dir(out_dir, step):
    return os.path.join(out_dir, f"checkpoint-{step}")


def _save_ckpt(path, model, opt, sched, step, cfg, wandb_id):
    os.makedirs(path, exist_ok=True)
    torch.save({
        "model": model.state_dict(), "opt": opt.state_dict(),
        "sched": sched.state_dict(), "step": step,
        "rng": {"torch": torch.get_rng_state(), "np": np.random.get_state(),
                "py": random.getstate()},
        "wandb_id": wandb_id,
    }, os.path.join(path, "state.pt"))
    model.save(path, cfg)


def _push_ckpt(cfg, ckpt_path, run_name, blocking=False):
    """Mirror a just-saved checkpoint to the runs repo so progress is never lost. Periodic
    pushes run in a background daemon thread (never block or crash training); the final push
    is blocking so it completes before the process exits. No-op unless cfg.train.push_to_hub
    is set and a runs repo is configured."""
    if not bool(getattr(cfg.train, "push_to_hub", False)) or not getattr(cfg.repos, "runs", None):
        return

    def _do():
        try:
            from .env import ensure_repo, hf_login, upload_folder
            hf_login(); ensure_repo(cfg.repos.runs, "model")
            dest = f"runs/{run_name}/{os.path.basename(ckpt_path)}"
            upload_folder(ckpt_path, cfg.repos.runs, "model", path_in_repo=dest,
                          commit_message=f"ckpt {os.path.basename(ckpt_path)}")
            log.info("↑ checkpoint synced -> %s:%s", cfg.repos.runs, dest)
        except Exception as e:
            log.warning("checkpoint Hub sync failed (checkpoint safe locally): %s", e)

    if blocking:
        _do()
    else:
        import threading
        threading.Thread(target=_do, daemon=True).start()


def _latest_ckpt(out_dir):
    cks = glob.glob(os.path.join(out_dir, "checkpoint-*"))
    cks = [c for c in cks if os.path.exists(os.path.join(c, "state.pt"))]
    if not cks:
        return None
    return max(cks, key=lambda c: int(c.rsplit("-", 1)[-1]))


def train(cfg, phase: int, resume: str | None = None, init_from: str | None = None) -> str:
    _seed(int(cfg.seed))
    dev = device_auto()
    ps = phase_setup(phase, cfg)
    mode = "feats" if int(phase) == PHASE_ALIGN and bool(getattr(cfg.data, "use_cached_feats", False)) else "raw"

    run_name = resume if (resume and resume != "auto") else \
        f"{cfg.project}-p{phase}-{time.strftime('%Y%m%d-%H%M%S')}"
    out_dir = os.path.join(cfg.paths.runs_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    model = KupeFDXModel.build(cfg).to(dev)
    # phase chaining: warm-start weights ONLY from a prior phase's run (fresh optimizer/step),
    # so e.g. phase 4/5 inherit phase 2's aligned projector+Nandi. --resume is for continuing
    # the SAME run; --init-from is for starting a NEW phase from a finished one.
    if init_from and not resume:
        src = init_from if (os.path.isdir(init_from) and os.path.exists(os.path.join(init_from, "state.pt"))) \
            else _latest_ckpt(os.path.join(cfg.paths.runs_dir, init_from)) or _latest_ckpt(init_from)
        if not src:
            raise SystemExit(f"--init-from: no checkpoint found for '{init_from}'")
        sd = torch.load(os.path.join(src, "state.pt"), map_location=dev, weights_only=False)
        missing, unexpected = model.load_state_dict(sd["model"], strict=False)
        log.info("init-from %s: loaded weights (fresh optimizer) | missing=%d unexpected=%d",
                 src, len(missing), len(unexpected))
    model.stage(encoder=ps["enc"], ctc=ps["c"], projector=ps["proj"], decoder=ps["dec"],
                fc=ps["fch"])
    from .constants import STAGE_NAME, STAGE_OF_PHASE
    st = STAGE_OF_PHASE.get(int(phase), "?")
    log.info("phase %d (STAGE %s: %s) on %s | trainable params: %.2fM | transcript producer=Nandi",
             phase, st, STAGE_NAME.get(st, ""), dev, model.trainable_params() / 1e6)

    train_dl, va_rows, coll = _loaders(cfg, model, mode)
    epochs = int(cfg.train.epochs)
    accum = int(getattr(cfg.train, "grad_accum", 1))
    total_steps = max(1, (len(train_dl) * epochs) // accum)
    # LLaVA-style projector warmup: train the projector alone for the first N steps, then
    # unfreeze Nandi. Lets ALIGN be ONE run (merges the old warmup+align phases).
    warmup_proj = int(getattr(cfg.train, "warmup_proj_steps", 0))
    if warmup_proj > 0 and ps["dec"]:
        model._set(model.lm, False)                 # freeze Nandi during the warmup window
        log.info("projector warmup: Nandi frozen for first %d steps", warmup_proj)
    opt = _optim(model, cfg, ps)
    sched = _sched(opt, cfg, total_steps)

    wb = init_wandb(cfg.project, run_name, cfg.to_dict())
    wandb_id = getattr(wb, "id", None) if wb else None

    start_step = 0
    resume_dir = None
    if resume:
        resume_dir = _latest_ckpt(out_dir) if resume == "auto" else \
            (_ckpt_dir(out_dir, resume) if str(resume).isdigit() else _latest_ckpt(os.path.join(cfg.paths.runs_dir, resume)))
    if resume_dir and os.path.exists(os.path.join(resume_dir, "state.pt")):
        # our own trusted checkpoint; contains numpy RNG state -> weights_only=False.
        sd = torch.load(os.path.join(resume_dir, "state.pt"), map_location=dev,
                        weights_only=False)
        model.load_state_dict(sd["model"]); opt.load_state_dict(sd["opt"])
        sched.load_state_dict(sd["sched"]); start_step = int(sd["step"])
        try:
            torch.set_rng_state(sd["rng"]["torch"].to("cpu", torch.uint8))
            np.random.set_state(sd["rng"]["np"])
            random.setstate(sd["rng"]["py"])
        except Exception as e:
            log.warning("rng restore skipped: %s", e)
        log.info("resumed %s at step %d", resume_dir, start_step)

    use_bf16 = dev == "cuda" and bool(getattr(cfg.train, "bf16", True))
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else _null_ctx()

    step = start_step
    eval_steps = int(cfg.train.eval_steps)
    save_steps = int(cfg.train.save_steps)
    log_steps = int(getattr(cfg.train, "logging_steps", 10))
    model.train()
    log.info("=== training %s | %d optim steps ===", run_name, total_steps)
    t0 = time.time()
    opt.zero_grad()
    micro = 0
    for ep in range(epochs):
        for batch in train_dl:
            with autocast:
                out = model(**batch, lm_weight=ps["lm"], ctc_weight=ps["ctc"],
                            fc_weight=ps["fc"], fc_chunk_weight=ps["fcc"])
                loss = out["loss"] / accum
            loss.backward()
            micro += 1
            if micro % accum != 0:
                continue
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(cfg.train.max_grad_norm))
            opt.step(); sched.step(); opt.zero_grad()
            step += 1

            if warmup_proj and step == warmup_proj and ps["dec"]:
                model._set(model.lm, True)          # end of warmup: unfreeze Nandi
                log.info("projector warmup done at step %d — Nandi unfrozen", step)

            if step % log_steps == 0:
                parts = " ".join(f"{k}={float(v):.3f}" for k, v in out.items() if k != "loss")
                log.info("step %d/%d | loss=%.4f %s | lr=%.2e | %.1fs",
                         step, total_steps, float(out["loss"]), parts,
                         sched.get_last_lr()[0], time.time() - t0)
                if wb:
                    wb.log({"train/loss": float(out["loss"]),
                            **{f"train/{k}": float(v) for k, v in out.items() if k != "loss"},
                            "lr": sched.get_last_lr()[0]}, step=step)
            if eval_steps and step % eval_steps == 0:
                _do_eval(model, va_rows, coll, cfg, step, wb, mode)
                model.train()
            if save_steps and step % save_steps == 0:
                ck = _ckpt_dir(out_dir, step)
                _save_ckpt(ck, model, opt, sched, step, cfg, wandb_id)
                log.info("saved checkpoint step %d", step)
                _push_ckpt(cfg, ck, run_name)
            if step >= total_steps:
                break
        if step >= total_steps:
            break

    ck = _ckpt_dir(out_dir, step)
    _save_ckpt(ck, model, opt, sched, step, cfg, wandb_id)
    _push_ckpt(cfg, ck, run_name, blocking=True)
    rep = _do_eval(model, va_rows, coll, cfg, step, wb, mode, final=True)
    _record_run(cfg, run_name, phase, "done", rep)
    _phase_verdict(cfg, phase, rep)
    log.info("done. run dir: %s", out_dir)
    return out_dir


class _null_ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _do_eval(model, va_rows, coll, cfg, step, wb, mode, final=False):
    n = min(len(va_rows), int(cfg.eval.subset_samples if not final else cfg.eval.max_samples))
    if n == 0:
        return {}
    rep = run_eval(model, va_rows.rows[:n], coll, model.device,
                   max_new_tokens=int(cfg.eval.max_new_tokens),
                   batch_size=int(cfg.eval.batch_size), mode=mode)
    tag = "FINAL" if final else "eval"
    log.info("%s step %d | WER=%.4f CER=%.4f ctcWER=%.4f (n=%d)", tag, step,
             rep["wer"], rep["cer"], rep.get("ctc_wer", float("nan")), rep["n"])
    if wb:
        wb.log({"eval/wer": rep["wer"], "eval/cer": rep["cer"],
                "eval/ctc_wer": rep.get("ctc_wer", 0.0)}, step=step)
    return rep


def _record_run(cfg, run_name, phase, status, rep):
    led = Ledger(os.path.join(cfg.paths.runs_dir, "runs.json"),
                 repo_id=cfg.repos.runs if getattr(cfg.repos, "runs", None) else None,
                 repo_type="model", path_in_repo="ledger/runs.json",
                 default={"project": cfg.project, "runs": []})
    led.d.setdefault("runs", [])
    led.d["runs"] = [r for r in led.d["runs"] if r.get("name") != run_name]
    led.d["runs"].append({"name": run_name, "phase": phase, "status": status,
                          "updated": iso_now(),
                          "wer": rep.get("wer"), "cer": rep.get("cer")})
    led.save()
    if getattr(cfg.train, "push_to_hub", False):
        led.push(f"run {run_name}: {status}")


def _phase_verdict(cfg, phase, rep):
    if not rep:
        return
    target = float(getattr(cfg.eval, "wer_pass", 0.05))
    wer = rep.get("wer", 1.0)
    if cfg.backend != "real":
        log.info("phase %d WER=%.4f (tiny/synthetic smoke — accuracy not meaningful)", phase, wer)
        return
    if int(phase) == PHASE_JOINT:
        verdict = "PASS ✅ <5% gate met" if wer <= target else \
                  ("BROKEN ❌ pipeline bug" if wer >= float(getattr(cfg.eval, "wer_broken", 0.9))
                   else "PARTIAL ⚠️ needs more clean data/epochs")
        log.info("PHASE-3 VERDICT: %s (WER=%.4f, target<=%.2f)", verdict, wer, target)
    else:
        log.info("phase %d WER=%.4f", phase, wer)
