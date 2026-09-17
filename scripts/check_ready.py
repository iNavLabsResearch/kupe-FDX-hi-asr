#!/usr/bin/env python3
"""ONE readiness gate — are we ready to train? Checks data + quality + generated sets.

Runs three checks and prints a single verdict (exit 0 = ready, 1 = not):
  1. encoded ASR corpus   (train.jsonl): hours, NPZ shape/NaN sanity, dup ratio
  2. floor-control data    (fc.jsonl):    schema, scenario/flag mix, dupes
  3. domain-correction data(domain.jsonl):schema, dupes

    python scripts/check_ready.py --config configs/en.yaml          # all three
    python scripts/check_ready.py --config configs/en.yaml --skip-gen   # base corpus only
"""
import argparse

import _bootstrap  # noqa: F401
from kupefdx.checks import check_data, check_gen
from kupefdx.config import load_config
from kupefdx.fcgen.generate import C, cprint


def section(title, ok, lines):
    cprint(C.INFO + C.BOLD, f"── {title} ──")
    for ln in lines:
        cprint(C.BAD if ln.startswith("FAIL") else C.DIM, "   " + ln)
    cprint(C.OK if ok else C.BAD, f"   {'PASS ✅' if ok else 'FAIL ❌'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/en.yaml")
    ap.add_argument("--train", default="data/manifests/train.jsonl")
    ap.add_argument("--fc", default="data/manifests/fc.jsonl")
    ap.add_argument("--domain", default="data/manifests/domain.jsonl")
    ap.add_argument("--sample", type=int, default=500, help="clips to NPZ-sanity-check")
    ap.add_argument("--skip-gen", action="store_true", help="check the base corpus only")
    a = ap.parse_args()
    cfg = load_config(a.config)
    lang = getattr(cfg, "lang", "en")
    dim = int(getattr(cfg.model, "d_model", 512)) if hasattr(cfg, "model") else 512

    results = []
    ok, lines = check_data(cfg, a.train, a.sample, dim)
    results.append(section("1/3 encoded ASR corpus", ok, lines))
    if not a.skip_gen:
        ok, lines = check_gen(a.fc, "fc", lang)
        results.append(section("2/3 floor-control data", ok, lines))
        ok, lines = check_gen(a.domain, "domain", lang)
        results.append(section("3/3 domain-correction data", ok, lines))

    ready = all(results)
    cprint((C.OK if ready else C.BAD) + C.BOLD,
           "═" * 40 + ("\n  READY FOR TRAINING ✅" if ready else "\n  NOT READY ❌"))
    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
