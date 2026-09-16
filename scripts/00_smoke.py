#!/usr/bin/env python3
"""End-to-end smoke test on synthetic data (CPU/MPS). Run before any H100 job.

    python scripts/00_smoke.py                    # uses configs/smoke.yaml
    python scripts/00_smoke.py --config configs/smoke.yaml --set train.epochs=1
"""
import argparse

import _bootstrap  # noqa: F401
from kupefdx.config import load_config
from kupefdx.smoke import run_smoke


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. train.lr=1e-4")
    a = ap.parse_args()
    cfg = load_config(a.config, overrides=a.set)
    ok = run_smoke(cfg)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
