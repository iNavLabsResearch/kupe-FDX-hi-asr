#!/usr/bin/env python3
"""Train one phase. Resumable.

    python scripts/03_train.py --config configs/gpu.yaml --phase 1
    python scripts/03_train.py --config configs/gpu.yaml --phase 3 --resume auto
    python scripts/03_train.py --config configs/gpu.yaml --phase 3 --resume <run_name>

Phases: 1 CTC encoder | 2 projector+Nandi align | 3 joint (<5% gate) |
        4 floor-control | 5 domain correction.
"""
import argparse

import _bootstrap  # noqa: F401
from kupefdx.config import load_config
from kupefdx.train import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/gpu.yaml")
    ap.add_argument("--phase", type=int, required=True, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--resume", default=None, help="'auto', a step number, or a run name (same phase)")
    ap.add_argument("--init-from", default=None,
                    help="warm-start weights from a finished phase's run (new phase, fresh optimizer)")
    ap.add_argument("--set", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, overrides=a.set)
    out = train(cfg, a.phase, resume=a.resume, init_from=a.init_from)
    print("run dir:", out)


if __name__ == "__main__":
    main()
