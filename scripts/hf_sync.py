#!/usr/bin/env python3
"""Keep everything synced to the Hub after each stage (PLAN §5). Resumable: only the
named artifacts are (re)uploaded; ledgers carry the resume state so any box continues.

    python scripts/hf_sync.py push --config configs/gpu.yaml --what manifests
    python scripts/hf_sync.py push --config configs/gpu.yaml --what run --run <run_name>
    python scripts/hf_sync.py pull --config configs/gpu.yaml --what manifests
"""
import argparse
import os

import _bootstrap  # noqa: F401
from kupefdx.config import load_config
from kupefdx.env import ensure_repo, hf_login, log, upload_folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "pull"])
    ap.add_argument("--config", default="configs/gpu.yaml")
    ap.add_argument("--what", required=True,
                    choices=["manifests", "feats", "run", "model"])
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    hf_login()

    if a.cmd == "push":
        if a.what == "manifests":
            ensure_repo(cfg.repos.data, "dataset")
            upload_folder(cfg.paths.ledger_dir, cfg.repos.data, "dataset",
                          path_in_repo="manifests", commit_message="sync manifests")
        elif a.what == "feats":
            ensure_repo(cfg.repos.data, "dataset")
            upload_folder(cfg.paths.feats_dir, cfg.repos.data, "dataset",
                          path_in_repo="encoded/feats", commit_message="sync feats")
        elif a.what == "run":
            assert a.run, "--run required"
            ensure_repo(cfg.repos.runs, "model")
            upload_folder(os.path.join(cfg.paths.runs_dir, a.run), cfg.repos.runs, "model",
                          path_in_repo=f"runs/{a.run}", commit_message=f"sync {a.run}")
        elif a.what == "model":
            assert a.run, "--run required"
            ensure_repo(cfg.repos.model, "model")
            md = os.path.join(cfg.paths.runs_dir, a.run)
            # push the latest checkpoint's saved model dir
            cks = sorted([d for d in os.listdir(md) if d.startswith("checkpoint-")],
                         key=lambda c: int(c.rsplit("-", 1)[-1]))
            upload_folder(os.path.join(md, cks[-1]), cfg.repos.model, "model",
                          commit_message=f"latest model from {a.run}")
    else:  # pull
        from huggingface_hub import snapshot_download
        repo = {"manifests": cfg.repos.data, "feats": cfg.repos.data,
                "run": cfg.repos.runs, "model": cfg.repos.model}[a.what]
        rtype = "dataset" if a.what in ("manifests", "feats") else "model"
        p = snapshot_download(repo, repo_type=rtype)
        log.info("pulled %s -> %s", repo, p)


if __name__ == "__main__":
    main()
