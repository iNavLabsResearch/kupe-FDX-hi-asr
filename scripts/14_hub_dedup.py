#!/usr/bin/env python3
"""Stage 14 — remove PROVEN-redundant shard folders from the Hub data repo.

Two classes of redundancy (both verified by CONTENT, never by name alone):
  1. whole-folder content-duplicates — same set of (text, rounded-dur) clips stored under
     two folder names (worker-namespace `..__wNxK_..` vs the old `.._n500_shard_..`).
     Keeps one per group (prefers the worker-namespace copy), deletes the rest.
  2. the orphan `openslr_librispeech_asr` namespace (pre-config/split-fix run) — deleted
     ONLY for folders whose every clip is proven to exist in a surviving non-orphan folder.

DRY-RUN by default (prints the plan, deletes nothing). Add --apply to actually delete.

    python scripts/14_hub_dedup.py                 # dry-run
    python scripts/14_hub_dedup.py --apply         # delete redundant folders
"""
import argparse
import collections
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

import _bootstrap  # noqa: F401
from kupefdx.config import load_config
from kupefdx.env import log, require_token


def src_of(sid):
    s = re.sub(r"(__w\d+x\d+)?_n\d+_shard_\d+$", "", sid)
    return re.sub(r"_shard_\d+$", "", s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/en.yaml")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    a = ap.parse_args()
    cfg = load_config(a.config)
    rid = cfg.repos.data
    tok = require_token()
    from huggingface_hub import CommitOperationDelete, HfApi, hf_hub_download
    api = HfApi()

    files = api.list_repo_files(rid, repo_type="dataset", token=tok)
    man_folders = sorted({f.split("/")[1] for f in files
                          if f.startswith("encoded/") and f.endswith("/manifest.jsonl")})
    log.info("scanning %d shard folders on %s", len(man_folders), rid)

    def fetch(sid):
        for _ in range(4):
            try:
                return sid, hf_hub_download(rid, f"encoded/{sid}/manifest.jsonl",
                                            repo_type="dataset", token=tok, etag_timeout=60)
            except Exception:
                time.sleep(2)
        return sid, None

    folder_keys, folder_hash = {}, {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for sid, p in ex.map(fetch, man_folders):
            if not p:
                raise SystemExit(f"could not fetch manifest for {sid}; aborting (no partial delete)")
            keys, h = set(), hashlib.md5()
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    k = ((r.get("text", "") or "").strip().lower(),
                         round(float(r.get("dur", 0) or 0), 1))
                    keys.add(k)
                    h.update((k[0] + f"|{k[1]}").encode("utf-8", "ignore"))
            folder_keys[sid], folder_hash[sid] = keys, h.hexdigest()

    groups = collections.defaultdict(list)
    for sid, hh in folder_hash.items():
        groups[hh].append(sid)
    to_delete = set()
    for sids in groups.values():
        if len(sids) > 1:
            keep = sorted(sids, key=lambda s: (0 if "__w" in s else 1, s))[0]
            to_delete.update(s for s in sids if s != keep)
    dup_n = len(to_delete)

    orphan = [s for s in man_folders if src_of(s) == "openslr_librispeech_asr"]
    survivors_keys = set()
    for sid, keys in folder_keys.items():
        if src_of(sid) != "openslr_librispeech_asr" and sid not in to_delete:
            survivors_keys |= keys
    safe_orphan = [s for s in orphan if folder_keys[s] <= survivors_keys]
    to_delete.update(safe_orphan)

    log.info("whole-folder dup redundancies: %d", dup_n)
    log.info("orphan folders: %d (safe to delete: %d)", len(orphan), len(safe_orphan))
    log.info("TOTAL folders to delete: %d  (of %d) -> %d remain",
             len(to_delete), len(man_folders), len(man_folders) - len(to_delete))
    for s in sorted(to_delete)[:8]:
        log.info("   del %s", s)

    if not a.apply:
        log.info("DRY-RUN — nothing deleted. Re-run with --apply to delete.")
        return
    ops = [CommitOperationDelete(path_in_repo=f"encoded/{s}/") for s in sorted(to_delete)]
    B = 60
    for i in range(0, len(ops), B):
        api.create_commit(rid, repo_type="dataset", operations=ops[i:i + B], token=tok,
                          commit_message=f"dedup: remove {len(ops[i:i+B])} redundant/orphan shard folders")
        log.info("committed delete %d..%d", i, i + len(ops[i:i + B]))
    log.info("DONE. deleted %d folders; %d remain", len(to_delete),
             len(man_folders) - len(to_delete))


if __name__ == "__main__":
    main()
