"""Ledgers: durable JSON state so no stage is ever recomputed, and any box resumes
exactly where another left off. Atomic writes (temp + rename) survive a crash.

Two ledger flavours:
  * Ledger      — a small dict store, optionally mirrored to a Hub file.
  * ShardLedger — per-shard state machine (pending|done|failed) for data/encode
                  stages, so re-running a script skips `done` shards only.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from typing import Any

from .env import hf_token, log, require_token


def atomic_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(path)), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log.warning("ledger read failed (%s): %s", path, e)
        return None


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class Ledger:
    def __init__(self, local_path: str, repo_id: str | None = None,
                 repo_type: str = "dataset", path_in_repo: str | None = None,
                 default: dict | None = None):
        self.local_path = local_path
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.path_in_repo = path_in_repo or os.path.basename(local_path)
        self.d: dict = read_json(local_path) or dict(default or {})

    def save(self) -> None:
        self.d["updated"] = iso_now()
        atomic_write_json(self.local_path, self.d)

    def push(self, message: str = "update ledger") -> None:
        if not self.repo_id:
            return
        from huggingface_hub import upload_file
        self.save()
        try:
            upload_file(path_or_fileobj=self.local_path, path_in_repo=self.path_in_repo,
                        repo_id=self.repo_id, repo_type=self.repo_type,
                        token=require_token(), commit_message=message)
        except Exception as e:
            log.warning("ledger push failed (%s): %s", self.path_in_repo, e)

    def sync_from_hub(self, merge=None) -> "Ledger":
        if not self.repo_id:
            return self
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(self.repo_id, self.path_in_repo, repo_type=self.repo_type,
                                token=hf_token())
            hub = read_json(p) or {}
        except Exception:
            hub = {}
        if hub and merge:
            self.d = merge(self.d, hub)
            self.save()
        return self


class ShardLedger:
    """Per-shard state machine. `states` maps shard_id -> 'pending'|'done'|'failed'.
    Re-running a stage processes only shards not already 'done'."""

    def __init__(self, local_path: str, stage: str, repo_id: str | None = None,
                 path_in_repo: str | None = None):
        self.local_path = local_path
        self.stage = stage
        self.repo_id = repo_id
        self.path_in_repo = path_in_repo or f"ledger/{stage}.json"
        self.d = read_json(local_path) or {"stage": stage, "states": {}, "meta": {}}
        self.d.setdefault("states", {})
        self.d.setdefault("meta", {})

    def state(self, shard_id: str) -> str:
        return self.d["states"].get(shard_id, "pending")

    def is_done(self, shard_id: str) -> bool:
        return self.state(shard_id) == "done"

    def pending(self, shard_ids) -> list[str]:
        return [s for s in shard_ids if not self.is_done(s)]

    def mark(self, shard_id: str, state: str, **meta) -> None:
        self.d["states"][shard_id] = state
        if meta:
            self.d["meta"].setdefault(shard_id, {}).update(meta)
        self.save()

    def counts(self) -> dict:
        c = {"pending": 0, "done": 0, "failed": 0}
        for v in self.d["states"].values():
            c[v] = c.get(v, 0) + 1
        return c

    def total_meta(self, key: str) -> float:
        return float(sum(m.get(key, 0) for m in self.d["meta"].values()))

    def save(self) -> None:
        self.d["updated"] = iso_now()
        atomic_write_json(self.local_path, self.d)

    def push(self, message: str = "update shard ledger") -> None:
        if not self.repo_id:
            return
        from huggingface_hub import upload_file
        self.save()
        try:
            upload_file(path_or_fileobj=self.local_path, path_in_repo=self.path_in_repo,
                        repo_id=self.repo_id, repo_type="dataset",
                        token=require_token(), commit_message=message)
        except Exception as e:
            log.warning("shard ledger push failed: %s", e)
