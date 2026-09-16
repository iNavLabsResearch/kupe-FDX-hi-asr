"""Config loading: one YAML -> nested, dot-accessible object, with env/CLI overrides.

`{owner}` in `repos` is resolved from `owner` (overridable by HF_OWNER/HF_USERNAME).
Any field is overridable from the env as KUPE_<SECTION>_<KEY> (e.g. KUPE_TRAIN_LR)
or by `--set section.key=value` CLI flags handled in the scripts.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import yaml

_HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG_PATH = os.path.join(_HERE, "configs", "smoke.yaml")


class Config(SimpleNamespace):
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        out: dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [i.to_dict() if isinstance(i, Config) else i for i in v]
            else:
                out[k] = v
        return out


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config(**{k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _coerce(cur: Any, val: str) -> Any:
    """Best-effort coerce a string override to the type of the existing value."""
    if isinstance(cur, bool):
        return val.lower() in ("1", "true", "yes", "on")
    if isinstance(cur, int) and not isinstance(cur, bool):
        try:
            return int(val)
        except ValueError:
            return float(val)
    if isinstance(cur, float):
        return float(val)
    return val


def _apply_dotted(cfg: dict, dotted: str, val: str) -> None:
    parts = dotted.split(".")
    d = cfg
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    key = parts[-1]
    d[key] = _coerce(d.get(key), val)


def load_config(path: str | None = None, overrides: list[str] | None = None) -> Config:
    path = path or os.environ.get("KUPE_CONFIG", DEFAULT_CONFIG_PATH)
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # env overrides: KUPE_TRAIN_LR=1e-4 -> raw["train"]["lr"]
    for k, v in os.environ.items():
        if not k.startswith("KUPE_"):
            continue
        rest = k[len("KUPE_"):].lower().split("_", 1)
        if len(rest) == 2 and rest[0] in raw and isinstance(raw[rest[0]], dict):
            raw[rest[0]][rest[1]] = _coerce(raw[rest[0]].get(rest[1]), v)

    for ov in overrides or []:
        if "=" in ov:
            dotted, val = ov.split("=", 1)
            _apply_dotted(raw, dotted.strip(), val.strip())

    owner = os.environ.get("HF_OWNER") or os.environ.get("HF_USERNAME") or raw.get("owner")
    raw["owner"] = owner
    repos = raw.get("repos", {})
    for k, v in list(repos.items()):
        if isinstance(v, str):
            repos[k] = v.format(owner=owner)
    raw["repos"] = repos

    cfg = _wrap(raw)
    cfg._path = path
    return cfg
