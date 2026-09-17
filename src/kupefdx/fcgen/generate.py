"""Unified LLM data generation — floor-control (FC) + domain-correction, one harness.

One provider-agnostic streaming client (any OpenAI-compatible /chat/completions endpoint:
Krutrim, OpenAI, vLLM, ...) drives two generators that share the exact same request loop:

  * generate_fc      -> audio-grounded floor-control rows (scenarios + flag rules)
  * generate_domain  -> domain-correction rows (raw transcript -> corrected transcript)

Every request logs ONE colored line (green OK / red FAIL) with latency, rows kept, and the
running token/cost total, so you always see the API working in real time. `--show-stream`
prints the live SSE token stream for a single call. `mock=True` skips the network entirely
and emits deterministic, schema-valid rows for offline smoke tests.

Configure the provider via env (see .env):
    KUPE_LLM_BASE_URL   e.g. https://cloud.olakrutrim.com/v1
    KUPE_LLM_API_KEY    provider key
    KUPE_LLM_MODEL      e.g. gemma-4-31b-it
    KUPE_LLM_MAX_OUT    max output tokens per hit   (default 8000)
    KUPE_LLM_TEMP       sampling temperature        (default 0.8)
    KUPE_LLM_STREAM     1=stream (default), 0=off
    KUPE_LLM_PRICE_IN / KUPE_LLM_PRICE_OUT   $ per 1M tokens (optional, shows cost)
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..constants import FC_TOKENS
from ..env import load_env, log
from ..text import normalize

load_env()  # pull KUPE_LLM_* (+ HF_*) from .env before reading any of them below
from .scenarios import RULES, sample_scenarios
from .schema import SchemaError, blank_row, validate_row

# ───────────────────────────── colored terminal output ─────────────────────────────
_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


class _C:
    OK = "\033[32m"; BAD = "\033[31m"; WARN = "\033[33m"
    INFO = "\033[36m"; DIM = "\033[2m"; BOLD = "\033[1m"; RST = "\033[0m"

    def __getattribute__(self, name):
        v = object.__getattribute__(self, name)
        return v if _USE_COLOR else ""


C = _C()
_PLOCK = threading.Lock()


def cprint(color: str, msg: str) -> None:
    """Thread-safe, timestamped, colored one-liner to stdout (never interleaves)."""
    with _PLOCK:
        print(f"{color}{time.strftime('%H:%M:%S')} | {msg}{C.RST}", flush=True)


# ───────────────────────────── shared token / cost meter ─────────────────────────────
_TOK = {"in": 0, "out": 0, "calls": 0}
_TLOCK = threading.Lock()


def _price():
    return (float(os.environ.get("KUPE_LLM_PRICE_IN", 0)),
            float(os.environ.get("KUPE_LLM_PRICE_OUT", 0)))


def token_report() -> str:
    pi, po = _price()
    cost = _TOK["in"] / 1e6 * pi + _TOK["out"] / 1e6 * po
    money = f" ~${cost:,.2f}" if (pi or po) else ""
    return (f"tok in={_TOK['in']:,} out={_TOK['out']:,} "
            f"total={_TOK['in'] + _TOK['out']:,} · {_TOK['calls']} calls{money}")


# ───────────────────────────── the LLM client ─────────────────────────────
def _cfg() -> dict:
    return {
        "base_url": os.environ.get("KUPE_LLM_BASE_URL", "https://cloud.olakrutrim.com/v1"),
        "api_key": os.environ.get("KUPE_LLM_API_KEY", ""),
        "model": os.environ.get("KUPE_LLM_MODEL", "gemma-4-31b-it"),
    }


_MAX_OUT = int(os.environ.get("KUPE_LLM_MAX_OUT", 8000))
_TEMP = float(os.environ.get("KUPE_LLM_TEMP", 0.8))
_STREAM = os.environ.get("KUPE_LLM_STREAM", "1") != "0"


def _read_sse(resp, on_delta):
    """Consume an OpenAI-compatible SSE stream -> (content, usage)."""
    parts, usage = [], {}
    for raw in resp:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except Exception:
            continue
        if chunk.get("usage"):
            usage = chunk["usage"]
        for ch in chunk.get("choices", []):
            delta = (ch.get("delta") or {}).get("content") or ""
            if delta:
                parts.append(delta)
                if on_delta:
                    on_delta(delta)
    return "".join(parts), usage


def call_llm(messages, cfg=None, timeout=120, stream=None, on_delta=None) -> str:
    """One /chat/completions request. Streams by default; retries on 429 with backoff;
    self-heals the common OpenAI-vs-Gemma param differences (max_tokens / temperature)."""
    cfg = cfg or _cfg()
    stream = _STREAM if stream is None else stream
    params = {"model": cfg["model"], "messages": messages,
              "temperature": _TEMP, "max_tokens": _MAX_OUT}
    if stream:
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg['api_key']}"}
    backoff = 5
    for _ in range(8):
        req = urllib.request.Request(url, data=json.dumps(params).encode(), headers=hdr)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                if params.get("stream"):
                    content, u = _read_sse(r, on_delta)
                else:
                    data = json.load(r)
                    u = data.get("usage", {}) or {}
                    content = data["choices"][0]["message"]["content"]
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:                                   # rate limited -> wait & retry
                ra = e.headers.get("Retry-After")
                wait = int(ra) if (ra and ra.isdigit()) else backoff
                time.sleep(min(wait, 60)); backoff = min(backoff * 2, 60); continue
            msg = e.read().decode()[:300] if e.code == 400 else ""
            if e.code == 400 and "max_completion_tokens" in msg and "max_tokens" in params:
                params["max_completion_tokens"] = params.pop("max_tokens"); continue
            if e.code == 400 and "temperature" in msg:
                params.pop("temperature", None); continue
            if e.code == 400 and params.get("stream") and "stream" in msg:
                params.pop("stream", None); params.pop("stream_options", None); continue
            raise
    else:
        raise RuntimeError("LLM call failed after retries (429 backoff exhausted)")
    if not u:                                                   # provider gave no usage: estimate
        u = {"prompt_tokens": sum(len(m["content"]) for m in messages) // 4,
             "completion_tokens": len(content) // 4}
    with _TLOCK:
        _TOK["in"] += int(u.get("prompt_tokens", 0))
        _TOK["out"] += int(u.get("completion_tokens", 0))
        _TOK["calls"] += 1
    return content


# ───────────────────────────── JSON parsing (truncation-safe) ─────────────────────────────
def _salvage_array(t: str) -> list[dict]:
    """Recover whole objects from a JSON array that got cut off at the token cap."""
    start = t.find("[")
    if start < 0:
        return []
    dec, i, out, n = json.JSONDecoder(), start + 1, [], len(t)
    while i < n:
        while i < n and t[i] in " \t\r\n,":
            i += 1
        if i >= n or t[i] == "]":
            break
        try:
            obj, i = dec.raw_decode(t, i)
        except Exception:
            break
        if isinstance(obj, dict):
            out.append(obj)
    return out


def parse_rows(text: str) -> list[dict]:
    if not text:
        return []
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t).rsplit("```", 1)[0]
    for cand in (t, None):
        if cand is None:
            m = re.search(r"\[.*\]", t, re.DOTALL)
            cand = m.group(0) if m else None
        if not cand:
            continue
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)]
        if isinstance(obj, dict):
            for k in ("rows", "data", "examples", "results"):
                if isinstance(obj.get(k), list):
                    return [r for r in obj[k] if isinstance(r, dict)]
            return [obj]
    return _salvage_array(t)


# ═══════════════════════════════ shared concurrent runner ═══════════════════════════════
def _run(batches, work, *, concurrency, push_cb, sync_cb, sync_every, kind):
    """Drive `work(batch) -> list[row]` across a thread pool. Colored per-hit lines come
    from `work`; here we collect rows, stream them to disk (push_cb), and periodically
    mirror to the Hub (sync_cb) so a crash never loses generated data."""
    out, done = [], 0
    cprint(C.INFO + C.BOLD, f"{kind}: {len(batches)} requests · conc={concurrency} · "
           f"model={_cfg()['model']} · max_out={_MAX_OUT} · stream={_STREAM}")
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(work, b): b for b in batches}
        for fut in as_completed(futs):
            try:
                rows = fut.result()
            except Exception as e:
                cprint(C.BAD, f"{kind} request crashed: {e}")
                rows = []
            out += rows
            if push_cb:
                push_cb(rows)
            done += 1
            if sync_cb and sync_every and done % sync_every == 0:
                sync_cb(done, len(batches))
    cprint(C.INFO + C.BOLD, f"{kind} DONE · {len(out)} rows kept · {token_report()}")
    return out


# ═══════════════════════════════ floor-control generation ═══════════════════════════════
SYSTEM_FC = (
    "You are a meticulous data engineer building English voice-agent floor-control training "
    "data. You output ONLY a JSON array of rows, no prose. Each row grounds its timeline in "
    "the provided audio card. You follow the flag rules exactly and keep the scenario mix.")


def _fc_prompt(batch: list[dict], n_rows: int, scen_hint: list[str]) -> str:
    cards = "\n\n".join(f"CLIP {i} (id={c['id']}, domain={c['domain']}):\n{c['card']}"
                        for i, c in enumerate(batch))
    schema = ('{"id","domain","scenario","transcript","source_clip"(int),'
              '"context":["Agent: ...","User: ..."],'
              '"timeline":[{"kind":"speech|pause","text?","dur_s?","surface?","flag?","t_s"}]}')
    return (f"{RULES}\n\nAUDIO CARDS (ground every row in one clip via source_clip):\n{cards}\n\n"
            f"Produce EXACTLY {n_rows} rows as a JSON array. Row schema:\n{schema}\n"
            f"`context` = 0-3 short prior turns in natural English that justify the scenario. "
            f"Aim for this scenario mix: {scen_hint}.\n"
            f"TOKEN SAVING: keep each row's `transcript` EXACTLY the clip transcript; only decide "
            f"flag placement, surfaces and context. Return ONLY the JSON array.")


def _fc_attach(row: dict, batch: list[dict], k: int) -> dict:
    idx = row.get("source_clip")
    clip = batch[idx] if isinstance(idx, int) and 0 <= idx < len(batch) else batch[k % len(batch)]
    row.setdefault("id", f"{clip['id']}_fc{k:03d}")
    row.setdefault("domain", clip["domain"])
    row.setdefault("lang", "en")
    row["audio"] = clip["audio"]
    row["audio_features"] = clip["features"]
    row["provenance"] = "audio-derived-llm"
    return row


def _fc_mock(batch, n_rows):
    from ..floorcontrol import SURFACES
    scen = sample_scenarios(n_rows, seed=hash(batch[0]["id"]) & 0xffff)
    out = []
    for k, s in enumerate(scen):
        clip = batch[k % len(batch)]
        t = clip["transcript"] or "hello how are you today"
        w = t.split(); mid = max(1, len(w) // 2)
        row = blank_row(f"{clip['id']}_fc{k:03d}", clip["domain"], s, t)
        row["context"] = {"clean_end_of_speech": ["Agent: what seems to be the problem?"],
                          "backchannel": ["Agent: go ahead, tell me"],
                          "thinking_sound": ["Agent: okay, tell me"],
                          "midsentence_pause": ["Agent: yes, go on"],
                          "false_trigger_trap": ["Agent: please tell me"]}.get(s, [])
        tl = [{"kind": "speech", "t_s": 0.1, "text": " ".join(w[:mid])}]
        if s == "clean_end_of_speech":
            tl += [{"kind": "speech", "t_s": 1.0, "text": " ".join(w[mid:])},
                   {"kind": "pause", "t_s": 2.5, "dur_s": 0.6, "flag": "<EOS_SPEECH>"}]
        elif s == "midsentence_pause":
            tl += [{"kind": "pause", "t_s": 1.0, "dur_s": 0.5},
                   {"kind": "speech", "t_s": 1.6, "text": " ".join(w[mid:])}]
        elif s == "backchannel":
            surf = SURFACES["<BC>"][k % len(SURFACES["<BC>"])]
            tl += [{"kind": "pause", "t_s": 1.0, "dur_s": 0.3, "flag": "<BC>", "surface": surf},
                   {"kind": "speech", "t_s": 1.4, "text": " ".join(w[mid:])}]
        elif s == "expression":
            surf = ["oh", "oh no", "wow", "whoa"][k % 4]
            row["context"] = ["Agent: oh, and then what happened?"]
            tl += [{"kind": "pause", "t_s": 1.0, "dur_s": 0.3, "flag": "<BC>", "surface": surf},
                   {"kind": "speech", "t_s": 1.4, "text": " ".join(w[mid:])}]
        elif s == "thinking_sound":
            tl += [{"kind": "speech", "t_s": 1.0, "text": " ".join(w[mid:])},
                   {"kind": "pause", "t_s": 2.4, "dur_s": 0.5, "flag": "<EOS_SPEECH>"},
                   {"kind": "pause", "t_s": 3.0, "dur_s": 0.7, "flag": "<THINK>", "surface": "hmm"}]
        elif s == "sustained_silence":
            tl = [{"kind": "speech", "t_s": 0.1, "text": t},
                  {"kind": "pause", "t_s": 2.0, "dur_s": 1.5, "flag": "<SILENCE>"}]
        elif s == "false_trigger_trap":
            tl = [{"kind": "speech", "t_s": 0.1, "text": "um"},
                  {"kind": "pause", "t_s": 0.6, "dur_s": 0.4},
                  {"kind": "speech", "t_s": 1.0, "text": t}]
        else:
            tl += [{"kind": "speech", "t_s": 1.2, "text": " ".join(w[mid:])}]
        row["timeline"] = tl
        out.append(_fc_attach(row, batch, k))
    return out


_DIAG = {"kept": 0, "dropped": 0, "reasons": {}, "hits": 0, "raw_shown": False}
_DLOCK = threading.Lock()


def _fc_hit(batch, n_rows, mock, on_delta):
    t0 = time.time()
    if mock:
        rows = _fc_mock(batch, n_rows)
    else:
        msgs = [{"role": "system", "content": SYSTEM_FC},
                {"role": "user", "content": _fc_prompt(batch, n_rows, sample_scenarios(n_rows))}]
        rows, raw, err = [], "", None
        for attempt in range(3):
            try:
                raw = call_llm(msgs, on_delta=on_delta)
                rows = parse_rows(raw)
                if rows:
                    break
            except Exception as e:
                err = e
                time.sleep(2 * (attempt + 1))
        if not rows:
            with _DLOCK:
                _DIAG["hits"] += 1; hd = _DIAG["hits"]
                show = not _DIAG["raw_shown"]; _DIAG["raw_shown"] = True
            cprint(C.BAD, f"hit {hd}: FAIL 0 rows in {time.time()-t0:4.1f}s "
                   f"({type(err).__name__ if err else 'empty/parse'}) · {token_report()}")
            if show and raw:
                cprint(C.DIM, f"  raw tail: ...{raw[-160:]}".replace("\n", " "))
            return []
        rows = [_fc_attach(r, batch, k) for k, r in enumerate(rows)]
    valid, n_parsed = [], len(rows)
    for r in rows:
        try:
            valid.append(validate_row(r))
        except SchemaError as e:
            with _DLOCK:
                _DIAG["dropped"] += 1
                key = str(e).split(":")[0][:50]
                _DIAG["reasons"][key] = _DIAG["reasons"].get(key, 0) + 1
    if not mock:
        with _DLOCK:
            _DIAG["hits"] += 1; _DIAG["kept"] += len(valid)
            hd, kept, drop = _DIAG["hits"], _DIAG["kept"], _DIAG["dropped"]
        cprint(C.OK, f"hit {hd}: OK kept {len(valid)}/{n_parsed} in {time.time()-t0:4.1f}s "
               f"· totals kept={kept} dropped={drop} · {token_report()}")
    return valid


def generate_fc(clips, *, rows_per_hit=22, clips_per_hit=5, concurrency=10, mock=False,
                seen_ledger=None, push_cb=None, sync_cb=None, sync_every=0, show_stream=False):
    cfg = _cfg()
    if not mock and not cfg["api_key"]:
        raise RuntimeError("set KUPE_LLM_API_KEY (+ KUPE_LLM_BASE_URL/MODEL) or use mock")
    batches = [clips[i:i + clips_per_hit] for i in range(0, len(clips), clips_per_hit)]
    if seen_ledger is not None:
        batches = [b for b in batches if not seen_ledger.is_done(_bid(b))]
    on_delta = (lambda s: sys.stdout.write(C.DIM + s + C.RST)) if (show_stream and not mock) else None

    def work(b):
        rows = _fc_hit(b, rows_per_hit, mock, on_delta)
        if seen_ledger is not None:
            seen_ledger.mark(_bid(b), "done", rows=len(rows))
        return rows

    rows = _run(batches, work, concurrency=1 if on_delta else concurrency,
                push_cb=push_cb, sync_cb=sync_cb, sync_every=sync_every, kind="FC gen")
    if not mock and _DIAG["reasons"]:
        cprint(C.WARN, f"drop reasons: {dict(sorted(_DIAG['reasons'].items(), key=lambda x:-x[1]))}")
    return rows


# ═══════════════════════════════ domain-correction generation ═══════════════════════════════
SYSTEM_DOM = ("You correct English ASR transcripts for a given domain. Fix domain terminology "
              "and obvious ASR errors using the conversation context. Output ONLY a JSON array "
              "of records. Never change meaning.")

_MOCK_TERMS = {
    "medical": [("high blood pressure", "hypertension"), ("sugar", "blood glucose"),
                ("diabetees", "diabetes"), ("heart beating fast", "tachycardia")],
    "technical": [("server went down", "server is down"), ("ram memory", "RAM")],
    "banking": [("send money", "transfer funds"), ("account no", "account number")],
    "general": [("okay so", "okay")],
}


def _dom_validate(rec: dict) -> dict:
    for k in ("domain", "omni_raw_transcript", "corrected_transcript"):
        if k not in rec:
            raise ValueError(f"missing {k}")
    rec.setdefault("context", []); rec.setdefault("correction_spans", [])
    rec.setdefault("chunk_boundaries_ms", [])
    rec["omni_raw_transcript"] = normalize(rec["omni_raw_transcript"])
    rec["corrected_transcript"] = normalize(rec["corrected_transcript"])
    return rec


def _dom_row(rec: dict, audio: str, rid: str) -> dict:
    rec = _dom_validate(rec)
    return {"id": rid, "lang": "en", "domain": rec["domain"], "scenario": "domain_correction",
            "audio": audio, "text": rec["omni_raw_transcript"],
            "target_sequence": rec["corrected_transcript"], "context": rec.get("context", []),
            "correction_spans": rec.get("correction_spans", []),
            "chunk_boundaries_ms": rec.get("chunk_boundaries_ms", []),
            "timeline": [], "provenance": rec.get("provenance", "domain-llm")}


def _dom_mock(clip: dict) -> dict:
    dom = clip["domain"] if clip["domain"] in _MOCK_TERMS else "general"
    raw, (a, b) = clip["transcript"], _MOCK_TERMS[dom][hash(clip["id"]) % len(_MOCK_TERMS[dom])]
    return {"domain": dom, "omni_raw_transcript": (raw + " " + a).strip(), "context": [],
            "corrected_transcript": (raw + " " + b).strip(),
            "correction_spans": [{"raw": a, "fixed": b, "reason": "domain term"}],
            "provenance": "domain-mock"}


def _dom_hit(batch, mock, on_delta):
    t0 = time.time()
    if mock:
        recs = [_dom_mock(c) for c in batch]
    else:
        cards = "\n".join(f"- clip {i} (id={c['id']}, domain={c['domain']}): \"{c['transcript']}\""
                          for i, c in enumerate(batch))
        user = (f"For each clip, produce a correction record. You MUST inject 1-3 realistic ASR "
                f"errors (domain term, homophone, or word-boundary slip) into omni_raw_transcript, "
                f"then give the fully corrected_transcript. The two MUST differ — never return them "
                f"identical, and every correction_spans[i].raw MUST actually appear in "
                f"omni_raw_transcript. Clips:\n{cards}\n"
                f'Schema: {{"domain","omni_raw_transcript","context":[],"corrected_transcript",'
                f'"correction_spans":[{{"raw","fixed","reason"}}]}}. Return ONLY the JSON array.')
        try:
            recs = parse_rows(call_llm([{"role": "system", "content": SYSTEM_DOM},
                                        {"role": "user", "content": user}], on_delta=on_delta))
        except Exception as e:
            cprint(C.BAD, f"domain hit FAIL in {time.time()-t0:4.1f}s ({e})")
            recs = []
    out, noop = [], 0
    for i, r in enumerate(recs):
        clip = batch[i % len(batch)]
        try:
            row = _dom_row(r, clip["audio"], f"{clip['id']}_dom{i:02d}")
        except Exception:
            continue
        if row["text"].strip() == row["target_sequence"].strip():   # no-op teaches nothing -> drop
            noop += 1
            continue
        out.append(row)
    if noop:
        with _DLOCK:
            _DIAG["dropped"] += noop
    if not mock:
        with _DLOCK:
            _DIAG["hits"] += 1; hd = _DIAG["hits"]
        cprint(C.OK if out else C.WARN,
               f"domain hit {hd}: kept {len(out)}/{len(recs)} ({noop} no-op dropped) "
               f"in {time.time()-t0:4.1f}s · {token_report()}")
    return out


def generate_domain(clips, *, clips_per_hit=6, concurrency=10, mock=False,
                    seen_ledger=None, push_cb=None, sync_cb=None, sync_every=0, show_stream=False):
    cfg = _cfg()
    if not mock and not cfg["api_key"]:
        raise RuntimeError("set KUPE_LLM_API_KEY (+ KUPE_LLM_BASE_URL/MODEL) or use mock")
    batches = [clips[i:i + clips_per_hit] for i in range(0, len(clips), clips_per_hit)]
    if seen_ledger is not None:
        batches = [b for b in batches if not seen_ledger.is_done(_bid(b))]
    on_delta = (lambda s: sys.stdout.write(C.DIM + s + C.RST)) if (show_stream and not mock) else None

    def work(b):
        rows = _dom_hit(b, mock, on_delta)
        if seen_ledger is not None:
            seen_ledger.mark(_bid(b), "done", rows=len(rows))
        return rows

    return _run(batches, work, concurrency=1 if on_delta else concurrency,
                push_cb=push_cb, sync_cb=sync_cb, sync_every=sync_every, kind="Domain gen")


def _bid(batch: list[dict]) -> str:
    return "batch_" + "_".join(c["id"] for c in batch)[:80]
