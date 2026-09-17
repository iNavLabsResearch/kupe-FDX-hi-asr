"""LLM agent that generates floor-control rows from audio-grounded clip batches.

- One LLM hit produces ~20-25 rows across scenarios, grounded in the batch's audio cards.
- Concurrency: a thread pool of `concurrency` (default 10) in-flight requests, tqdm bar.
- Provider-agnostic: any OpenAI-compatible /chat/completions endpoint (Sarvam, OpenAI,
  vLLM, ...). Configure via env: KUPE_LLM_BASE_URL, KUPE_LLM_API_KEY, KUPE_LLM_MODEL.
- `--mock` (mock=True) skips the network and emits deterministic, schema-valid rows so the
  whole pipeline can be smoke-tested offline.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..env import log

# ---- live token/cost meter (shared across the concurrent hits) ----
_TOK = {"in": 0, "out": 0, "calls": 0}
_TLOCK = threading.Lock()


def _price():
    return (float(os.environ.get("KUPE_LLM_PRICE_IN", 0)),    # $ per 1M input tokens
            float(os.environ.get("KUPE_LLM_PRICE_OUT", 0)))   # $ per 1M output tokens


def token_report() -> str:
    pi, po = _price()
    cost = _TOK["in"] / 1e6 * pi + _TOK["out"] / 1e6 * po
    money = f" ~${cost:,.2f}" if (pi or po) else " (set KUPE_LLM_PRICE_IN/OUT for $)"
    return (f"tokens so far: in={_TOK['in']:,} out={_TOK['out']:,} "
            f"total={_TOK['in'] + _TOK['out']:,} over {_TOK['calls']} calls{money}")
from .scenarios import RULES, sample_scenarios
from .schema import SchemaError, validate_row

try:
    from tqdm import tqdm
except Exception:                       # minimal fallback bar
    def tqdm(it, total=None, desc=None):
        return it


def _llm_cfg():
    return {
        "base_url": os.environ.get("KUPE_LLM_BASE_URL", "https://api.openai.com/v1"),
        "api_key": os.environ.get("KUPE_LLM_API_KEY", ""),
        "model": os.environ.get("KUPE_LLM_MODEL", "gpt-5.6-luna"),
    }


SYSTEM = (
    "You are a meticulous data engineer building English voice-agent floor-control training "
    "data. You output ONLY a JSON array of rows, no prose. Each row grounds its timeline in "
    "the provided audio card. You follow the flag rules exactly and keep the scenario mix.")


def build_user_prompt(batch: list[dict], n_rows: int, scen_hint: list[str]) -> str:
    cards = "\n\n".join(
        f"CLIP {i} (id={c['id']}, domain={c['domain']}):\n{c['card']}"
        for i, c in enumerate(batch))
    schema = ('{"id","domain","scenario","transcript","source_clip"(int),'
              '"context":["Agent: ...","User: ..."],'
              '"timeline":[{"kind":"speech|pause","text?","dur_s?","surface?","flag?","t_s"}]}')
    return (
        f"{RULES}\n\n"
        f"AUDIO CARDS (ground every row in one of these clips via source_clip):\n{cards}\n\n"
        f"Produce EXACTLY {n_rows} rows as a JSON array. Row schema:\n{schema}\n"
        f"`context` = 0-3 short prior conversation turns in natural English that make the scenario "
        f"natural — e.g. an Agent question before a user turn, so <EOS_SPEECH> / <BC> placement "
        f"is justified by the dialogue. Aim for this scenario mix: {scen_hint}.\n"
        f"TOKEN SAVING: keep each row's `transcript` EXACTLY the clip transcript (do not rewrite "
        f"it); only decide flag placement, surfaces and context. Return ONLY the JSON array.")


def call_llm(messages, cfg, timeout=90) -> str:
    # newer OpenAI models use max_completion_tokens and only accept default temperature.
    params = {"model": cfg["model"], "messages": messages,
              "temperature": 0.8, "max_completion_tokens": 4000}
    url = cfg["base_url"].rstrip("/") + "/chat/completions"
    hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {cfg['api_key']}"}
    backoff = 5
    for _ in range(8):
        req = urllib.request.Request(url, data=json.dumps(params).encode(), headers=hdr)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.load(r)
            break
        except urllib.error.HTTPError as e:
            if e.code == 429:                                       # rate limited -> wait, retry
                ra = e.headers.get("Retry-After")
                wait = int(ra) if (ra and ra.isdigit()) else backoff
                time.sleep(min(wait, 60)); backoff = min(backoff * 2, 60); continue
            msg = e.read().decode()[:300] if e.code == 400 else ""
            if e.code == 400 and "temperature" in msg:
                params.pop("temperature", None); continue          # model wants default temp
            if e.code == 400 and "max_tokens" in msg and "max_completion_tokens" not in params:
                params["max_completion_tokens"] = params.pop("max_tokens", 4000); continue
            raise
    else:
        raise RuntimeError("LLM call failed after retries (429/backoff exhausted)")
    u = data.get("usage", {}) or {}
    with _TLOCK:
        _TOK["in"] += int(u.get("prompt_tokens", 0))
        _TOK["out"] += int(u.get("completion_tokens", 0))
        _TOK["calls"] += 1
    return data["choices"][0]["message"]["content"]


def parse_rows(text: str) -> list[dict]:
    if not text:
        return []
    t = text.strip()
    if t.startswith("```"):                       # strip ```json ... ``` fences
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t).rsplit("```", 1)[0]
    # try whole thing, then an object with a rows/data key, then the first [...] block
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
            for key in ("rows", "data", "examples", "results"):
                if isinstance(obj.get(key), list):
                    return [r for r in obj[key] if isinstance(r, dict)]
            return [obj]                          # a single row object
    return []


def _attach_audio(row: dict, batch: list[dict], k: int) -> dict:
    idx = row.get("source_clip")
    clip = batch[idx] if isinstance(idx, int) and 0 <= idx < len(batch) else batch[k % len(batch)]
    row.setdefault("id", f"{clip['id']}_fc{k:03d}")
    row.setdefault("domain", clip["domain"])
    row.setdefault("lang", "en")
    row["audio"] = clip["audio"]
    row["audio_features"] = clip["features"]
    row["provenance"] = "audio-derived-llm"
    return row


# ---------------------------------------------------------------- mock generator
def _mock_rows(batch: list[dict], n_rows: int) -> list[dict]:
    from .schema import blank_row
    scen = sample_scenarios(n_rows, seed=hash(batch[0]["id"]) & 0xffff)
    out = []
    for k, s in enumerate(scen):
        clip = batch[k % len(batch)]
        t = clip["transcript"] or "hello how are you today"
        words = t.split()
        mid = max(1, len(words) // 2)
        row = blank_row(f"{clip['id']}_fc{k:03d}", clip["domain"], s, t)
        # a short prior-turn context that makes the scenario natural
        ctx_by_scen = {
            "clean_end_of_speech": ["Agent: what seems to be the problem?"],
            "backchannel": ["Agent: go ahead, tell me"],
            "thinking_sound": ["Agent: okay, tell me"],
            "midsentence_pause": ["Agent: yes, go on"],
            "false_trigger_trap": ["Agent: please tell me"],
        }
        row["context"] = ctx_by_scen.get(s, [])
        tl = [{"kind": "speech", "t_s": 0.1, "text": " ".join(words[:mid])}]
        if s == "clean_end_of_speech":
            tl += [{"kind": "speech", "t_s": 1.0, "text": " ".join(words[mid:])},
                   {"kind": "pause", "t_s": 2.5, "dur_s": 0.6, "flag": "<EOS_SPEECH>"}]
        elif s == "midsentence_pause":
            tl += [{"kind": "pause", "t_s": 1.0, "dur_s": 0.5},   # trap: no flag
                   {"kind": "speech", "t_s": 1.6, "text": " ".join(words[mid:])}]
        elif s == "backchannel":
            from ..floorcontrol import SURFACES
            surf = SURFACES["<BC>"][k % len(SURFACES["<BC>"])]
            tl += [{"kind": "pause", "t_s": 1.0, "dur_s": 0.3, "flag": "<BC>", "surface": surf},
                   {"kind": "speech", "t_s": 1.4, "text": " ".join(words[mid:])}]
        elif s == "expression":
            # mock stays context-safe (neutral surprise/concern); the real LLM picks laughter
            # vs sigh from actual context per the rules. This avoids absurd mock samples.
            surf = ["oh", "oh no", "wow", "whoa"][k % 4]
            row["context"] = ["Agent: oh, and then what happened?"]
            tl += [{"kind": "pause", "t_s": 1.0, "dur_s": 0.3, "flag": "<BC>", "surface": surf},
                   {"kind": "speech", "t_s": 1.4, "text": " ".join(words[mid:])}]
        elif s == "thinking_sound":
            tl += [{"kind": "speech", "t_s": 1.0, "text": " ".join(words[mid:])},
                   {"kind": "pause", "t_s": 2.4, "dur_s": 0.5, "flag": "<EOS_SPEECH>"},
                   {"kind": "pause", "t_s": 3.0, "dur_s": 0.7, "flag": "<THINK>", "surface": "hmm"}]
        elif s == "sustained_silence":
            tl = [{"kind": "speech", "t_s": 0.1, "text": t},
                  {"kind": "pause", "t_s": 2.0, "dur_s": 1.5, "flag": "<SILENCE>"}]
        else:  # nothing_happens / false_trigger_trap / barge_in -> no flag
            if s == "false_trigger_trap":
                tl = [{"kind": "speech", "t_s": 0.1, "text": "um"},
                      {"kind": "pause", "t_s": 0.6, "dur_s": 0.4},
                      {"kind": "speech", "t_s": 1.0, "text": t}]
            else:
                tl += [{"kind": "speech", "t_s": 1.2, "text": " ".join(words[mid:])}]
        row["timeline"] = tl
        out.append(_attach_audio(row, batch, k))
    return out


# ---- shared diagnostics so a silent 0-rows run is impossible ----
_DIAG = {"parsed": 0, "kept": 0, "dropped": 0, "reasons": {}, "raw_dumped": False,
         "hits_done": 0}
_DLOCK = threading.Lock()


def _diag_drop(reason):
    key = str(reason).split(":")[0][:60]
    with _DLOCK:
        _DIAG["dropped"] += 1
        _DIAG["reasons"][key] = _DIAG["reasons"].get(key, 0) + 1


def _one_hit(batch, n_rows, cfg, mock) -> list[dict]:
    t0 = time.time()
    if mock:
        rows = _mock_rows(batch, n_rows)
        n_parsed = len(rows)
    else:
        scen = sample_scenarios(n_rows)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_user_prompt(batch, n_rows, scen)}]
        raw = ""
        rows = []
        err = None
        for attempt in range(3):
            try:
                raw = call_llm(msgs, cfg)
                rows = parse_rows(raw)
                if rows:
                    break
            except Exception as e:
                err = e
                log.warning("LLM hit failed (attempt %d): %s", attempt + 1, e)
                time.sleep(2 * (attempt + 1))
        n_parsed = len(rows)
        if not rows:                              # every hit result is logged, pass or fail
            with _DLOCK:
                _DIAG["hits_done"] += 1
                hd = _DIAG["hits_done"]
            log.info("hit %d: FAIL 0 rows in %.1fs (%s)", hd, time.time() - t0,
                     type(err).__name__ if err else "empty/parse")
        if not rows:                              # parse failed -> show WHY, once
            with _DLOCK:
                if not _DIAG["raw_dumped"] and raw:
                    _DIAG["raw_dumped"] = True
                    log.warning("FC parse produced 0 rows. RAW model output (first 600 chars):\n%s",
                                raw[:600])
            _diag_drop("parse: no JSON array in response")
            return []
        with _DLOCK:
            _DIAG["parsed"] += len(rows)
        rows = [_attach_audio(r, batch, k) for k, r in enumerate(rows)]
    valid = []
    for r in rows:
        try:
            valid.append(validate_row(r))
            with _DLOCK:
                _DIAG["kept"] += 1
        except SchemaError as e:
            _diag_drop(e)
            with _DLOCK:
                if not _DIAG["raw_dumped"] and not mock:
                    _DIAG["raw_dumped"] = True
                    log.warning("FC first invalid row dropped (%s). ROW was:\n%s",
                                e, json.dumps(r)[:600])
    if not mock:
        with _DLOCK:
            _DIAG["hits_done"] += 1
            hd, kept, drop = _DIAG["hits_done"], _DIAG["kept"], _DIAG["dropped"]
        log.info("hit %d: OK kept %d/%d rows in %.1fs | totals kept=%d dropped=%d out=%d tok",
                 hd, len(valid), n_parsed, time.time() - t0, kept, drop,
                 _TOK["in"] + _TOK["out"])
    return valid


def generate(clips: list[dict], *, rows_per_hit=22, clips_per_hit=5,
             concurrency=10, mock=False, seen_ledger=None, push_cb=None) -> list[dict]:
    """clips: [{id, audio, transcript, domain, features, card}]. Returns validated rows."""
    cfg = _llm_cfg()
    if not mock and not cfg["api_key"]:
        raise RuntimeError("set KUPE_LLM_API_KEY (and KUPE_LLM_BASE_URL/MODEL), or use --mock")
    batches = [clips[i:i + clips_per_hit] for i in range(0, len(clips), clips_per_hit)]
    if seen_ledger is not None:
        batches = [b for b in batches if not seen_ledger.is_done(_batch_id(b))]
    log.info("FC gen: %d clips -> %d hits x ~%d rows (mock=%s, conc=%d)",
             len(clips), len(batches), rows_per_hit, mock, concurrency)
    out = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_one_hit, b, rows_per_hit, cfg, mock): b for b in batches}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="fc-gen"):
            b = futs[fut]
            try:
                rows = fut.result()
            except Exception as e:
                log.warning("batch failed: %s", e)
                rows = []
            out += rows
            if seen_ledger is not None:
                seen_ledger.mark(_batch_id(b), "done", rows=len(rows))
            if push_cb:
                push_cb(rows)
            if not mock and _TOK["calls"] % 10 == 0:
                log.info("[live] hits=%d/%d kept=%d dropped=%d | %s",
                         _TOK["calls"], len(batches), _DIAG["kept"], _DIAG["dropped"],
                         token_report())
    if not mock:
        log.info("[cost] FINAL %s", token_report())
        log.info("[diag] parsed=%d kept=%d dropped=%d", _DIAG["parsed"], _DIAG["kept"],
                 _DIAG["dropped"])
        if _DIAG["reasons"]:
            log.info("[diag] drop reasons: %s",
                     dict(sorted(_DIAG["reasons"].items(), key=lambda x: -x[1])))
        if _DIAG["kept"] == 0:
            log.error("[diag] 0 valid rows — the RAW sample above shows what the model returned; "
                      "fix the prompt/schema before spending more tokens.")
    return out


def _batch_id(batch: list[dict]) -> str:
    return "batch_" + "_".join(c["id"] for c in batch)[:80]
