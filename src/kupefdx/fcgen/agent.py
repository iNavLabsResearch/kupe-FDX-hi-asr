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
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..env import log
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
    "You are a meticulous data engineer building Hindi voice-agent floor-control training "
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
        f"`context` = 0-3 short prior conversation turns in DEVANAGARI that make the scenario "
        f"natural — e.g. an Agent question before a user turn, so <EOS_SPEECH> / <BC> placement "
        f"is justified by the dialogue. Aim for this scenario mix: {scen_hint}.\n"
        f"TOKEN SAVING: keep each row's `transcript` EXACTLY the clip transcript (do not rewrite "
        f"it); only decide flag placement, surfaces and context. Return ONLY the JSON array.")


def call_llm(messages, cfg, timeout=90) -> str:
    body = json.dumps({"model": cfg["model"], "messages": messages,
                       "temperature": 0.8, "max_tokens": 4000}).encode()
    req = urllib.request.Request(
        cfg["base_url"].rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {cfg['api_key']}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"]


def parse_rows(text: str) -> list[dict]:
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        rows = json.loads(m.group(0))
        return rows if isinstance(rows, list) else []
    except Exception:
        return []


def _attach_audio(row: dict, batch: list[dict], k: int) -> dict:
    idx = row.get("source_clip")
    clip = batch[idx] if isinstance(idx, int) and 0 <= idx < len(batch) else batch[k % len(batch)]
    row.setdefault("id", f"{clip['id']}_fc{k:03d}")
    row.setdefault("domain", clip["domain"])
    row.setdefault("lang", "hi")
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
        t = clip["transcript"] or "नमस्ते आप कैसे हैं"
        words = t.split()
        mid = max(1, len(words) // 2)
        row = blank_row(f"{clip['id']}_fc{k:03d}", clip["domain"], s, t)
        # a short prior-turn context that makes the scenario natural
        ctx_by_scen = {
            "clean_end_of_speech": ["Agent: आपकी क्या समस्या है?"],
            "backchannel": ["Agent: अपनी बात बताइए"],
            "thinking_sound": ["Agent: ठीक है, बताइए"],
            "midsentence_pause": ["Agent: हाँ जी बोलिए"],
            "false_trigger_trap": ["Agent: कृपया बताइए"],
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
            surf = ["ओह", "अरे", "अच्छा", "आह"][k % 4]
            row["context"] = ["Agent: अच्छा, फिर क्या हुआ?"]
            tl += [{"kind": "pause", "t_s": 1.0, "dur_s": 0.3, "flag": "<BC>", "surface": surf},
                   {"kind": "speech", "t_s": 1.4, "text": " ".join(words[mid:])}]
        elif s == "thinking_sound":
            tl += [{"kind": "speech", "t_s": 1.0, "text": " ".join(words[mid:])},
                   {"kind": "pause", "t_s": 2.4, "dur_s": 0.5, "flag": "<EOS_SPEECH>"},
                   {"kind": "pause", "t_s": 3.0, "dur_s": 0.7, "flag": "<THINK>", "surface": "हम्म"}]
        elif s == "sustained_silence":
            tl = [{"kind": "speech", "t_s": 0.1, "text": t},
                  {"kind": "pause", "t_s": 2.0, "dur_s": 1.5, "flag": "<SILENCE>"}]
        else:  # nothing_happens / false_trigger_trap / barge_in -> no flag
            if s == "false_trigger_trap":
                tl = [{"kind": "speech", "t_s": 0.1, "text": "उम्म"},
                      {"kind": "pause", "t_s": 0.6, "dur_s": 0.4},
                      {"kind": "speech", "t_s": 1.0, "text": t}]
            else:
                tl += [{"kind": "speech", "t_s": 1.2, "text": " ".join(words[mid:])}]
        row["timeline"] = tl
        out.append(_attach_audio(row, batch, k))
    return out


# ---------------------------------------------------------------- orchestration
def _one_hit(batch, n_rows, cfg, mock) -> list[dict]:
    if mock:
        rows = _mock_rows(batch, n_rows)
    else:
        scen = sample_scenarios(n_rows)
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_user_prompt(batch, n_rows, scen)}]
        for attempt in range(3):
            try:
                rows = parse_rows(call_llm(msgs, cfg))
                if rows:
                    break
            except Exception as e:
                log.warning("LLM hit failed (attempt %d): %s", attempt + 1, e)
                time.sleep(2 * (attempt + 1))
        else:
            return []
        rows = [_attach_audio(r, batch, k) for k, r in enumerate(rows)]
    valid = []
    for r in rows:
        try:
            valid.append(validate_row(r))
        except SchemaError as e:
            log.debug("dropped invalid row: %s", e)
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
    return out


def _batch_id(batch: list[dict]) -> str:
    return "batch_" + "_".join(c["id"] for c in batch)[:80]
