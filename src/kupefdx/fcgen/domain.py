"""Domain-correction data — the Phase-5 task: given audio + domain tag + prior turns,
output the CORRECTED transcript (domain terms fixed, ASR/spelling errors fixed).

Training-record format (as specified):
  {domain, omni_raw_transcript, context, corrected_transcript, correction_spans,
   chunk_boundaries_ms}

Converted to a KupeFDX training row:
  text            = omni_raw_transcript   (CTC / raw-acoustic target + WER-raw ref)
  target_sequence = corrected_transcript  (what Nandi decodes, conditioned on context+domain)
  context, correction_spans, chunk_boundaries_ms kept as provenance.

So the split is clean: CTC learns what the acoustics literally say; Nandi learns to
correct it using domain + conversation context.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..env import log
from ..text import normalize
from .agent import _llm_cfg, call_llm, tqdm

# English term maps for the offline mock (raw ASR term -> corrected domain term).
# Real coverage comes from the LLM.
MOCK_TERMS = {
    "medical": [("high blood pressure", "hypertension"), ("sugar", "blood glucose"),
                ("diabetees", "diabetes"), ("heart beating fast", "tachycardia")],
    "technical": [("server went down", "server is down"), ("send the data", "transfer the data"),
                  ("ram memory", "RAM")],
    "banking": [("send money", "transfer funds"), ("account no", "account number")],
    "general": [("okay so", "okay")],
}

SYSTEM = ("You correct English ASR transcripts for a given domain. Fix domain terminology and "
          "obvious ASR errors using the conversation context. Output ONLY a JSON array of "
          "records. Never change meaning.")


def validate_record(rec: dict) -> dict:
    for k in ("domain", "omni_raw_transcript", "corrected_transcript"):
        if k not in rec:
            raise ValueError(f"missing {k}")
    rec.setdefault("context", [])
    rec.setdefault("correction_spans", [])
    rec.setdefault("chunk_boundaries_ms", [])
    rec["omni_raw_transcript"] = normalize(rec["omni_raw_transcript"])
    rec["corrected_transcript"] = normalize(rec["corrected_transcript"])
    return rec


def to_training_row(rec: dict, audio: str, rid: str) -> dict:
    rec = validate_record(rec)
    return {"id": rid, "lang": "hi", "domain": rec["domain"], "scenario": "domain_correction",
            "audio": audio, "text": rec["omni_raw_transcript"],
            "target_sequence": rec["corrected_transcript"], "context": rec.get("context", []),
            "correction_spans": rec.get("correction_spans", []),
            "chunk_boundaries_ms": rec.get("chunk_boundaries_ms", []),
            "timeline": [], "provenance": rec.get("provenance", "domain-llm")}


# ---------------------------------------------------------------- mock
def _mock_record(clip: dict) -> dict:
    dom = clip["domain"] if clip["domain"] in MOCK_TERMS else "general"
    raw = clip["transcript"]
    terms = MOCK_TERMS[dom]
    a, b = terms[hash(clip["id"]) % len(terms)]
    corrupted = (raw + " " + a).strip()
    corrected = (raw + " " + b).strip()
    return {"domain": dom, "omni_raw_transcript": corrupted, "context": [],
            "corrected_transcript": corrected,
            "correction_spans": [{"raw": a, "fixed": b, "reason": "domain term"}],
            "chunk_boundaries_ms": clip.get("chunk_boundaries_ms", []),
            "provenance": "domain-mock"}


def _parse(text):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    try:
        return json.loads(m.group(0)) if m else []
    except Exception:
        return []


def _one_hit(batch, cfg, mock):
    if mock:
        recs = [_mock_record(c) for c in batch]
    else:
        cards = "\n".join(f"- clip {i} (id={c['id']}, domain={c['domain']}): "
                          f"\"{c['transcript']}\"" for i, c in enumerate(batch))
        user = (f"For each clip, produce a correction record. Inject a realistic domain "
                f"ASR error into omni_raw_transcript and give the corrected_transcript, with "
                f"correction_spans (raw/fixed/reason). Clips:\n{cards}\n"
                f'Schema: {{"domain","omni_raw_transcript","context":[],'
                f'"corrected_transcript","correction_spans":[{{"raw","fixed","reason"}}]}}. '
                f"Return ONLY the JSON array.")
        try:
            recs = _parse(call_llm([{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": user}], cfg))
        except Exception as e:
            log.warning("domain hit failed: %s", e)
            recs = []
    out = []
    for i, r in enumerate(recs):
        clip = batch[i % len(batch)]
        try:
            out.append(to_training_row(r, clip["audio"], f"{clip['id']}_dom{i:02d}"))
        except Exception as e:
            log.debug("dropped domain rec: %s", e)
    return out


def generate_domain(clips, *, clips_per_hit=6, concurrency=10, mock=False,
                    seen_ledger=None, push_cb=None):
    cfg = _llm_cfg()
    if not mock and not cfg["api_key"]:
        raise RuntimeError("set KUPE_LLM_API_KEY or use --mock")
    batches = [clips[i:i + clips_per_hit] for i in range(0, len(clips), clips_per_hit)]
    out = []
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {ex.submit(_one_hit, b, cfg, mock): b for b in batches}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="domain-gen"):
            rows = fut.result()
            out += rows
            if push_cb:
                push_cb(rows)
    return out
