"""Canonical floor-control training-row schema + validation + target rendering.

One row is one audio-grounded example. The trainer reads `audio` + `target_sequence`;
everything else is provenance/analysis kept so the data is auditable and re-renderable.

Flags are MUTUALLY EXCLUSIVE per timeline point (a segment carries at most one). The
rendered `target_sequence` is what Nandi decodes:
  * speech segment            -> its text
  * pause with <BC>/<THINK>   -> "<FLAG> <surface>"   (teaches the actual word too)
  * pause with <EOS_SPEECH>/<SILENCE> -> "<FLAG>"
  * pause with no flag (a NOP / trap) -> NOTHING       (absence == nothing-happens)

So "nothing happens" is represented by the ABSENCE of a control token — which is why a
midsentence pause (trap) row has the pause in `audio_features` but no flag in the target.
That is the core defence against over-firing.
"""
from __future__ import annotations

from ..constants import FC_BACKCHANNEL, FC_EOS_SPEECH, FC_SILENCE, FC_THINK
from ..text import normalize

FLAGS = {FC_BACKCHANNEL, FC_THINK, FC_EOS_SPEECH, FC_SILENCE}
SURFACE_FLAGS = {FC_BACKCHANNEL, FC_THINK}      # these carry a spoken surface word
BARE_FLAGS = {FC_EOS_SPEECH, FC_SILENCE}

REQUIRED = ("id", "lang", "domain", "scenario", "transcript", "timeline")


def render_target(timeline: list[dict]) -> str:
    parts: list[str] = []
    for seg in timeline:
        kind = seg.get("kind")
        flag = seg.get("flag")
        if kind == "speech":
            parts.append(normalize(seg.get("text", "")))
        elif kind == "pause" and flag in SURFACE_FLAGS:
            surf = normalize(seg.get("surface", ""))
            parts.append(f"{flag} {surf}".strip())
        elif kind == "pause" and flag in BARE_FLAGS:
            parts.append(flag)
        # pause with no flag (NOP / trap) -> emit nothing
    # NOTE: join WITHOUT a final normalize() — that would strip the < > of the flag tokens.
    # Each text/surface piece is already normalized above; flags are kept verbatim.
    return " ".join(p for p in parts if p).strip()


class SchemaError(ValueError):
    pass


# flag aliases the LLM might emit ("BC", "eos", "backchannel", ...) -> canonical token
_FLAG_CANON = {
    "bc": FC_BACKCHANNEL, "backchannel": FC_BACKCHANNEL,
    "think": FC_THINK, "thinking": FC_THINK,
    "eos": FC_EOS_SPEECH, "eos_speech": FC_EOS_SPEECH, "eosspeech": FC_EOS_SPEECH,
    "end_of_speech": FC_EOS_SPEECH,
    "silence": FC_SILENCE,
}


def _canon_flag(flag):
    if flag in (None, "", "none", "null", "nop", "<NOP>"):
        return None
    if flag in FLAGS:
        return flag
    key = str(flag).strip().strip("<>").lower()
    return _FLAG_CANON.get(key)          # None if unrecognizable -> treated as no-flag


def _coerce_seg(seg):
    """Make one timeline segment schema-shaped, tolerating LLM drift (missing `kind`,
    aliased flags, flag accidentally on a text segment). Returns None if unusable."""
    if not isinstance(seg, dict):
        return None
    seg = dict(seg)
    seg["flag"] = _canon_flag(seg.get("flag"))
    text = (seg.get("text") or "").strip()
    kind = seg.get("kind")
    if kind not in ("speech", "pause"):          # infer when missing/other
        if seg["flag"] or seg.get("surface") or ("dur_s" in seg) or ("dur" in seg):
            kind = "pause"
        elif text:
            kind = "speech"
        else:
            return None
    if kind == "speech" and seg["flag"]:         # a flag belongs to a pause, not speech
        seg["flag"] = None                       #   keep the words, drop the stray flag
    seg["kind"] = kind
    if seg["flag"] is None:
        seg.pop("flag", None)
    return seg


def validate_row(row: dict, *, fix: bool = True) -> dict:
    if not isinstance(row, dict):
        raise SchemaError("row is not an object")
    row.setdefault("lang", "en")
    if fix and isinstance(row.get("timeline"), list):
        row["timeline"] = [s for s in (_coerce_seg(x) for x in row["timeline"]) if s]
    for k in REQUIRED:
        if k not in row:
            raise SchemaError(f"missing key: {k}")
    if not isinstance(row["timeline"], list) or not row["timeline"]:
        raise SchemaError("timeline must be a non-empty list")
    seen_speech = False
    eos_at = None
    n_eos = 0
    for i, seg in enumerate(row["timeline"]):
        if seg.get("kind") not in ("speech", "pause"):
            raise SchemaError(f"seg {i}: kind must be speech|pause")
        flag = seg.get("flag")
        if flag is not None and flag not in FLAGS:
            raise SchemaError(f"seg {i}: unknown flag {flag!r}")
        if seg.get("kind") == "speech":
            if flag in FLAGS:
                raise SchemaError(f"seg {i}: speech segment must not carry a flag")
            seen_speech = True
            continue
        if flag in SURFACE_FLAGS and not normalize(seg.get("surface", "")):
            raise SchemaError(f"seg {i}: {flag} requires a non-empty surface word")
        # ---- semantic ordering (reject contextually-wrong placements) ----
        if flag == FC_BACKCHANNEL and not seen_speech:
            raise SchemaError("seg %d: <BC> before any speech" % i)
        if flag == FC_THINK and eos_at is None:
            raise SchemaError("seg %d: <THINK> without a preceding <EOS_SPEECH>" % i)
        if flag == FC_BACKCHANNEL and eos_at is not None:
            raise SchemaError("seg %d: <BC> after <EOS_SPEECH> (turn already ended)" % i)
        if flag == FC_EOS_SPEECH:
            n_eos += 1
            eos_at = i
            if n_eos > 1:
                raise SchemaError("more than one <EOS_SPEECH>")
    # no speech after end-of-speech (only <THINK> may follow)
    if eos_at is not None:
        for seg in row["timeline"][eos_at + 1:]:
            if seg.get("kind") == "speech":
                raise SchemaError("speech after <EOS_SPEECH>")
    tgt = render_target(row["timeline"])
    if not tgt:
        raise SchemaError("empty target_sequence after render")
    if fix or "target_sequence" not in row:
        row["target_sequence"] = tgt
    # flags actually present, for balancing / metrics
    row["flags"] = sorted({s["flag"] for s in row["timeline"] if s.get("flag") in FLAGS})
    row["transcript"] = normalize(row["transcript"])
    _reconcile_scenario(row)                       # relabel so scenario ALWAYS matches the flags
    return row


_NOFLAG_SCEN = {"nothing_happens", "midsentence_pause", "false_trigger_trap", "barge_in"}
_EMO = ("haha", "oh no", "ugh", "wow", "whoa", "yikes", "phew", "aw", "hahaha")


def _reconcile_scenario(row: dict) -> None:
    """gpt-luna sometimes mislabels (e.g. 'clean_end_of_speech' with no <EOS_SPEECH> placed).
    Set the scenario from the flags actually present, so the label never lies and the realized
    distribution is honest."""
    has = set(row.get("flags", []))
    if FC_THINK in has:
        s = "thinking_sound"
    elif FC_EOS_SPEECH in has:
        s = "clean_end_of_speech"
    elif FC_BACKCHANNEL in has:
        surf = " ".join(sg.get("surface", "") for sg in row["timeline"]
                        if sg.get("flag") == FC_BACKCHANNEL).lower()
        s = "expression" if any(w in surf for w in _EMO) else "backchannel"
    elif FC_SILENCE in has:
        s = "sustained_silence"
    else:                                          # no flag: keep a specific no-flag label else default
        s = row.get("scenario") if row.get("scenario") in _NOFLAG_SCEN else "nothing_happens"
    row["scenario"] = s


def blank_row(rid: str, domain: str, scenario: str, transcript: str) -> dict:
    return {"id": rid, "lang": "hi", "domain": domain, "scenario": scenario,
            "transcript": normalize(transcript), "timeline": [], "provenance": "synthetic"}
