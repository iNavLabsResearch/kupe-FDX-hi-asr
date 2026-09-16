"""Scenario catalogue, generation rules, and the target distribution.

The distribution is the anti-over-fire contract: the model sees mostly nothing-happens
and explicit traps (pauses that must NOT fire), with rare, well-placed signal events.
"""
from __future__ import annotations

import random

# scenario -> (weight, one-line intent). Weights are DEFAULTS; override via config
# (fc.distribution). Rebalanced so the agent backchannels realistically often (humans do),
# while ~45% of rows still carry NO flag so firing stays APPROPRIATE, not compulsive.
DEFAULT_WEIGHTS = {
    "nothing_happens":     20,
    "clean_end_of_speech": 15,
    "midsentence_pause":   12,
    "backchannel":         18,
    "expression":          8,
    "thinking_sound":      8,
    "sustained_silence":   6,
    "false_trigger_trap":  8,
    "barge_in":            5,
}
_INTENT = {
    "nothing_happens":     "ordinary speech, no control token at all (pure ASR)",
    "clean_end_of_speech": "user clearly finishes a COMPLETE thought -> <EOS_SPEECH> at the end",
    "midsentence_pause":   "TRAP: a real pause mid-utterance that is NOT turn-end -> NO flag",
    "backchannel":         "long user turn; a natural micro-pause for a short <BC> ack (हाँ/अच्छा/हूँ/जी)",
    "expression":          "emotional reaction as <BC> with an expressive surface (हाहाहा/उफ़/ओह/वाह) — ONLY when context is funny/surprising/sad",
    "thinking_sound":      "AFTER the user turn ends; system composing -> <EOS_SPEECH> then <THINK> (हम्म/उम्म)",
    "sustained_silence":   "no/low speech for a stretch -> <SILENCE> (NOT end-of-speech)",
    "false_trigger_trap":  "filler/hesitation/breath (उम्म, आ...) -> NO flag despite a gap",
    "barge_in":            "user resumes right after a pause -> pause gets NO flag",
}
SCENARIOS = {k: (DEFAULT_WEIGHTS[k], _INTENT[k]) for k in DEFAULT_WEIGHTS}


def set_weights(weights: dict | None):
    """Override the scenario distribution from config (partial dicts allowed)."""
    global SCENARIOS
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: int(v) for k, v in weights.items() if k in w})
    SCENARIOS = {k: (w[k], _INTENT[k]) for k in w}

# hard rules injected verbatim into the prompt.
RULES = """\
FLAGS (mutually exclusive; at most ONE per pause; NEVER on a speech segment):
  <EOS_SPEECH>  user turn genuinely finished (semantic completeness + a real trailing pause).
  <BC>          short listener acknowledgment WHILE the user is still speaking, at a natural
                micro-pause; MUST be followed by a surface word (हाँ / अच्छा / हूँ / जी).
  <THINK>       filler while the SYSTEM composes a reply — only AFTER an <EOS_SPEECH>; MUST be
                followed by a surface sound (हम्म / अच्छा / देखिए).
  <SILENCE>     a sustained low-speech stretch that is NOT a turn boundary.
Absence of any flag == "nothing happens". Use it generously.

SURFACE INVENTORY (Devanagari; pick what fits the moment, add close variants as needed):
  acknowledge: हाँ, हूँ, जी, जी हाँ, अच्छा, ठीक, ठीक है, बिलकुल, सही, ओके
  emotional  : हाहाहा (laughter), हे हे, उफ़ (sigh/ughh), आह, ओह, अरे, अरे वाह, वाह, बाप रे, ओहो
  thinking   : हम्म, हम्म्म, उम्म, आह, देखिए, एक मिनट, ज़रा रुकिए, सोचने दीजिए

HARD CONSTRAINTS (semantic correctness — do NOT force a flag that the text/context does not justify):
- Every flag must MAKE SENSE for this transcript and context. If unsure, use NO flag. Never
  attach a flag that is semantically or contextually wrong for the words spoken.
- A pause alone is NOT end-of-speech. Mid-utterance pauses, hesitations, breaths and fillers
  get NO flag. Fire <EOS_SPEECH> ONLY when the utterance is a COMPLETE thought AND it is the
  LAST event of the turn.
- <BC> only mid-turn and only after some speech has occurred (never as the first segment);
  it must fit as a listener reaction to what was just said.
- Emotional expressions (हाहाहा, उफ़, ओह...) only when the context genuinely warrants it
  (something funny, surprising, or sad) — never randomly.
- <THINK> only AFTER an <EOS_SPEECH> in the same row (system is now composing a reply).
- Ground every row in the given audio card: place pauses at plausible times from its pause
  list; do not invent pauses that contradict it.
- DEVANAGARI ONLY for Hindi: write हिंदी, not "hindi"; कैसे हो, not "kaise ho". NEVER romanize
  Hindi words. Use the clip transcript AS GIVEN (it is already Devanagari) — do not rewrite or
  translate it; you only choose where flags go and the surface words. Surfaces in Devanagari.
  (English is allowed ONLY for genuine English tech/proper terms, e.g. BP, RAM, server.)
- Keep surface words short and natural.
- timeline t_s must be non-decreasing and within [0, duration]."""


def sample_scenarios(n: int, seed: int | None = None) -> list[str]:
    rng = random.Random(seed)
    names = list(SCENARIOS)
    weights = [SCENARIOS[k][0] for k in names]
    return rng.choices(names, weights=weights, k=n)


def distribution_table() -> list[tuple[str, float, str]]:
    tot = sum(w for w, _ in SCENARIOS.values())
    return [(k, round(100 * w / tot, 1), d) for k, (w, d) in SCENARIOS.items()]


def rebalance(rows: list[dict], tol: float = 1.6) -> list[dict]:
    """Cap over-represented scenarios so the realized mix tracks the target. Keeps at
    most tol x the target share of each scenario relative to the smallest represented."""
    from collections import Counter
    c = Counter(r.get("scenario", "?") for r in rows)
    tot = sum(w for w, _ in SCENARIOS.values())
    keep, seen = [], Counter()
    target = {k: (w / tot) for k, (w, _) in SCENARIOS.items()}
    n = len(rows)
    for r in rows:
        s = r.get("scenario", "nothing_happens")
        cap = max(1, int(tol * target.get(s, 0.1) * n))
        if seen[s] < cap:
            keep.append(r)
            seen[s] += 1
    return keep
