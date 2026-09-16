"""Read a wav and extract timing/energy features, then serialize a compact text
"audio card" the LLM can reason over. Energy-based VAD by default (no extra deps);
Silero is used if installed (better boundaries).

The card is deliberately textual and small — the agent needs pause structure and
durations, not the waveform.
"""
from __future__ import annotations

import numpy as np

from ..audio import load_wav
from ..constants import SAMPLE_RATE


def _energy_vad(wav: np.ndarray, sr: int, frame_ms=25, hop_ms=10,
                thresh_db=-35.0, min_pause_ms=200):
    hop = int(sr * hop_ms / 1000)
    win = int(sr * frame_ms / 1000)
    n = 1 + max(0, (len(wav) - win) // hop)
    energies = np.empty(n, dtype=np.float32)
    for i in range(n):
        seg = wav[i * hop: i * hop + win]
        energies[i] = 10 * np.log10(np.mean(seg ** 2) + 1e-9)
    speech = energies > thresh_db
    # collapse to pause intervals (runs of non-speech), keep those >= min_pause
    pauses, i = [], 0
    min_frames = int(min_pause_ms / hop_ms)
    while i < n:
        if not speech[i]:
            j = i
            while j < n and not speech[j]:
                j += 1
            if (j - i) >= min_frames:
                pauses.append((i * hop_ms / 1000, j * hop_ms / 1000))
            i = j
        else:
            i += 1
    return energies, speech, pauses


def probe(path: str, sr: int = SAMPLE_RATE) -> dict:
    wav = load_wav(path, sr)
    dur = len(wav) / sr
    energies, speech, pauses = _energy_vad(wav, sr)
    speech_frac = float(speech.mean()) if len(speech) else 0.0
    lead = pauses[0][1] if pauses and pauses[0][0] <= 0.02 else 0.0
    trail = (dur - pauses[-1][0]) if pauses and pauses[-1][1] >= dur - 0.05 else 0.0
    return {
        "duration_s": round(dur, 2),
        "sample_rate": sr,
        "n_pauses": len(pauses),
        "pauses": [{"start_s": round(a, 2), "end_s": round(b, 2), "dur_s": round(b - a, 2)}
                   for a, b in pauses],
        "leading_silence_s": round(float(lead), 2),
        "trailing_silence_s": round(float(trail), 2),
        "speech_fraction": round(speech_frac, 2),
        "mean_energy_db": round(float(energies.mean()) if len(energies) else -99.0, 1),
    }


def speech_rate_wps(features: dict, transcript: str) -> float:
    speech_s = features["duration_s"] * max(features.get("speech_fraction", 1.0), 1e-3)
    return round(len(transcript.split()) / max(speech_s, 1e-3), 2)


def audio_card(features: dict, transcript: str) -> str:
    """Compact textual description for the LLM prompt."""
    ps = "; ".join(f"{p['start_s']}-{p['end_s']}s ({p['dur_s']}s)" for p in features["pauses"]) or "none"
    return (
        f"- duration: {features['duration_s']}s @ {features['sample_rate']}Hz\n"
        f"- transcript: \"{transcript}\"\n"
        f"- internal pauses (>=0.2s): {ps}\n"
        f"- leading silence: {features['leading_silence_s']}s; "
        f"trailing silence: {features['trailing_silence_s']}s\n"
        f"- speech fraction: {features['speech_fraction']}; "
        f"est. speech rate: {speech_rate_wps(features, transcript)} words/s\n"
        f"- mean energy: {features['mean_energy_db']} dB"
    )
