"""Read a wav and extract timing/energy features, then serialize a compact text
"audio card" the LLM can reason over. Energy-based VAD by default (no extra deps);
Silero is used if installed (better boundaries).

The card is deliberately textual and small — the agent needs pause structure and
durations, not the waveform.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..constants import SAMPLE_RATE


def _read_native(path: str) -> tuple[np.ndarray, int]:
    """Decode a wav to mono float32 at its NATIVE sample rate (no resample).
    The VAD works in ms, so the sample rate is irrelevant to pause/timing — skipping
    the 16 kHz resample removes the biggest per-clip cost."""
    import soundfile as sf
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    return wav.astype(np.float32, copy=False), int(sr)


def _energy_vad(wav: np.ndarray, sr: int, frame_ms=25, hop_ms=10,
                thresh_db=-35.0, min_pause_ms=200):
    hop = max(1, int(sr * hop_ms / 1000))
    win = max(1, int(sr * frame_ms / 1000))
    if len(wav) < win:
        return np.zeros(0, np.float32), np.zeros(0, bool), []
    # vectorized framing: strided windows [n, win], energy in one numpy pass
    frames = sliding_window_view(wav, win)[::hop]
    energies = (10 * np.log10(np.mean(frames.astype(np.float32) ** 2, axis=1) + 1e-9)
                ).astype(np.float32)
    n = len(energies)
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
    wav, sr = _read_native(path)            # native sr; VAD is sr-agnostic (works in ms)
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
