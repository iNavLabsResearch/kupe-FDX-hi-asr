"""Waveform IO + resample to 16 kHz mono. No torchaudio dependency (absent on the
Mac dev box): read with soundfile, resample with a light polyphase/linear path, or
librosa if present. Returns float32 numpy in [-1, 1]."""
from __future__ import annotations

import numpy as np

from .constants import SAMPLE_RATE


def _resample(wav: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return wav
    try:
        import librosa
        return librosa.resample(wav, orig_sr=sr_in, target_sr=sr_out).astype(np.float32)
    except Exception:
        # linear-interp fallback: fine for a smoke test / low-fidelity path.
        n_out = int(round(len(wav) * sr_out / sr_in))
        x_old = np.linspace(0.0, 1.0, num=len(wav), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        return np.interp(x_new, x_old, wav).astype(np.float32)


def load_wav(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    import soundfile as sf
    wav, sr_in = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:                       # to mono
        wav = wav.mean(axis=1)
    return _resample(wav.astype(np.float32), sr_in, sr)


def save_wav(path: str, wav: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    import soundfile as sf
    sf.write(path, np.asarray(wav, dtype=np.float32), sr)


def duration_s(path: str) -> float:
    import soundfile as sf
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)


def synth_speechish(seconds: float, sr: int = SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    """Deterministic pseudo-speech (formant-ish tones + noise) for the smoke test.
    Not real speech — just gives the pipeline a plausible waveform to chew on."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    wav = np.zeros(n, dtype=np.float32)
    for f0 in (rng.uniform(90, 160), rng.uniform(300, 700), rng.uniform(1200, 2400)):
        wav += (0.3 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)
    env = (0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 6) * t)).astype(np.float32)
    wav = wav * env + 0.02 * rng.standard_normal(n).astype(np.float32)
    return (0.8 * wav / (np.abs(wav).max() + 1e-6)).astype(np.float32)
