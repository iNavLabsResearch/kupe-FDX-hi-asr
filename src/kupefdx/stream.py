"""Streaming session — the almost-full-duplex front end.

The AUTHORITATIVE transcript is `corrected_transcript` — produced by NANDI. The CTC head
does not produce the transcript; `raw_ctc_transcript` is an optional, non-authoritative
low-latency DRAFT and is OFF by default (`emit_ctc_draft=False`). CTC blank-runs remain
available purely as an end-of-speech/silence TIMING signal.

Emits ONE record per audio chunk, in the exact interleaved-token spec:

  {"chunk_id","timestamp_ms","raw_ctc_transcript","corrected_transcript",
   "backchannel","thinking_sound","eos_flag","silence_flag"}

Stability contract (the whole point): the per-chunk floor-control head defaults to
NOTHING, and `StreamDecider` only lets a signal through when it clearly beats NOTHING
and passes anti-chatter hysteresis. So on an ordinary chunk EVERY signal field is
null/false and only the transcript advances — the model stays quiet unless a signal is
genuinely due.

Cost control: the CTC fast path runs every chunk (cheap, frame-synchronous); the Nandi
AR correction is throttled (recomputed at end-of-turn and every `correct_every` chunks),
so `corrected_transcript` is stable rather than re-decoded and flickering each chunk.
"""
from __future__ import annotations

import numpy as np
import torch

from .constants import (FC_BACKCHANNEL, FC_EOS_SPEECH, FC_SILENCE, FC_THINK,
                        SAMPLE_RATE)
from .floorcontrol import (CLASS_OF_FLAG, DEFAULT_SURFACE, StreamControls,
                          StreamDecider)


class StreamingSession:
    def __init__(self, model, chunk_ms: int | None = None, correct_every: int = 8,
                 max_buffer_s: float = 30.0, controls: "StreamControls | None" = None,
                 emit_ctc_draft: bool | None = None):
        self.model = model.eval()
        self.controls = controls or StreamControls()
        # chunk size + CTC-draft come from the low-latency profile unless overridden.
        chunk_ms = chunk_ms if chunk_ms is not None else self.controls.chunk_ms
        self.chunk = int(SAMPLE_RATE * chunk_ms / 1000)
        self.correct_every = int(correct_every)
        self.emit_ctc_draft = self.controls.emit_ctc_draft if emit_ctc_draft is None else bool(emit_ctc_draft)
        self.max_buf = int(SAMPLE_RATE * max_buffer_s)
        self.decider = StreamDecider(self.controls)
        self.buf = np.zeros(0, dtype=np.float32)
        self.n_samples = 0
        self.chunk_id = -1
        self.last_corrected = ""

    def reset(self):
        self.buf = np.zeros(0, dtype=np.float32)
        self.n_samples = 0
        self.chunk_id = -1
        self.last_corrected = ""
        self.decider.reset()

    def _blank_record(self):
        return {"chunk_id": self.chunk_id,
                "timestamp_ms": int(1000 * self.n_samples / SAMPLE_RATE),
                "raw_ctc_transcript": "", "corrected_transcript": self.last_corrected,
                "backchannel": None, "thinking_sound": None,
                "eos_flag": False, "silence_flag": False}

    @torch.no_grad()
    def _surface(self, flag, wave_t, wl):
        """The head decides WHEN; Nandi decides WHICH word. Extract the token(s) Nandi
        emits right after the flag (a context-appropriate ack/expression, e.g. हाहाहा/हम्म);
        fall back to a default surface if AR gives nothing."""
        fid = self.model.special_ids.get(flag)
        try:
            gen = self.model.generate(wave=wave_t, wave_len=wl, max_new_tokens=48)[0]
            if fid in gen:
                specials = set(self.model.special_ids.values())
                surf_ids = []
                for t in gen[gen.index(fid) + 1:]:
                    if t in specials:
                        break
                    surf_ids.append(t)
                surf = self.model.tok.decode(surf_ids, skip_special_tokens=True).strip()
                if surf:
                    return surf.split()[0] if flag == FC_BACKCHANNEL else surf
        except Exception:
            pass
        return DEFAULT_SURFACE[flag]

    @torch.no_grad()
    def _correct(self, wave_t, wl):
        gen = self.model.generate(wave=wave_t, wave_len=wl, max_new_tokens=128)
        # decode transcript only (strip inline flag/special tokens)
        return self.model.transcribe(gen)[0]

    @torch.no_grad()
    def push(self, samples: np.ndarray) -> dict:
        self.chunk_id += 1
        self.n_samples += len(samples)
        chunk_np = np.asarray(samples, np.float32)
        # Speech-resume guard: if we already fired end-of-speech but this chunk clearly has
        # speech energy, unlatch and keep transcribing — a premature predictive EOS self-heals,
        # so the utterance tail is never dropped.
        rms_db = 10 * np.log10(np.mean(chunk_np ** 2) + 1e-9) if len(chunk_np) else -99.0
        if self.decider.eos_latched and rms_db > -35.0:
            self.decider.unlatch()
        self.buf = np.concatenate([self.buf, chunk_np])[-self.max_buf:]
        wave_t = torch.from_numpy(self.buf)[None].to(self.model.device)
        wl = torch.tensor([self.buf.shape[0]], device=self.model.device)

        rec = self._blank_record()
        # CTC does NOT produce the transcript. Optional non-authoritative draft only.
        if self.emit_ctc_draft:
            rec["raw_ctc_transcript"] = self.model.ctc_transcribe(wave=wave_t, wave_len=wl)[0]

        # per-chunk floor-control decision on the MOST RECENT chunk (stable default NOTHING).
        # LOGITS in -> decider applies temperature/bias/threshold/hysteresis (controllable).
        logits = self.model.chunk_logits(wave=wave_t, wave_len=wl)     # [n_ch, C]
        cls = self.decider.decide(logits[-1], self.chunk_id)
        if cls == CLASS_OF_FLAG[FC_BACKCHANNEL]:
            rec["backchannel"] = self._surface(FC_BACKCHANNEL, wave_t, wl)
        elif cls == CLASS_OF_FLAG[FC_THINK]:
            rec["thinking_sound"] = self._surface(FC_THINK, wave_t, wl)
        elif cls == CLASS_OF_FLAG[FC_EOS_SPEECH]:
            rec["eos_flag"] = True
        elif cls == CLASS_OF_FLAG[FC_SILENCE]:
            rec["silence_flag"] = True

        # throttled AR correction: refresh at end-of-turn or every correct_every chunks
        if cls == 3 or (self.chunk_id % self.correct_every == 0):
            self.last_corrected = self._correct(wave_t, wl)
        rec["corrected_transcript"] = self.last_corrected
        return rec

    @torch.no_grad()
    def run_file(self, wav: np.ndarray) -> list[dict]:
        """Convenience: stream a whole clip, return the per-chunk record list."""
        self.reset()
        out = []
        for i in range(0, len(wav), self.chunk):
            out.append(self.push(wav[i:i + self.chunk]))
        return out
