"""Floor-control data generation.

An LLM agent turns real ASR clips (audio + transcript) into structured, audio-aware
floor-control training rows. The audio is *read* (VAD/energy probe) to extract pauses,
durations, speech-rate, trailing silence — serialized to a compact text "audio card" —
then passed to the agent, which returns 20-25 JSON rows spanning scenarios with the
flags placed correctly, following a controlled distribution (mostly nothing-happens,
rare signals, plus explicit false-trigger traps).

Modules:
  schema        canonical FC row format + validation + target rendering
  audio_probe   read a wav -> timing/energy features -> LLM "audio card"
  scenarios     scenario catalogue, rules, and the target distribution/sampler
  agent         async LLM client (concurrency + tqdm), prompt, parse, validate, balance
"""
