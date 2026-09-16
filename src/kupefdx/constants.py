"""Static contract shared by every stage. No config, no env, no side effects.

Phase 1 language scope is HINDI. The system is designed as an almost-full-duplex
floor controller (transcription + backchannel + thinking-sound + end-of-speech)
from the start, but validated on Hindi before scaling to the other Indic langs.
"""
from __future__ import annotations

# --------------------------------------------------------------------------
# Base models.
# --------------------------------------------------------------------------
NANDI_ID = "FrontiersMind/Nandi-Mini-150M"        # decoder SLM (trust_remote_code)
NANDI_HIDDEN = 832                                # verified from config.json 2026-09
NANDI_VOCAB = 131072                              # BPE, native Devanagari
NANDI_EMBED_RANK = 196                            # factorized embedding rank

# omniASR_W2V — Meta Omnilingual SSL wav2vec2 backbone (raw, no baked-in vocab).
# Exact HF id / package API confirmed on the GPU box (PLAN §9 item 1); the encoder
# wrapper discovers out_dim + frame_rate at runtime, never hardcodes them.
OMNI_W2V_ID = "facebook/omniASR_W2V_300M"         # placeholder id; overridable in config
SAMPLE_RATE = 16_000                              # all audio resampled to 16 kHz mono

# --------------------------------------------------------------------------
# Special tokens added to Nandi's vocab (see vocab.py for the full list).
# Floor-control signals are MUTUALLY EXCLUSIVE per emission point.
# --------------------------------------------------------------------------
TOK_AUDIO_BOS = "<audio>"
TOK_AUDIO_EOS = "</audio>"
TOK_HIST_BOS = "<hist>"
TOK_HIST_EOS = "</hist>"
# floor-control
FC_NOP = "<NOP>"            # default: nothing happens (the vast majority of frames)
FC_BACKCHANNEL = "<BC>"     # short listener ack while user still speaking + micro-pause
FC_THINK = "<THINK>"        # filler while system composes a long answer (post-turn only)
FC_EOS_SPEECH = "<EOS_SPEECH>"  # user turn genuinely finished (acoustic+semantic agree)
FC_SILENCE = "<SILENCE>"    # sustained silence, no turn boundary
FC_TOKENS = [FC_NOP, FC_BACKCHANNEL, FC_THINK, FC_EOS_SPEECH, FC_SILENCE]

# discrete audio-code token template (2048 codes by default) — "teach Nandi the
# omni encoder's audio tokens": these share Nandi's embedding table with text.
AUDIO_CODE_FMT = "<aud_{k}>"

# --------------------------------------------------------------------------
# Dataset config views (HF `name`) in the data repo.
# --------------------------------------------------------------------------
CONFIG_RAW = "raw"        # 16 kHz waveform bytes + text + fc labels — encode reads this
CONFIG_FEATS = "feats"    # cached omniASR_W2V features + text — train (frozen enc) reads this
CONFIG_TOKENS = "tokens"  # discrete audio-code ids + text — audio-token branch reads this

SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"

# --------------------------------------------------------------------------
# WHO PRODUCES THE TRANSCRIPT: Nandi (the SLM), always. The CTC head is NOT a
# transcript producer — it exists only to (a) fine-tune the Omni SSL encoder on
# Hindi (Stage A training objective) and (b) give end-of-speech / silence TIMING
# via blank-run detection. Omni's own CTC weights are never loaded.
# --------------------------------------------------------------------------

# Training is organised as THREE STAGES (the user-facing mental model), implemented
# as five resumable training runs (--phase). The stages map onto phases as below.
#
#   STAGE A — adapt the ears:  fine-tune the Omni SSL encoder on Hindi        -> phase 1
#   STAGE B — teach Nandi to transcribe from the adapted encoder's audio      -> phases 2, 3
#   STAGE C — combined floor-controller + correction + behaviours             -> phases 4, 5
#
PHASE_CTC = 1        # STAGE A: encoder + our CTC head (causal), Devanagari chars — ENCODER ADAPTATION
PHASE_ALIGN = 2      # STAGE B: projector + audio-token embeds + Nandi (encoder frozen)
PHASE_JOINT = 3      # STAGE B: everything, low-LR encoder — MAIN <5% WER gate (Nandi transcribes)
PHASE_FC = 4         # STAGE C: floor-control signals + heavy "nothing" negatives
PHASE_DOMAIN = 5     # STAGE C: domain-term correction

STAGE_OF_PHASE = {1: "A", 2: "B", 3: "B", 4: "C", 5: "C"}
STAGE_NAME = {"A": "adapt encoder on Hindi", "B": "teach Nandi to transcribe",
              "C": "floor-control + correction"}
