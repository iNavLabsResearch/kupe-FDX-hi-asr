# KupeFDX-hi-asr

Streaming Hindi ASR + almost-full-duplex floor control, fusing **omniASR_W2V** (raw SSL
wav2vec2 encoder — no baked-in vocab) with **Nandi-Mini-150M** (Hindi/Devanagari SLM).
Target: **< 5% streaming Hindi WER**, plus live domain-term correction and floor-control
signals (backchannel / thinking-sound / end-of-speech / silence).

See **[PLAN.md](PLAN.md)** for the full architecture, data budget, phases, and rationale.

## Quick start

```bash
pip install -r requirements.txt
# 1) prove the whole pipeline on this machine (CPU/MPS, synthetic data, seconds):
bash run_smoke.sh
```

The smoke test runs prep → encode → k-means quantizer → discrete codes → train
(Phase 1 CTC + Phase 3 joint) → resume-from-checkpoint → eval (WER/CER/CTC/FC) →
AR inference → streaming, using `TinyEncoder` + `TinyNandi` (a faithful factorized/tied/
layer-sharing mirror of Nandi). It needs no network and no tokens.

## On the H100 (real models)

```bash
# .env: HF_TOKEN=hf_...   WANDB_API_KEY=...   HF_OWNER=FrontiersMind
python scripts/01_dataprep.py --config configs/gpu.yaml --hf fleurs_hi   # + more sources
python scripts/02_encode.py   --config configs/gpu.yaml --feats --fit-quantizer --codes
python scripts/hf_sync.py push --config configs/gpu.yaml --what manifests
python scripts/03_train.py    --config configs/gpu.yaml --phase 1               # CTC encoder
python scripts/03_train.py    --config configs/gpu.yaml --phase 2               # align projector+Nandi
python scripts/03_train.py    --config configs/gpu.yaml --phase 3 --resume auto # joint — <5% gate
# floor-control data (audio-aware LLM agent) then Phase 4 — see DATA.md
python scripts/06_gen_fc.py   --config configs/gpu.yaml --src data/manifests/train.jsonl \
                              --out data/manifests/fc.jsonl --concurrency 10 --rows-per-hit 22
python scripts/03_train.py    --config configs/gpu.yaml --phase 4 --set data.manifest=data/manifests/fc.jsonl
python scripts/03_train.py    --config configs/gpu.yaml --phase 5               # domain correction
python scripts/04_eval.py     --config configs/gpu.yaml --ckpt <ckpt> --split test
python scripts/05_infer.py    --config configs/gpu.yaml --ckpt <ckpt> --wav clip.wav --stream
```

Same code as the smoke run — only `backend: real`, data scale, and sizes change
(`configs/gpu.yaml`). Every stage is resumable (shard ledgers) and mirrors state to the
Hub (`hf_sync.py`). Confirm the two spikes in PLAN §9 (exact omniASR_W2V id/API; Nandi
factorized-embedding extension against the real model) on the box before the long run.

## Layout
```
src/kupefdx/   config env ledger  encoders ctc_head frontend quantizer  decoders tokens
               model dataset collate metrics  train evaluate stream  smoke
   fcgen/      schema audio_probe scenarios agent   (floor-control data generation)
scripts/       00_smoke 01_dataprep 02_encode 03_train 04_eval 05_infer 06_gen_fc hf_sync
configs/       smoke.yaml (tiny/Mac)   gpu.yaml (H100/real)
tests/         test_token_extension.py  test_causal_mask.py
samples/       fc_samples.jsonl        (one real generated row per scenario)
```

Docs: **[PLAN.md](PLAN.md)** (architecture, phases) · **[DATA.md](DATA.md)** (floor-control
data format, scenarios, distribution, the generation agent).
