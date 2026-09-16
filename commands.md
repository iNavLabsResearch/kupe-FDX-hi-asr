# KupeFDX-hi-asr — Complete Commands (clone → data → train → eval → infer)

Everything, in order. Every stage is resumable and Hub-synced. Configs: `configs/smoke.yaml`
(tiny, CPU/MPS, for verification) and `configs/gpu.yaml` (H100, real models).

---

## 0. Get the code

```bash
git clone https://github.com/iNavLabsResearch/kupe-FDX-hi-asr.git
cd kupe-FDX-hi-asr
```
HF repos live under **`anuj-inavlabs/`** (set in `configs/*.yaml` → `owner`):
`anuj-inavlabs/kupe-hi-asr-data` (data), `anuj-inavlabs/KupeFDX-hi-asr-runs` (runs),
`anuj-inavlabs/KupeFDX-hi-asr` (model).

## 1. Environment setup

```bash
# --- option A: conda (recommended on the H100 box) ---
conda create -n kupefdx python=3.11 -y
conda activate kupefdx

# --- option B: venv ---
python3 -m venv .venv && source .venv/bin/activate

# core deps
pip install -r requirements.txt

# GPU box only — real encoder + Nandi + NeMo (FastConformer fallback) + audio:
pip install "transformers>=4.45" accelerate safetensors datasets soundfile librosa wandb python-dotenv
pip install "nemo_toolkit[asr]"          # only if using the FastConformer fallback encoder
# omniASR SSL encoder: install Meta's package on the box, then set encoder_id in configs/gpu.yaml
# pip install omnilingual-asr            # (confirm exact package/id — PLAN §9.1)
```

## 2. Secrets

```bash
cp .env.example .env
#   edit .env and fill:
#   HF_TOKEN=hf_...            HF_OWNER=FrontiersMind
#   WANDB_API_KEY=...          WANDB_PROJECT=kupe-fdx-hi-asr
#   KUPE_LLM_BASE_URL=https://api.openai.com/v1
#   KUPE_LLM_API_KEY=sk-...    KUPE_LLM_MODEL=gpt-5.6-luna
```

## 3. Verify everything on this machine FIRST (no GPU, no keys, seconds)

```bash
bash run_smoke.sh
# runs: unit tests + full pipeline (prep→encode→quantize→train P1+P3→resume→eval→FC-gen→P4→infer→stream)
# must print "✅ SMOKE PASS" before you spend GPU money.
```

## 4. Create the Hub repos (one-time)

```bash
python scripts/hf_sync.py push --config configs/gpu.yaml --what manifests   # creates data repo on first push
# (runs/model repos are auto-created by training when push_to_hub: true)
```

---

## 5. Data: SHARDED pipeline (download → encode → push raw+encoded → flush → next shard)

This is the correct disk-light flow: raw and encoded land on the Hub **continuously**, local
disk never fills, and it's resumable per shard. Target ≈**3,200 h** total (see §Hours below).

```bash
# one box, all shards (500 clips/shard):
python scripts/11_shard_pipeline.py --config configs/gpu.yaml --hf fleurs_hi        --shard-size 500
python scripts/11_shard_pipeline.py --config configs/gpu.yaml --hf common_voice_hi  --shard-size 500
python scripts/11_shard_pipeline.py --config configs/gpu.yaml --local-dir /data/hi_medical --domain medical --shard-size 500

# PARALLEL across 2 GPUs (interleaved shards run "meanwhile"):
CUDA_VISIBLE_DEVICES=0 python scripts/11_shard_pipeline.py --config configs/gpu.yaml --hf shrutilipi_hi --shard-size 500 --shard-start 0 --stride 2 &
CUDA_VISIBLE_DEVICES=1 python scripts/11_shard_pipeline.py --config configs/gpu.yaml --hf shrutilipi_hi --shard-size 500 --shard-start 1 --stride 2 &
wait
```
Each shard pushes `raw/<shard>/*.wav` + `encoded/<shard>/*.npy` + manifest to
`anuj-inavlabs/kupe-hi-asr-data`, records hours in the ledger, then flushes local files.
The ledger prints cumulative hours pushed so you can watch the total climb toward ~3,200 h.

> Prefer this over the two separate steps below. The old split flow (all raw → then encode)
> is still available if you ever want it: `scripts/01_dataprep.py` then `scripts/02_encode.py`.

## 6. Data hours (what we gather + push)

| Bucket | Hours | Used in |
|---|---|---|
| Core clean ASR | **2,500 h** | Stage A (encoder Hindi-adapt) + Stage B (Nandi transcription) |
| Domain packs (medical/technical/support/general) | **+400 h** | Stage B joint + Stage C domain correction |
| Floor-control (generated, §9) | **+300 h** | Stage C floor-control |
| **Total raw pushed to HF** | **≈ 3,200 h** | |
| Dev (held-out) | 15–20 h | early stopping |
| Blind test (clean + spontaneous + domain) | ~25 h | final WER |

Sources to fill 2,500 h: Shrutilipi (~6,400 h available) · IndicVoices · Kathbath · Vaani ·
Spring-INX · MUCS · Common Voice · FLEURS — select the cleanest ~2,500 h via the quality gate.

## 7. Check ASR data quality (per-flag/scenario/domain, Devanagari purity, dupes)

```bash
python scripts/09_data_quality.py --manifest data/manifests/train.jsonl
```

## 8. Tokenizer fertility (Nandi BPE on Hindi — sanity check)

```bash
python scripts/08_tokenizer_fertility.py --config configs/gpu.yaml
```

---

## 9. Generate floor-control data (audio-aware LLM agent, gpt-5.6-luna)

```bash
# offline dry-run first (no keys, proves the generator):
python scripts/06_gen_fc.py --config configs/gpu.yaml --src data/manifests/train.jsonl \
    --out data/manifests/fc.jsonl --mock --limit 200

# real generation: 10 concurrent, ~22 rows/hit, resumable per batch
python scripts/06_gen_fc.py --config configs/gpu.yaml --src data/manifests/train.jsonl \
    --out data/manifests/fc.jsonl --concurrency 10 --rows-per-hit 22 --push

# quality-gate the generated data (blocks training in CI if it fails)
python scripts/09_data_quality.py --manifest data/manifests/fc.jsonl
```

## 10. Generate domain-correction data (Phase 5)

```bash
python scripts/07_gen_domain.py --config configs/gpu.yaml --src data/manifests/train.jsonl \
    --out data/manifests/domain.jsonl --concurrency 10        # add --mock for offline
python scripts/09_data_quality.py --manifest data/manifests/domain.jsonl --kind domain
```

---

## 11. Training — 3 stages / 5 runs (resume any run with `--resume auto`)

```bash
# STAGE A — adapt the Omni SSL encoder on Hindi (our CTC head; NOT the transcript)
python scripts/03_train.py --config configs/gpu.yaml --phase 1

# STAGE B — teach Nandi to transcribe from the adapted encoder
python scripts/03_train.py --config configs/gpu.yaml --phase 2
python scripts/03_train.py --config configs/gpu.yaml --phase 3 --resume auto   # MAIN <5% WER gate

# STAGE C — floor-control, then domain correction
python scripts/03_train.py --config configs/gpu.yaml --phase 4 --set data.manifest=data/manifests/fc.jsonl
python scripts/03_train.py --config configs/gpu.yaml --phase 5 --set data.manifest=data/manifests/domain.jsonl

# push a finished run + its best model to the Hub
python scripts/hf_sync.py push --config configs/gpu.yaml --what run   --run <run_name>
python scripts/hf_sync.py push --config configs/gpu.yaml --what model --run <run_name>
```

Resume examples:
```bash
python scripts/03_train.py --config configs/gpu.yaml --phase 3 --resume auto           # latest ckpt of this phase
python scripts/03_train.py --config configs/gpu.yaml --phase 3 --resume <run_name>      # a specific run
```

## 12. Evaluation — WER/CER (Nandi), CTC-diagnostic WER, floor-control F1 + false-fire

```bash
python scripts/04_eval.py --config configs/gpu.yaml --ckpt checkpoints/<run>/checkpoint-XXXX --split test
python scripts/04_eval.py --config configs/gpu.yaml --ckpt checkpoints/<run>/checkpoint-XXXX --split val
```

## 13. Latency benchmark — back the <100 ms claim with p50/p95 numbers

```bash
python scripts/10_latency_bench.py --config configs/gpu.yaml --chunk-ms 80
```

## 14. Inference

```bash
# offline: authoritative transcript (Nandi) + CTC diagnostic
python scripts/05_infer.py --config configs/gpu.yaml --ckpt <ckpt> --wav clip.wav

# streaming: per-chunk records (corrected transcript = Nandi, + backchannel/thinking/eos/silence)
python scripts/05_infer.py --config configs/gpu.yaml --ckpt <ckpt> --wav clip.wav --stream

# dial floor-control at inference (no retraining):
python scripts/05_infer.py --config configs/gpu.yaml --ckpt <ckpt> --wav clip.wav --stream --bc-bias 1.0
python scripts/05_infer.py --config configs/gpu.yaml --ckpt <ckpt> --wav clip.wav --stream --temperature 0.7 --no-think
```

---

## 15. Cross-machine resume (pull state, continue)

```bash
python scripts/hf_sync.py pull --config configs/gpu.yaml --what manifests
python scripts/hf_sync.py pull --config configs/gpu.yaml --what feats
python scripts/03_train.py --config configs/gpu.yaml --phase 3 --resume auto
```

## 16. Tests + rebuild the proposal PDF

```bash
python tests/test_token_extension.py      # Nandi factorized/tied vocab extension
python tests/test_causal_mask.py          # streaming mask: no future leakage
python paper/make_figs.py && (cd paper && tectonic paper.tex)   # -> paper/paper.pdf
```

---

## One-shot: full local dry-run of the whole flow (offline, tiny models)

```bash
bash run_smoke.sh                                                   # end-to-end sanity
python scripts/08_tokenizer_fertility.py --config configs/smoke.yaml
python scripts/10_latency_bench.py       --config configs/smoke.yaml
```

## Key config knobs (edit configs/gpu.yaml)
- `backend: real` — use omniASR_W2V + Nandi (vs `tiny` for smoke).
- `base.encoder_type: omni_w2v` (SSL only) — or `fastconformer` fallback if WER stalls.
- `audio.n_codes` — 0 = continuous only (recommended); >0 = discrete audio tokens.
- `stream.chunk_ms: 80`, `stream.endpoint_mode: predictive`, `audio.right_chunks: 0` — latency.
- `train.eos_lead_ms: 160` — predictive end-of-turn labels.
- `data.min_hours` — refuses to launch the main run on too little data.
- `train.push_to_hub`, `train.push_checkpoints` — Hub sync during training.
