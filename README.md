# KupeFDX — English ASR + full-duplex floor control

Streaming English ASR + almost-full-duplex **floor control**, fusing a **FastConformer**
encoder (NVIDIA `stt_en_fastconformer_hybrid_large_pc`, encoder only) with **Nandi-Mini-150M**
as the decoder. Targets **~3% streaming WER**, plus live domain-term correction and
floor-control signals (backchannel `<BC>` · thinking `<THINK>` · end-of-speech `<EOS_SPEECH>`
· silence `<SILENCE>`).

Config: [`configs/en.yaml`](configs/en.yaml) (real run) · [`configs/smoke.yaml`](configs/smoke.yaml)
(tiny, CPU/MPS). See **[PLAN.md](PLAN.md)** for architecture, the 5 phases, and data budget.

---

## 0. Clone, environment, install

```bash
git clone https://github.com/iNavLabsResearch/kupe-FDX-hi-asr.git
cd kupe-FDX-hi-asr
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install hf_transfer                      # optional: faster Hub downloads

cp .env.example .env                         # then fill in the secrets below
set -a && . ./.env && set +a                 # export them into the shell
```

`.env` keys (`.env` is git-ignored — secrets never leave the box):

| key | what |
|---|---|
| `HF_TOKEN`, `HF_OWNER` | Hugging Face auth + `{owner}` in `configs/*.yaml` repos |
| `KUPE_LLM_BASE_URL` | LLM endpoint — default `https://cloud.olakrutrim.com/v1` (Krutrim) |
| `KUPE_LLM_API_KEY` | Krutrim API key |
| `KUPE_LLM_MODEL` | `gemma-4-31b-it` |
| `KUPE_LLM_MAX_OUT`, `KUPE_LLM_TEMP`, `KUPE_LLM_STREAM` | per-request output cap / temperature / SSE streaming (1) |
| `WANDB_API_KEY`, `WANDB_PROJECT` | optional; metrics still print if absent |

## 1. Smoke test — prove the whole pipeline (no GPU, no network)

```bash
python scripts/00_smoke.py --config configs/smoke.yaml
```

## 2. Pull the encoded data from the Hub

`feats.npz` (encoder features) is all that frozen-encoder training (phases 1, 2, 4, 5)
needs and is already on the Hub. Raw audio is only for phase 3 (unfrozen encoder).

```bash
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download anuj-inavlabs/kupe-en-asr-data \
    --repo-type dataset --local-dir data --include "encoded/**" "ledger/**"
python scripts/12_build_manifest.py --config configs/en.yaml --out data/manifests/train.jsonl
```
(To *gather* new raw data instead of pulling, see the `gather_all_en.sh` flow in
[`commands.md`](commands.md) §2.)

## 3. Generate LLM training data — ONE command (FC + domain)

`gen_data.py` produces **both** floor-control and domain-correction data, streams each
request's result in **color** (green OK / red FAIL) with live token/cost totals, writes
rows to disk as they arrive, and **auto-syncs to the Hub** every `--sync-every` requests.
Resumable per request (ledger), so re-running skips finished work.

```bash
# 3a. dry-run first (offline, no keys) — proves the generator end to end
python scripts/gen_data.py --config configs/en.yaml --src data/manifests/train.jsonl \
    --mock --limit 50 --no-push

# 3b. real run — both FC + domain, high concurrency, mirrors to the Hub as it goes
python scripts/gen_data.py --config configs/en.yaml --src data/manifests/train.jsonl \
    --concurrency 40 --sync-every 25

# watch ONE call's live SSE token stream (runs sequentially so it's readable)
python scripts/gen_data.py --config configs/en.yaml --src data/manifests/train.jsonl \
    --only fc --limit 3 --show-stream --no-push

# see rows land live in a second terminal (full concurrency, no interruption)
tail -f data/manifests/fc.jsonl
```

Outputs: `data/manifests/fc.jsonl` and `data/manifests/domain.jsonl`.
Useful flags: `--only fc|domain|both` · `--limit N` (cap clips) · `--rows-per-hit`
· `--clips-per-hit` · `--concurrency` · `--sync-every N` (0 = only at end) · `--no-push`.

## 4. Readiness gate — ONE command (data + quality + generated sets)

Checks the encoded corpus (hours, NPZ shape/NaN sanity, dup ratio) **and** both generated
manifests (schema, scenario/flag mix, duplicates). Exits non-zero if anything is unfit —
chain it in CI before a run.

```bash
python scripts/check_ready.py --config configs/en.yaml            # train + fc + domain
python scripts/check_ready.py --config configs/en.yaml --skip-gen # base corpus only
```
Prints per-section `PASS ✅ / FAIL ❌` and a final **READY FOR TRAINING ✅ / NOT READY ❌**.

## 5. Train — 5 phases, resumable, checkpoints auto-synced to the Hub

Every saved checkpoint is mirrored to the runs repo (`push_to_hub: true` in the config), so
a dropped box never loses progress. Resume any phase with `--resume auto`.

```bash
python scripts/03_train.py --config configs/en.yaml --phase 1   # CTC / projector warmup (frozen encoder)
python scripts/03_train.py --config configs/en.yaml --phase 2   # projector + Nandi align (frozen encoder)
python scripts/03_train.py --config configs/en.yaml --phase 4 --set data.manifest=data/manifests/fc.jsonl     # floor-control
python scripts/03_train.py --config configs/en.yaml --phase 5 --set data.manifest=data/manifests/domain.jsonl # domain correction
# optional — needs RAW audio (see §2), unfreezes the encoder, gated at <5% WER:
python scripts/03_train.py --config configs/en.yaml --phase 3
# resume:
python scripts/03_train.py --config configs/en.yaml --phase 2 --resume auto
```

| Phase | Trains | Loss mix | Encoder | Needs |
|---|---|---|---|---|
| 1 | CTC head + projector | lm 1.0 · ctc 0.3 | frozen | feats |
| 2 | projector + Nandi align | lm 1.0 · ctc 0.3 · fcc 0.2 | frozen | feats |
| 3 | joint full fine-tune | lm 1.0 · ctc 0.3 · fcc 0.2 | **unfrozen** (tiny LR) | **raw audio** |
| 4 | floor-control head | lm 1.0 · ctc 0.1 · fc 3.0 | frozen | `fc.jsonl` |
| 5 | domain correction | lm 1.0 · ctc 0.1 · fc 1.0 | frozen | `domain.jsonl` |

## 6. Evaluate + latency

```bash
python scripts/04_eval.py --config configs/en.yaml --ckpt checkpoints/<run>/checkpoint-XXXX --split test
python scripts/10_latency_bench.py --config configs/en.yaml --ckpt checkpoints/<run>/checkpoint-XXXX
```

## 7. Inference — offline + streaming with controllable floor control

```bash
python scripts/05_infer.py --config configs/en.yaml --ckpt <ckpt> --wav clip.wav
python scripts/05_infer.py --config configs/en.yaml --ckpt <ckpt> --wav clip.wav --stream \
    --bc-bias 0.0 --think-bias 0.0 --temperature 1.0        # dial floor-control at inference
```

## 8. Sync artifacts to the Hub (any stage)

```bash
python scripts/hf_sync.py push --config configs/en.yaml --what manifests
python scripts/hf_sync.py push --config configs/en.yaml --what run   --run <run_name>
python scripts/hf_sync.py push --config configs/en.yaml --what model --run <run_name>
```

---

## Layout
```
src/kupefdx/   config env ledger  model dataset collate  train evaluate stream  smoke checks
   fcgen/      schema scenarios audio_probe  generate   (LLM data gen: FC + domain, one module)
scripts/       00_smoke 01_dataprep 02_encode 03_train 04_eval 05_infer
               gen_data (FC+domain, one cmd)  check_ready (readiness gate, one cmd)
               10_latency_bench 11_shard_pipeline 12_build_manifest 14_hub_dedup  hf_sync
configs/       en.yaml (real)   smoke.yaml (tiny/CPU)
```

Full operational notes (disk-full recovery, Hub inventory, raw-data gather) live in
**[commands.md](commands.md)**. Architecture & phases: **[PLAN.md](PLAN.md)**.
