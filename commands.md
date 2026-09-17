# KupeFDX — full command reference (English ASR + floor control)

Repo: `iNavLabsResearch/kupe-FDX-hi-asr` · HF data: `anuj-inavlabs/kupe-en-asr-data` (public)
Config: `configs/en.yaml` (FastConformer encoder + Nandi-Mini-150M).

> **Two data tracks — read this first**
> - **`feats.npz`** (encoder features) → all you need for **frozen-encoder** training
>   (phases 1,2,4,5). This is standard for speech-LLMs and reaches ~3% WER because
>   FastConformer is already a strong pretrained ASR encoder. **Already on the Hub.**
> - **raw audio** → only needed if you **unfreeze the encoder** (phase 3 JOINT full-FT).
>   Raw is *not* on the Hub yet (we gathered feats-only). See §7.

---

## 0. Clone + environment (any box)

```bash
git clone https://github.com/iNavLabsResearch/kupe-FDX-hi-asr.git
cd kupe-FDX-hi-asr
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill HF_TOKEN, KUPE_LLM_API_KEY, (WANDB optional)
set -a && . ./.env && set +a
```

## 1. Smoke test (proves the whole pipeline wires up, no GPU/net)

```bash
python scripts/00_smoke.py --config configs/smoke.yaml
```

## 1b. Box is full / `No usable temporary directory` / `no shard manifests`

This is the gather box filling itself. Encoded shards are **already on the Hub** — delete
local caches, then build the manifest. Do this **before** `git pull` if pull itself fails.

```bash
# 1. kill stragglers (bash wrappers survive a python pkill — kill both)
pkill -9 -f 11_shard_pipeline.py; pkill -9 -f gather_all_en.sh; sleep 2
jobs -l; ps aux | grep -E '[1]1_shard_pipeline|[g]ather_all_en'   # must be empty

# 2. free disk (source caches + raw wavs + leftover upload staging). KEEP data/encoded/
#    if you gathered with NO_FLUSH=1 and want a local train.jsonl without re-download.
df -h
rm -rf ~/.cache/huggingface/datasets ~/.cache/huggingface/hub/datasets--*
rm -rf data/raw data/hubbatch_* data/hubup_* /tmp/* /var/tmp/*
df -h   # need ≳20G free to import python / git pull

git pull
source .venv/bin/activate

# 3. build train.jsonl from local --no-flush shards (shards/<sid>/_hub/) OR from Hub
python scripts/12_build_manifest.py --config configs/en.yaml --out data/manifests/train.jsonl
# if that still says no manifests (you flushed to Hub):
python scripts/12_build_manifest.py --config configs/en.yaml --out data/manifests/train.jsonl --from-hub
python scripts/13_data_report.py --config configs/en.yaml --manifest data/manifests/train.jsonl --sample 500
```

People's Speech never ran (two overlapping gathers stole the GPU for VoxPopuli). After
disk is free, gather it **alone**, **no** `NO_FLUSH` (Hub already has everything else):

```bash
pkill -9 -f 11_shard_pipeline.py; sleep 2
ENCODE_BATCH=96 MAX_BATCH_SEC=400 MAXH_SPONT=2500 ONLY="peoples_speech" \
  DL_WORKERS=6 NO_FLUSH=0 RAW_ONLY=0 CFG=configs/en.yaml \
  nohup bash scripts/gather_all_en.sh > gather.log 2>&1 &
tail -f gather.log
```

## 2. Gather data (fast: parallel download → GPU encode → batched Hub push)

Always run detached so a dropped SSH can't kill it; only ONE job at a time.
`NO_FLUSH=1` keeps every shard on disk (~tens of GB per 1k hours) and will fill the box —
default is flush-after-upload. Pull feats back in §7 if you train on another machine.

```bash
pkill -9 -f 11_shard_pipeline.py; sleep 2
ps aux | grep -c "[1]1_shard_pipeline.py"   # must print 0

ENCODE_BATCH=96 MAX_BATCH_SEC=400 MAXH_SPONT=2500 \
  ONLY="peoples_speech voxpopuli" DL_WORKERS=6 \
  NO_FLUSH=0 RAW_ONLY=0 CFG=configs/en.yaml \
  nohup bash scripts/gather_all_en.sh > gather.log 2>&1 &
tail -f gather.log
```

Knobs: `DL_WORKERS` parallel download threads · `ENCODE_BATCH`/`MAX_BATCH_SEC` GPU batch
· `MAXH_SPONT`/`MAXH_ACCENT`/`MAXH_YODAS` per-source hour caps · `ONLY="id1 id2"` pick
sources · `NO_FLUSH=1` keep shards on local disk (needed for §4 local build, eats disk)
· `MIN_FREE_GB=20` abort/pause when free space drops below this.

Sources (all REAL human speech): LibriSpeech (done) · People's Speech (bulk, varied) ·
VoxPopuli (accented) · YODAS2 (opt-in, gated, YouTube conversational).
`MAXH_SPONT=2500` → ~960+2500+500 ≈ **3,960 h**.

## 3. Check what's on the Hub (how much / how many, per source)

```bash
python - <<'PY'
from huggingface_hub import HfApi, hf_hub_download; import os,re,collections,json
api=HfApi(); rid="anuj-inavlabs/kupe-en-asr-data"; tok=os.environ["HF_TOKEN"]
c=collections.Counter()
for f in api.list_repo_files(rid,repo_type="dataset",token=tok):
    if f.startswith("encoded/") and f.endswith("/feats.npz"):
        s=re.sub(r'(__w\d+x\d+)?_n\d+_shard_\d+$','',f.split('/')[1]); c[s]+=1
for s,n in sorted(c.items(),key=lambda x:-x[1]): print(f"{n:5d} shards  {s}")
p=hf_hub_download(rid,"ledger/shardpipe.json",repo_type="dataset",token=tok)
m=json.load(open(p)).get("meta",{}); print("LEDGER:",round(sum(v.get('hours',0) for v in m.values()),1),"h")
PY
```

## 4. Build the training manifest (from local shards; content-dedups)

Torch-free. Recursive glob finds `shards/<sid>/_hub/manifest.jsonl` (local) and
`encoded/<sid>/manifest.jsonl` (Hub pull). `--from-hub` downloads manifests only.

```bash
python scripts/12_build_manifest.py --config configs/en.yaml --out data/manifests/train.jsonl
# or, if local shards were flushed:
python scripts/12_build_manifest.py --config configs/en.yaml --out data/manifests/train.jsonl --from-hub
```
Then in `configs/en.yaml`: `data.manifest: data/manifests/train.jsonl`,
`data.use_cached_feats: true`.

## 5. Dataset distribution + NPZ sanity report  ← sanity check

```bash
python scripts/13_data_report.py --config configs/en.yaml \
    --manifest data/manifests/train.jsonl --sample 500
```
Prints hours by domain/split, duration histogram, transcript stats, duplicate ratio,
and a real NPZ pass (loads sampled feats.npz, checks shape `[T,512]`, dtype, NaN/Inf,
frame-rate vs duration). Exits non-zero on a hard problem (missing/NaN/wrong-dim feats).

## 6. gpt-luna data generation (floor-control + domain correction)

Both agents show live token/cost. Always `--mock` first (no keys) to prove the generator,
then the real run, then GATE with the quality checker.

```bash
# 6a. Floor-control data (conversational clips only; NPTEL/read excluded by default)
python scripts/06_gen_fc.py --config configs/en.yaml --src data/manifests/train.jsonl \
    --out data/manifests/fc.jsonl --mock --limit 50          # dry-run
python scripts/06_gen_fc.py --config configs/en.yaml --src data/manifests/train.jsonl \
    --out data/manifests/fc.jsonl --concurrency 10           # real (uses KUPE_LLM_* from .env)

# 6b. Domain-correction data (Phase 5)
python scripts/07_gen_domain.py --config configs/en.yaml --src data/manifests/train.jsonl \
    --out data/manifests/domain.jsonl --mock --limit 100     # dry-run
python scripts/07_gen_domain.py --config configs/en.yaml --src data/manifests/train.jsonl \
    --out data/manifests/domain.jsonl --concurrency 10       # real
```

Set `KUPE_LLM_PRICE_IN` / `KUPE_LLM_PRICE_OUT` in `.env` to see `$` in the `[cost]` lines.

## 6c. Data-quality gate (strict; exits non-zero if it fails)

```bash
python scripts/09_data_quality.py --manifest data/manifests/fc.jsonl --lang en
python scripts/09_data_quality.py --manifest data/manifests/domain.jsonl --kind domain --lang en
```

## 7. Fetch data onto the training box (H100 / RTX Pro 6000)

**Frozen-encoder training (phases 1,2,4,5) — feats only, small & fast:**
```bash
# grabs feats.npz + manifests into ./data (LFS, resumable, parallel)
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download anuj-inavlabs/kupe-en-asr-data \
    --repo-type dataset --local-dir data --include "encoded/**" "ledger/**"
python scripts/12_build_manifest.py --config configs/en.yaml --out data/manifests/train.jsonl
python scripts/13_data_report.py --config configs/en.yaml --manifest data/manifests/train.jsonl
```
(`pip install hf_transfer` first for max download speed.)

**Full-FT with UNFROZEN encoder (phase 3) needs RAW audio — not on the Hub yet.**
Cached feats are frozen-encoder outputs, so backprop into the encoder can't use them.
Recommended: keep the encoder **frozen** (reaches ~3%, no raw needed). If you truly want
phase 3, gather a bounded subset WITH raw and flush local disk (raw wav for 4,000 h ≈
460 GB — do a subset):
```bash
PUSH_RAW=1 NO_FLUSH=0 MAXH_SPONT=800 ONLY="peoples_speech" DL_WORKERS=6 \
  CFG=configs/en.yaml nohup bash scripts/gather_all_en.sh > gather_raw.log 2>&1 &
# then on the training box:
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download anuj-inavlabs/kupe-en-asr-data \
    --repo-type dataset --local-dir data --include "raw/**"
```

## 8. Train (resumable; per-phase). See phases in 03_train.py.

```bash
python scripts/03_train.py --config configs/en.yaml --phase 1   # projector warmup (CTC)
python scripts/03_train.py --config configs/en.yaml --phase 2   # projector+Nandi align (frozen enc)
python scripts/03_train.py --config configs/en.yaml --phase 4   # floor-control (needs fc.jsonl)
python scripts/03_train.py --config configs/en.yaml --phase 5   # domain correction (needs domain.jsonl)
# optional, needs raw audio (§7):
python scripts/03_train.py --config configs/en.yaml --phase 3   # joint full-FT (encoder unfrozen)
# resume:
python scripts/03_train.py --config configs/en.yaml --phase 2 --resume auto
```

## 9. Evaluate + latency

```bash
python scripts/04_eval.py --config configs/en.yaml --ckpt checkpoints/<run>/checkpoint-XXXX --split test
python scripts/10_latency_bench.py --config configs/en.yaml --ckpt checkpoints/<run>/checkpoint-XXXX
```

## 10. Inference (offline + streaming with controllable floor control)

```bash
python scripts/05_infer.py --config configs/en.yaml --ckpt <ckpt> --wav clip.wav
python scripts/05_infer.py --config configs/en.yaml --ckpt <ckpt> --wav clip.wav --stream \
    --bc-bias 0.0 --think-bias 0.0 --temperature 1.0        # dial floor-control at inference
```

## 11. Sync artifacts to the Hub (any stage)

```bash
python scripts/hf_sync.py push --config configs/en.yaml --what manifests
python scripts/hf_sync.py push --config configs/en.yaml --what run   --run <run_name>
python scripts/hf_sync.py push --config configs/en.yaml --what model --run <run_name>
```
