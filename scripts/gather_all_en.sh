#!/usr/bin/env bash
# One-shot ENGLISH data pipeline: download -> encode -> push -> flush, for ALL sources.
# Resumable (per-shard ledger) and continue-on-error (a gated/failed source won't stop the rest).
#
#   RAW_ONLY=1  (default) : gather raw audio only (fast, no encoder). Do this first.
#   RAW_ONLY=0            : also encode features+codes (needs FastConformer/NeMo working).
#   CFG=configs/en.yaml SHARD=500 bash scripts/gather_all_en.sh
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-configs/en.yaml}
SHARD=${SHARD:-500}
RAW_ONLY=${RAW_ONLY:-1}
NO_FLUSH=${NO_FLUSH:-0}          # 1 = keep raw+encoded on local disk (single-box training)
EXTRA=""
[ "$RAW_ONLY" = "1" ] && EXTRA="$EXTRA --raw-only"
[ "$NO_FLUSH" = "1" ] && EXTRA="$EXTRA --no-flush"

# id | config | split | domain     (columns auto-detected; --split matters for LibriSpeech)
DATASETS=(
  "openslr/librispeech_asr|clean|train.clean.100|read"
  "openslr/librispeech_asr|clean|train.clean.360|read"
  "openslr/librispeech_asr|other|train.other.500|read"
  "mozilla-foundation/common_voice_17_0|en|train|accented"   # accept license first if gated
)
# Big optional corpora (uncomment for volume; large downloads — you have 400GB):
# DATASETS+=("MLCommons/peoples_speech|clean|train|spontaneous")   # ~30k h
# DATASETS+=("speechcolab/gigaspeech|l|train|mixed")               # gated

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r ID CFGNAME SPLIT DOMAIN <<< "$entry"
  echo "════════ $ID  [$CFGNAME / $SPLIT]  domain=$DOMAIN ════════"
  python scripts/11_shard_pipeline.py --config "$CFG" \
      --hf-id "$ID" --hf-config "$CFGNAME" --split "$SPLIT" \
      --domain "$DOMAIN" --shard-size "$SHARD" $EXTRA \
    || echo "!! $ID failed (gated/license/column) — continuing to next source"
done

echo "════════ ALL SOURCES DONE — cumulative hours ════════"
python - <<'PY'
import json, os
p = "data/manifests/shardpipe.json"
if os.path.exists(p):
    d = json.load(open(p)); meta = d.get("meta", {})
    h = sum(m.get("hours", 0) for m in meta.values())
    done = sum(1 for v in d.get("states", {}).values() if v == "done")
    print(f"  {h:.1f} h pushed across {done} shards")
else:
    print("  no ledger yet")
PY
