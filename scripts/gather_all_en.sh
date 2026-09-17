#!/usr/bin/env bash
# One-shot ENGLISH data pipeline: download -> encode -> push -> flush, for ALL sources.
# Diversified pretraining mix (~4,000+ h) incl. ~1,000 h Indian-English (AI4Bharat NPTEL).
# Resumable per shard; continue-on-error (a gated/failed source won't stop the rest).
#
#   RAW_ONLY=1 (default): raw audio only (fast, no encoder). RAW_ONLY=0: also encode.
#   NO_FLUSH=1          : keep raw+encoded on local disk (single-box training).
#   CFG=configs/en.yaml SHARD=500 RAW_ONLY=0 NO_FLUSH=1 bash scripts/gather_all_en.sh
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=${CFG:-configs/en.yaml}
SHARD=${SHARD:-500}
RAW_ONLY=${RAW_ONLY:-1}
NO_FLUSH=${NO_FLUSH:-0}
EXTRA=""
[ "$RAW_ONLY" = "1" ] && EXTRA="$EXTRA --raw-only"
[ "$NO_FLUSH" = "1" ] && EXTRA="$EXTRA --no-flush"

# id | config | split | domain | max_hours   (config "-" = none; max_hours 0 = all)
# Diversify accents/styles: US read, spontaneous, podcasts/audiobooks, crowd accents,
# and INDIAN-English lectures. Total pretraining ~4,000+ h.
DATASETS=(
  "openslr/librispeech_asr|clean|train.100|read_us|0"          # ~100 h
  "openslr/librispeech_asr|clean|train.360|read_us|0"          # ~360 h
  "openslr/librispeech_asr|other|train.500|read_us|0"          # ~500 h
  "ai4bharat/NPTEL|-|train|indian_english|1000"                      # ~1000 h INDIAN English
  "MLCommons/peoples_speech|clean|train|spontaneous|1000"           # ~1000 h spontaneous
  "speechcolab/gigaspeech|l|train|podcasts_audiobooks|1000"          # ~1000 h (gated: accept)
  "facebook/voxpopuli|en|train|accented|500"                       # ~500 h accented English (ungated; CV moved off HF)
)
# NOTE: ai4bharat/Svarah (9.6 h Indian-English) is an EVAL benchmark — do NOT train on it;
#       use it as a held-out Indian-accent test set for scripts/04_eval.py.

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r ID CFGNAME SPLIT DOMAIN MAXH <<< "$entry"
  echo "════════ $ID  [$CFGNAME / $SPLIT]  domain=$DOMAIN  max=${MAXH}h ════════"
  CFGFLAG=(--hf-config "$CFGNAME"); [ "$CFGNAME" = "-" ] && CFGFLAG=()
  python scripts/11_shard_pipeline.py --config "$CFG" \
      --hf-id "$ID" "${CFGFLAG[@]}" --split "$SPLIT" \
      --domain "$DOMAIN" --shard-size "$SHARD" --max-hours "$MAXH" $EXTRA \
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
