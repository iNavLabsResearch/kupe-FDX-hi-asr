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
[ "${PUSH_RAW:-0}" = "1" ] && EXTRA="$EXTRA --push-raw"   # also store raw wav+text on HF (~10x storage; needed only for full-FT)

# id | config | split | domain | max_hours   (config "-" = none; max_hours 0 = all)
# Diversify accents/styles: US read, spontaneous, podcasts/audiobooks, crowd accents,
# and INDIAN-English lectures. Total pretraining ~4,000+ h.
# DL_WORKERS parallel download threads per source (disjoint file-shards → 1 GPU encoder).
# Set DL_WORKERS=4 (default) to cut download-bound wall-clock ~4×. MAXH_* cap hours/source.
DATASETS=(
  "openslr/librispeech_asr|clean|train.100|read_us|0"          # ~100 h (done → hub-sync skips)
  "openslr/librispeech_asr|clean|train.360|read_us|0"          # ~360 h (done)
  "openslr/librispeech_asr|other|train.500|read_us|0"          # ~500 h (done)
  "MLCommons/peoples_speech|clean|train|spontaneous|${MAXH_SPONT:-2500}"      # REAL varied speech (main lever)
  "facebook/voxpopuli|en|train|accented|${MAXH_ACCENT:-500}"                  # REAL parliamentary, accented
  "espnet/yodas2|en000|train|youtube_natural|${MAXH_YODAS:-500}"             # REAL YouTube (optional; gated, noisier labels)
)
# ~4,000 h of REAL human speech: LibriSpeech ~960 + People's Speech (raise cap) + VoxPopuli 500.
# People's Speech is the bulk lever — real, varied, fast (kept_h≈1.9), NOT robotic read speech.
# To reach 4000h: MAXH_SPONT=2500 gives PS ~2500h → 960+2500+500 ≈ 3960h.
# YODAS2 is OFF by default (MAXH_YODAS=0); enable it for natural conversational diversity.
# Dropped: MLS English (real, but read-audiobook style), GigaSpeech (slow, kept_h≈0.5),
#          ai4bharat/NPTEL (En→Indic translation, no ASR audio).
# NOTE: ai4bharat/Svarah (9.6 h Indian-English) is an EVAL benchmark — do NOT train on it;
#       use it as a held-out Indian-accent test set for scripts/04_eval.py.

# ONLY="id1 id2" runs just those sources (substring match) — use it to parallel-gather
# the un-done sources without touching ones already complete under the old naming, e.g.:
#   ONLY="peoples_speech gigaspeech voxpopuli" DL_WORKERS=4 bash scripts/gather_all_en.sh
for entry in "${DATASETS[@]}"; do
  IFS='|' read -r ID CFGNAME SPLIT DOMAIN MAXH <<< "$entry"
  if [ -n "${ONLY:-}" ]; then
    keep=0; for tok in $ONLY; do case "$ID" in *"$tok"*) keep=1;; esac; done
    [ "$keep" = "1" ] || { echo "-- skip $ID (not in ONLY)"; continue; }
  fi
  echo "════════ $ID  [$CFGNAME / $SPLIT]  domain=$DOMAIN  max=${MAXH}h ════════"
  CFGFLAG=(--hf-config "$CFGNAME"); [ "$CFGNAME" = "-" ] && CFGFLAG=()
  python scripts/11_shard_pipeline.py --config "$CFG" \
      --hf-id "$ID" "${CFGFLAG[@]}" --split "$SPLIT" \
      --domain "$DOMAIN" --shard-size "$SHARD" --upload-every "${UPLOAD_EVERY:-16}" \
      --dl-workers "${DL_WORKERS:-4}" \
      --encode-batch "${ENCODE_BATCH:-64}" --max-batch-sec "${MAX_BATCH_SEC:-300}" \
      --prefetch "${PREFETCH:-12}" --max-hours "$MAXH" $EXTRA \
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
