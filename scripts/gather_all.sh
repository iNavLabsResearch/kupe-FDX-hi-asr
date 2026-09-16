#!/usr/bin/env bash
# gather_all.sh — one command for every Hindi ASR source:
#   download → encode → push (raw + encoded) → flush → next dataset
#
# Continues on error so one bad source does not block the rest.
# Column names are auto-detected by 11_shard_pipeline.py (no peek needed).
#
# Usage:
#   bash scripts/gather_all.sh                         # full pipeline (default)
#   RAW_ONLY=1 bash scripts/gather_all.sh              # download + push raw only
#   SHARD_SIZE=500 CONFIG=configs/gpu.yaml bash scripts/gather_all.sh
#   ONLY=fleurs_hi,shrutilipi_hi bash scripts/gather_all.sh   # subset by name
#
# Encoder: configs/gpu.yaml → facebook/omniASR-W2V-300M (Meta SSL). Needs
#   pip install omnilingual-asr   OR the HF Wav2Vec2 mirror is used automatically.
#
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/gpu.yaml}"
SHARD_SIZE="${SHARD_SIZE:-500}"
RAW_ONLY="${RAW_ONLY:-0}"
ONLY="${ONLY:-}"

EXTRA=()
if [[ "$RAW_ONLY" == "1" ]]; then
  EXTRA+=(--raw-only)
fi

# name | hf_id | config | split | domain
# OpenSLR-only packs (MUCS, Gramvaani) are not HF-streamable — drop those in via --local-dir.
# CV 17 is gated / often empty without accepting terms — use 16_1 (also gated; accept on HF).
DATASETS=(
  "fleurs_hi|google/fleurs|hi_in|train|general"
  "common_voice_hi|mozilla-foundation/common_voice_16_1|hi|train|general"
  "shrutilipi_hi|ai4bharat/Shrutilipi|hindi|train|news"
  "indicvoices_hi|ai4bharat/IndicVoices|hindi|train|spontaneous"
  "kathbath_hi|ai4bharat/Kathbath|hindi|train|read"
)

ok=()
failed=()

want() {
  local name="$1"
  [[ -z "$ONLY" ]] && return 0
  [[ ",${ONLY}," == *",${name},"* ]]
}

for entry in "${DATASETS[@]}"; do
  IFS='|' read -r name hf_id cfg split domain <<< "$entry"
  want "$name" || continue

  echo
  echo "======== ${name}  (${hf_id} / ${cfg} / ${split}) ========"
  if python scripts/11_shard_pipeline.py \
      --config "$CONFIG" \
      --hf-id "$hf_id" \
      --hf-config "$cfg" \
      --split "$split" \
      --domain "$domain" \
      --shard-size "$SHARD_SIZE" \
      ${EXTRA[@]+"${EXTRA[@]}"}; then
    ok+=("$name")
  else
    echo "WARN: ${name} failed — continuing"
    failed+=("$name")
  fi
done

echo
echo "======== summary ========"
echo "ok:      ${ok[*]:-(none)}"
echo "failed:  ${failed[*]:-(none)}"
if ((${#failed[@]} > 0)); then
  exit 1
fi
