#!/usr/bin/env bash
# gather_all.sh — continuous download ‖ encode ‖ Hub push (packed feats, no raw by default).
#
# Cost-tuned defaults for a 24GB 4090 (~$0.79/hr):
#   ENCODE_BATCH=16  MAX_BATCH_SEC=48  PREFETCH=6  SHARD_SIZE=2000
#   (OOM auto-splits; done shards skip disk write)
#
#   ONLY=shrutilipi_hi,indicvoices_hi,kathbath_hi bash scripts/gather_all.sh
#
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/gpu.yaml}"
SHARD_SIZE="${SHARD_SIZE:-2000}"
ENCODE_BATCH="${ENCODE_BATCH:-16}"
MAX_BATCH_SEC="${MAX_BATCH_SEC:-48}"
PREFETCH="${PREFETCH:-6}"
RAW_ONLY="${RAW_ONLY:-0}"
PUSH_RAW="${PUSH_RAW:-0}"
ONLY="${ONLY:-}"

EXTRA=(--encode-batch "$ENCODE_BATCH" --max-batch-sec "$MAX_BATCH_SEC" --prefetch "$PREFETCH")
if [[ "$RAW_ONLY" == "1" ]]; then
  EXTRA=(--raw-only --prefetch "$PREFETCH")
elif [[ "$PUSH_RAW" == "1" ]]; then
  EXTRA+=(--push-raw)
fi

DATASETS=(
  "fleurs_hi|google/fleurs|hi_in|train|general"
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
