#!/usr/bin/env bash
# gather_all.sh — one command for every Hindi ASR source:
#   download → encode → push packed feats → flush → next dataset
#
# Continues on error so one bad source does not block the rest.
# Column names are auto-detected by 11_shard_pipeline.py (no peek needed).
#
# Usage (4090 + ~400GB disk — tuned for $/hr):
#   bash scripts/gather_all.sh
#   ONLY=shrutilipi_hi,indicvoices_hi,kathbath_hi bash scripts/gather_all.sh
#   RAW_ONLY=1 bash scripts/gather_all.sh                  # download + push raw only
#   PUSH_RAW=1 bash scripts/gather_all.sh                  # also upload wavs (slow)
#   SHARD_SIZE=2000 ENCODE_BATCH=48 bash scripts/gather_all.sh
#
# Encoder: configs/gpu.yaml → facebook/omniASR-W2V-300M (Meta SSL). Needs
#   pip install omnilingual-asr   OR the HF Wav2Vec2 mirror is used automatically.
#
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/gpu.yaml}"
# Bigger shards = fewer Hub commits. 2000 ≈ 3–4 h audio/shard.
SHARD_SIZE="${SHARD_SIZE:-2000}"
# GPU encode minibatch — 4090 bf16 handles 48–64 for ~10s clips.
ENCODE_BATCH="${ENCODE_BATCH:-48}"
RAW_ONLY="${RAW_ONLY:-0}"
PUSH_RAW="${PUSH_RAW:-0}"
ONLY="${ONLY:-}"

EXTRA=(--encode-batch "$ENCODE_BATCH")
if [[ "$RAW_ONLY" == "1" ]]; then
  EXTRA=(--raw-only)
elif [[ "$PUSH_RAW" == "1" ]]; then
  EXTRA+=(--push-raw)
fi

# name | hf_id | config | split | domain
# OpenSLR-only packs (MUCS, Gramvaani) are not HF-streamable — drop those in via --local-dir.
# Common Voice: HF mirrors are gated/empty since Mozilla moved to Data Collective — skip here.
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
