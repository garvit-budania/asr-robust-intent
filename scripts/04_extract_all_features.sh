#!/usr/bin/env bash
# Extracts and caches frozen embeddings for every encoder x dataset x split.
# This is the ONE expensive step in the efficient approach -- everything after
# this (src/train_head.py) is fast, since it never touches the large encoders.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

mkdir -p features logs
FAIL_LOG="logs/failed_extractions.log"
: > "$FAIL_LOG"

run_job() {
    echo ">> $*"
    if ! "$@"; then
        echo "FAILED: $*" >> "$FAIL_LOG"
    fi
}

DATASETS=(slurp fsc)
SPLITS=(train dev test)
AUDIO_ENCODERS=("facebook/wav2vec2-base" "facebook/hubert-base-ls960" "microsoft/wavlm-base-plus")
TEXT_ENCODERS=("distilbert-base-uncased" "roberta-base")

echo "=================================================================="
echo "Extracting AUDIO embeddings (3 encoders x 2 datasets x 3 splits = 18 jobs)"
echo "=================================================================="
for ds in "${DATASETS[@]}"; do
  for split in "${SPLITS[@]}"; do
    for audio in "${AUDIO_ENCODERS[@]}"; do
      run_job python3 -m src.features.extract_embeddings \
        --dataset "$ds" --encoder_type audio --encoder_name "$audio" --split "$split"
    done
  done
done

echo "=================================================================="
echo "Extracting TEXT embeddings, clean transcripts (2 encoders x 2 datasets x 3 splits = 12 jobs)"
echo "=================================================================="
for ds in "${DATASETS[@]}"; do
  for split in "${SPLITS[@]}"; do
    for text in "${TEXT_ENCODERS[@]}"; do
      run_job python3 -m src.features.extract_embeddings \
        --dataset "$ds" --encoder_type text --encoder_name "$text" --split "$split"
    done
  done
done

echo "=================================================================="
N_FAILED=$(wc -l < "$FAIL_LOG")
if [ "$N_FAILED" -gt 0 ]; then
    echo "Extraction finished with $N_FAILED failed job(s). See $FAIL_LOG."
else
    echo "All feature extraction completed successfully, zero failures."
fi
echo "Next: bash scripts/05_train_all_heads.sh"
echo "=================================================================="
