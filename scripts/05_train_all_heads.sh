#!/usr/bin/env bash
# Trains every fused config + audio-only + text-only baseline over CACHED
# embeddings only (src/train_head.py) -- no large encoders touched here, so this
# entire script should run in minutes-to-a-few-hours, not days. Restores full
# 5-seed statistical rigor since head training is cheap.
#
# Still dataset-prioritized: completes everything SLURP needs before starting FSC,
# so an interruption still leaves one fully complete, analyzable dataset.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

mkdir -p logs
FAIL_LOG="logs/failed_head_runs.log"
: > "$FAIL_LOG"

run_job() {
    echo ">> $*"
    if ! "$@"; then
        echo "FAILED: $*" >> "$FAIL_LOG"
    fi
}

SEEDS=(13 42 123 777 2024)
AUDIO_ENCODERS=("facebook/wav2vec2-base" "facebook/hubert-base-ls960" "microsoft/wavlm-base-plus")
TEXT_ENCODERS=("distilbert-base-uncased" "roberta-base")

run_dataset_complete() {
    local ds="$1"
    echo "=================================================================="
    echo "Completing ALL required head-training jobs for dataset: $ds"
    echo "=================================================================="

    for audio in "${AUDIO_ENCODERS[@]}"; do
      for text in "${TEXT_ENCODERS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          run_job python3 -m src.train_head \
            --dataset "$ds" --model_type fused \
            --audio_encoder "$audio" --text_encoder "$text" \
            --seed "$seed"
        done
      done
    done

    for seed in "${SEEDS[@]}"; do
      run_job python3 -m src.train_head --dataset "$ds" --model_type text_only \
        --text_encoder roberta-base --seed "$seed"
    done

    for seed in "${SEEDS[@]}"; do
      run_job python3 -m src.train_head --dataset "$ds" --model_type audio_only \
        --audio_encoder facebook/hubert-base-ls960 --seed "$seed"
    done

    echo "$ds is now FULLY COMPLETE."
}

run_dataset_complete "slurp"
run_dataset_complete "fsc"

echo "=================================================================="
N_FAILED=$(wc -l < "$FAIL_LOG")
if [ "$N_FAILED" -gt 0 ]; then
    echo "Finished with $N_FAILED failed job(s). See $FAIL_LOG."
else
    echo "All head-training runs completed successfully, zero failures."
fi
echo "Aggregate with: python3 -m src.stats.aggregate_seeds"
echo "=================================================================="
