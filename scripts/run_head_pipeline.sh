#!/usr/bin/env bash
# Runs the ENTIRE frozen-encoder pipeline end to end. Target: 14-15 hours max
# on a single GPU. This is the one command to run.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "########## STEP 0: PRE-FLIGHT CHECKS ##########"
echo "-- Clearing stale multiprocessing temp files (prevents 'No space left on device' crashes) --"
rm -rf /tmp/pymp-* 2>/dev/null || true
df -h /tmp | tail -1

echo "-- Checking real audio duration (right-sizes extraction) --"
source .venv/bin/activate 2>/dev/null || true
python3 src/data/derive_audio_len.py --config configs/experiment.yaml
echo "!! If the numbers above differ meaningfully from configs/experiment.yaml's"
echo "   max_audio_seconds, this run will be slower than necessary. Consider"
echo "   Ctrl+C now, updating the config, and re-launching -- proceeding in 15s."
sleep 15

echo "########## STEP 1: EXTRACT ALL EMBEDDINGS (the expensive step) ##########"
bash scripts/04_extract_all_features.sh

echo "########## STEP 2: TRAIN ALL HEADS (fast) ##########"
bash scripts/05_train_all_heads.sh

echo "########## STEP 3: ABLATIONS (fast) ##########"
bash scripts/06_run_ablations_head.sh

echo "########## STEP 4: ASR ROBUSTNESS SWEEP ##########"
bash scripts/07_run_robustness_head.sh

echo "########## STEP 5: STATISTICS ##########"
python3 -m src.stats.aggregate_seeds
echo "Check results/stats/seed_aggregated_accuracy.csv above, then run McNemar's"
echo "test manually on the actual best/second-best fused configs, e.g.:"
echo "  python3 -m src.stats.mcnemar_test_head --dataset slurp \\"
echo "      --config_a wav2vec2-base_roberta-base --config_b wavlm-base-plus_roberta-base"

echo "########## STEP 6: BUILD TABLES ##########"
python3 -m src.reporting.build_tables

echo "Done. See results/paper_tables.md for everything ready to paste into the paper."
