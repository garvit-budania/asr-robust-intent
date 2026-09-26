#!/usr/bin/env bash
# NOTE: --skip_whisper below skips the (slow) Whisper decoding pass and only
# produces the synthetic-corruption transcripts, since those are the only ones
# actually consumed downstream. This is the main lever for staying under the
# 14-15 hour budget for this step. If you have time to spare, remove
# --skip_whisper to also get real Whisper-decoded realized-WER numbers.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

# NOTE: update these two if the encoder grid results (Table II) show a different
# config as best -- check results/stats/seed_aggregated_accuracy.csv first.
BEST_AUDIO="microsoft/wavlm-base-plus"
BEST_TEXT="roberta-base"

for ds in slurp fsc; do
  echo "== Generating WER transcript sets for $ds (unchanged step, uses existing script) =="
  python3 -m src.robustness.generate_wer_transcripts --dataset "$ds" --split test --skip_whisper

  echo "== Extracting TEXT embeddings for each corrupted WER level ($ds) =="
  # WER 0% uses the already-extracted clean-transcript cache -- no extra work needed.
  # wer_key values must match Python's str(float) serialization used when the
  # transcript JSON was written (0.1, 0.2, 0.3 -- NOT "0.10"/"0.20"/"0.30").
  declare -A WER_TAGS=( ["0.1"]="wer10" ["0.2"]="wer20" ["0.3"]="wer30" )
  for wer_key in "${!WER_TAGS[@]}"; do
    tag="${WER_TAGS[$wer_key]}"
    python3 -m src.features.extract_embeddings --dataset "$ds" --encoder_type text \
      --encoder_name "$BEST_TEXT" --split test \
      --corrupted_transcripts_json "results/robustness/transcripts/${ds}_test_synthetic.json" \
      --wer_key "$wer_key" --tag "$tag"
  done

  echo "== Evaluating robustness curve for $ds =="
  python3 -m src.robustness.eval_robustness_head --dataset "$ds" \
      --best_fused_audio "$BEST_AUDIO" --best_fused_text "$BEST_TEXT" --seed 42
done

echo "Robustness results -> results/robustness/"
