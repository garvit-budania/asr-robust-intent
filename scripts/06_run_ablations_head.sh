#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

SEEDS=(13 42 123 777 2024)
DATASETS=(slurp fsc)
AUDIO_ENCODERS=("facebook/wav2vec2-base" "facebook/hubert-base-ls960" "microsoft/wavlm-base-plus")
TEXT_ENCODERS=("distilbert-base-uncased" "roberta-base")

echo "== Calibration ablation (all fused configs) =="
for ds in "${DATASETS[@]}"; do
  for audio in "${AUDIO_ENCODERS[@]}"; do
    for text in "${TEXT_ENCODERS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        python3 -m src.ablate_calibration_head \
          --dataset "$ds" --audio_encoder "$audio" --text_encoder "$text" --seed "$seed"
      done
    done
  done
done

echo "== Fusion-level ablation: train embedding-level variant (one seed per config is enough) =="
for ds in "${DATASETS[@]}"; do
  for audio in "${AUDIO_ENCODERS[@]}"; do
    for text in "${TEXT_ENCODERS[@]}"; do
      python3 -m src.train_head --dataset "$ds" --model_type fused \
        --audio_encoder "$audio" --text_encoder "$text" \
        --seed 42 --embedding_level_fusion
      python3 -m src.ablate_fusion_level \
        --dataset "$ds" --audio_encoder "$audio" --text_encoder "$text" --seed 42
    done
  done
done

echo "Ablations complete -> results/ablations/"
