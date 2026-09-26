## Table II: Intent Accuracy (mean +/- std over 5 seeds)

| System | FSC | SLURP |
|---|---|---|
| text_only_roberta-base | 1.000 +/- 0.000 | 0.883 +/- 0.003 |
| audio_only_hubert-base-ls960 | 0.926 +/- 0.004 | 0.455 +/- 0.003 |
| fused_wav2vec2-base_distilbert-base-uncased | 1.000 +/- 0.000 | 0.880 +/- 0.001 |
| fused_wav2vec2-base_roberta-base | 1.000 +/- 0.000 | 0.881 +/- 0.002 |
| fused_hubert-base-ls960_distilbert-base-uncased | 1.000 +/- 0.000 | 0.882 +/- 0.002 |
| fused_hubert-base-ls960_roberta-base | 1.000 +/- 0.000 | 0.881 +/- 0.002 |
| fused_wavlm-base-plus_distilbert-base-uncased | 1.000 +/- 0.000 | 0.881 +/- 0.002 |
| fused_wavlm-base-plus_roberta-base | 1.000 +/- 0.000 | 0.880 +/- 0.004 |


## Table III: Accuracy under Transcript Corruption (SLURP)

| WER | Text-only RoBERTa | Best fused | Audio-only floor |
|---|---|---|---|
| 0% | 0.883 | 0.874 | 0.093 |
| 10% | 0.858 | 0.848 | 0.093 |
| 20% | 0.811 | 0.811 | 0.093 |
| 30% | 0.770 | 0.768 | 0.093 |


## Ablation Summary

- Calibration ablation: 60 runs, mean accuracy delta (calibrated - uncalibrated) = -0.0089
- Fusion-level ablation: 12 runs, mean (probability-level minus embedding-level) = -0.0022