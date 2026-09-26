# ASR-Robust Multimodal Intent Recognition

Reproducibility repository for the frozen-encoder multimodal intent-recognition experiments described in the accompanying paper.

The pipeline combines pretrained audio and text encoders with lightweight trainable classification/fusion heads. The pretrained encoders are kept frozen; training is performed on cached embeddings.

---

## 1. Repository Overview

The repository contains:

- Source code for dataset preparation, feature extraction, head training, fusion, robustness evaluation, statistical tests, and reporting.
- Configuration used for the experiments.
- Trained model checkpoints.
- Previously generated experimental results and paper tables.

Large datasets and cached feature embeddings are intentionally not included in the Git repository because they are regenerable and substantially larger than the source code and checkpoints.

### Repository structure

```text
asr-robust-intent/
├── checkpoints/          # Supplied trained model checkpoints
├── configs/
│   └── experiment.yaml
├── results/              # Generated results and paper tables
├── scripts/
│   ├── 00_setup_env.sh
│   ├── 01_download_datasets.sh
│   ├── 02_download_whisper.sh
│   ├── 04_extract_all_features.sh
│   ├── 05_train_all_heads.sh
│   ├── 06_run_ablations_head.sh
│   ├── 07_run_robustness_head.sh
│   └── run_head_pipeline.sh
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── reporting/
│   ├── robustness/
│   ├── stats/
│   └── train_head.py
├── README.md
├── REQUIREMENTS.md
└── requirements.txt
