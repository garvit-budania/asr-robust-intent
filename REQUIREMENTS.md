# What You Need Before Running This

## Hardware
- CUDA GPU workstation/cluster node — 16GB+ VRAM per job recommended (RoBERTa-base + WavLM
  fused model is the heaviest at ~220M params). Multi-GPU not required; jobs are single-GPU,
  run sequentially or array-parallel across nodes.
- Raspberry Pi 5 (8GB) for Step 8 only — not needed until deployment stage.
- Disk: ~40GB for datasets, ~15GB for all checkpoints (12 configs × 5 seeds), ~5GB scratch.

## Accounts / access needed
- Hugging Face account + token (`huggingface-cli login`) — required to download SLURP-related
  assets and all pretrained encoders (wav2vec2, HuBERT, WavLM, DistilBERT, RoBERTa).
- SLURP dataset: downloaded via the official PolyAI repo (script handles this, no account
  needed, but it's a few GB — see script for mirror links).
- Fluent Speech Commands (FSC): requires filling a short form on the official Fluent.ai page
  to get the download link (free, academic use) — `scripts/01_download_datasets.sh` will stop
  and print instructions if the archive isn't found locally.

## Python / library stack (all installed by `00_setup_env.sh`, listed here for reference)
- Python 3.10+
- torch, torchaudio (CUDA build matching your cluster's CUDA version — check before installing)
- transformers, datasets, accelerate (HuggingFace)
- openai-whisper
- onnx, onnxruntime, onnxruntime-tools, optimum[onnxruntime]
- jiwer (WER computation)
- scikit-learn, scipy, statsmodels (McNemar's test, metrics)
- pandas, numpy, matplotlib
- soundfile, librosa (audio I/O/resampling)
- tqdm, pyyaml, wandb (optional, logging)

## Pretrained model weights (auto-downloaded on first run via HuggingFace, ~2-3GB total)
- `facebook/wav2vec2-base`
- `facebook/hubert-base-ls960`
- `microsoft/wavlm-base-plus`
- `distilbert-base-uncased`
- `roberta-base`
- Whisper `tiny`, `base`, `small` (auto-downloaded by `openai-whisper` on first use, ~1.5GB total)

## Time budget (approximate, single GPU)
- Training: 12 configs × 5 seeds = 60 runs. Rough estimate 1-3 hrs/run depending on dataset
  size and GPU → budget several days if sequential, faster if you can parallelize across nodes.
- Ablations: reuse trained encoders, adds ~10-15% extra time.
- ASR robustness sweep: inference-only, ~1 week per the plan's own estimate (mostly Whisper
  decoding passes across WER conditions).
- Edge deployment: a few hours once models are exported (mostly manual copy + benchmarking).

## Nothing else requires a paid service
Everything above is free/open (HuggingFace models, SLURP, FSC academic license, Whisper).
No API keys or paid compute are hardcoded anywhere in this repo.
