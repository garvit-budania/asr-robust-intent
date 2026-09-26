#!/usr/bin/env bash
# NOTE: if you already have a working conda env (torch + CUDA confirmed working),
# you almost certainly do NOT need to run this script again -- just activate it
# and run `pip install -r requirements.txt` for anything new.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== Creating conda environment =="
conda create -n asr_intent python=3.11 -y
echo "Activate with: conda activate asr_intent"

echo "== Install PyTorch matching your CUDA version (check nvidia-smi first) =="
echo "Example for CUDA 12.4:"
echo "  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 --no-cache-dir"
echo "Install torch BEFORE requirements.txt, in its own command, matched to your actual CUDA version."

echo "== Then install the rest =="
echo "  pip install -r requirements.txt"

echo "== Hugging Face login (needed once, for model + dataset downloads) =="
echo "Run: huggingface-cli login"
