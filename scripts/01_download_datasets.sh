#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

mkdir -p data/slurp data/fsc

echo "== SLURP =="
if [ ! -d "data/slurp/raw" ]; then
    echo "Cloning official SLURP repo (audio + annotations)..."
    git clone https://github.com/pswietojanski/slurp.git data/slurp/raw
    echo "Running the real audio download script (this is large, several GB)..."
    if [ -f "data/slurp/raw/scripts/download_audio.sh" ]; then
        (cd data/slurp/raw/scripts && bash download_audio.sh)
        # NOTE: the download script extracts audio to data/slurp/raw/scripts/audio/ --
        # our prepare_slurp.py expects it at data/slurp/raw/audio/. Move it up one level:
        if [ -d "data/slurp/raw/scripts/audio" ] && [ ! -d "data/slurp/raw/audio" ]; then
            mv data/slurp/raw/scripts/audio data/slurp/raw/audio
        fi
    else
        echo "!! download_audio.sh not found at the expected path -- check the cloned repo's"
        echo "   scripts/ directory structure, it may have changed."
    fi
else
    echo "SLURP raw already present, skipping."
fi

echo "== Fluent Speech Commands (FSC) =="
if [ ! -f "data/fsc/fluent_speech_commands_dataset.tar.gz" ] && [ ! -f "data/fsc/fluent-speech-corpus.zip" ]; then
    cat <<'EOF'
FSC download options:
  A) Official site (needs a short form): https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/
     -> place the result at data/fsc/fluent_speech_commands_dataset.tar.gz
  B) Kaggle mirror (no form, needs a free Kaggle account + API token):
       pip install kaggle
       kaggle datasets download -d tommyngx/fluent-speech-corpus -p data/fsc/
     -> produces data/fsc/fluent-speech-corpus.zip
Re-run this script once either file is present -- it will extract automatically.
EOF
else
    if [ -f "data/fsc/fluent_speech_commands_dataset.tar.gz" ]; then
        echo "Extracting FSC (tar.gz)..."
        tar -xzf data/fsc/fluent_speech_commands_dataset.tar.gz -C data/fsc/
    elif [ -f "data/fsc/fluent-speech-corpus.zip" ]; then
        echo "Extracting FSC (Kaggle zip)..."
        (cd data/fsc && unzip -o fluent-speech-corpus.zip)
        # Kaggle mirror nests the CSVs one level deeper than expected -- flatten if needed:
        if [ -f "data/fsc/fluent_speech_commands_dataset/data/train_data.csv" ]; then
            mv data/fsc/fluent_speech_commands_dataset/data/*.csv data/fsc/fluent_speech_commands_dataset/
        fi
    fi
fi

echo "== Preparing splits =="
python3 src/data/prepare_slurp.py --config configs/experiment.yaml
python3 src/data/prepare_fsc.py --config configs/experiment.yaml

echo "== Deriving max token length from actual data =="
python3 src/data/derive_max_len.py --config configs/experiment.yaml

echo "== Datasets ready. =="
echo "NEXT: run 'python3 src/data/derive_audio_len.py' and update configs/experiment.yaml's"
echo "max_audio_seconds values before extraction -- this materially affects runtime."
