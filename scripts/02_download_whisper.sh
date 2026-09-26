#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

python3 - <<'PYEOF'
import whisper
for size in ["tiny", "base", "small"]:
    print(f"Downloading Whisper {size}...")
    whisper.load_model(size)
print("All Whisper models cached.")
PYEOF
