"""
Compute REAL audio duration percentiles from the actual prepared datasets, so
max_audio_seconds in configs/experiment.yaml can be right-sized instead of guessed.
Uses soundfile.info() to read duration from file headers only -- does not decode
full audio, so this runs fast even across tens of thousands of files.

Usage:
  python3 src/data/derive_audio_len.py --config configs/experiment.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def durations_for(jsonl_path, sample_n=5000):
    """Sample up to sample_n entries (fast) rather than every single file if the
    dataset is huge -- percentiles from a few thousand samples are already stable."""
    examples = [json.loads(l) for l in open(jsonl_path)]
    if len(examples) > sample_n:
        rng = np.random.default_rng(42)
        examples = [examples[i] for i in rng.choice(len(examples), sample_n, replace=False)]
    durs = []
    for ex in tqdm(examples, desc=f"scanning {jsonl_path.name}"):
        try:
            info = sf.info(ex["audio_path"])
            durs.append(info.frames / info.samplerate)
        except Exception:
            continue
    return durs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    for ds_key in ["slurp", "fsc"]:
        train_path = Path(cfg["datasets"][ds_key]["root"]) / "train.jsonl"
        if not train_path.exists():
            print(f"!! {train_path} not found, skipping {ds_key}")
            continue
        durs = durations_for(train_path)
        if not durs:
            continue
        durs = np.array(durs)
        p50, p90, p99, mx = np.percentile(durs, 50), np.percentile(durs, 90), np.percentile(durs, 99), durs.max()
        current = cfg["datasets"][ds_key]["max_audio_seconds"]
        recommended = float(np.ceil(p99))
        print(f"\n[{ds_key}] duration stats (seconds): "
              f"p50={p50:.2f} p90={p90:.2f} p99={p99:.2f} max={mx:.2f}")
        print(f"[{ds_key}] current config max_audio_seconds = {current}")
        print(f"[{ds_key}] RECOMMENDED max_audio_seconds = {recommended}  "
              f"(covers 99% of utterances, drop the rest instead of padding everything to worst case)")


if __name__ == "__main__":
    main()
