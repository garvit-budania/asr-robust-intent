"""
Re-derive the max text token length from the ACTUAL prepared datasets. Uses the
99th percentile of tokenized transcript lengths across train splits (both
datasets combined). Writes the result into data/derived_max_len.json.

Usage:
  python3 src/data/derive_max_len.py --config configs/experiment.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from transformers import AutoTokenizer


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--tokenizer", default="roberta-base",
                     help="Reference tokenizer used only to measure length distribution.")
    args = ap.parse_args()
    cfg = load_config(args.config)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    lengths = []

    for ds_key in ["slurp", "fsc"]:
        train_path = Path(cfg["datasets"][ds_key]["root"]) / "train.jsonl"
        if not train_path.exists():
            print(f"!! {train_path} not found yet, skipping in length derivation.")
            continue
        with open(train_path) as f:
            for line in f:
                obj = json.loads(line)
                n = len(tok.encode(obj["transcript"], add_special_tokens=True))
                lengths.append(n)

    ceiling = cfg.get("max_text_len_ceiling", 128)
    if not lengths:
        print(f"No data found yet -- using config ceiling {ceiling} as placeholder.")
        derived = ceiling
    else:
        p99 = int(np.percentile(lengths, 99))
        derived = min(p99, ceiling)
        print(f"Token length stats: mean={np.mean(lengths):.1f}, p99={p99}, "
              f"max={max(lengths)}, using max_len={derived}")

    with open("data/derived_max_len.json", "w") as f:
        json.dump({"max_text_len": derived}, f, indent=2)


if __name__ == "__main__":
    main()
