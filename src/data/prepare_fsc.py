"""
Prepare Fluent Speech Commands (FSC) into (audio_path, transcript, label) JSONL splits.

FSC ships with train_data.csv / valid_data.csv / test_data.csv, each row containing
path, transcription, action, object, location. We combine action+object+location into
a single intent label (the standard FSC intent-classification setup: 31 unique combos).

Expected input layout (after extracting the archive):
  data/fsc/fluent_speech_commands_dataset/{train_data.csv, valid_data.csv, test_data.csv, wavs/...}

Output:
  data/fsc/{train,dev,test}.jsonl
  data/fsc/label_map.json
"""
import argparse
import csv
import json
from pathlib import Path

import yaml
from tqdm import tqdm


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def process_split(root, csv_name, out_path, split_name):
    csv_path = root / csv_name
    if not csv_path.exists():
        print(f"!! Missing {csv_path} -- did you download+extract FSC? Skipping {split_name}.")
        return set()

    labels = set()
    with open(csv_path) as fin, open(out_path, "w") as fout:
        reader = csv.DictReader(fin)
        for row in tqdm(reader, desc=f"FSC {split_name}"):
            label = f"{row['action']}_{row['object']}_{row['location']}"
            labels.add(label)
            audio_path = str(root / row["path"])
            out = {
                "audio_path": audio_path,
                "transcript": row["transcription"],
                "label": label,
                "utt_id": Path(row["path"]).stem,
            }
            fout.write(json.dumps(out) + "\n")
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    fsc_cfg = cfg["datasets"]["fsc"]

    root = Path(fsc_cfg["root"]) / "fluent_speech_commands_dataset"
    out_root = Path(fsc_cfg["root"])
    out_root.mkdir(parents=True, exist_ok=True)

    split_map = {"train": "train_data.csv", "dev": "valid_data.csv", "test": "test_data.csv"}
    all_labels = set()
    for split_name, fname in split_map.items():
        out_path = out_root / f"{split_name}.jsonl"
        labels = process_split(root, fname, out_path, split_name)
        all_labels |= labels

    if all_labels:
        label_map = {lab: i for i, lab in enumerate(sorted(all_labels))}
        with open(out_root / "label_map.json", "w") as f:
            json.dump(label_map, f, indent=2)
        print(f"FSC label map: {len(label_map)} classes -> data/fsc/label_map.json")


if __name__ == "__main__":
    main()
