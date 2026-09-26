"""
Prepare SLURP into (audio_path, transcript, label) JSONL splits.

SLURP's raw annotation format is JSONL with nested 'recordings' and 'sentence_annotation'
fields; this script flattens it into a single flat JSONL per split, at the intent
granularity chosen in configs/experiment.yaml (scenario / action / scenario_action).

Expected input layout (from the official SLURP repo, after download_audio.sh):
  data/slurp/raw/dataset/slurp/{train,devel,test}.jsonl   -- annotations
  data/slurp/raw/audio/slurp_real/*.flac                  -- real recordings
  data/slurp/raw/audio/slurp_synth/*.flac                 -- synthetic recordings

NOTE: the audio lives directly under data/slurp/raw/audio/, NOT under
data/slurp/raw/dataset/slurp/audio/ -- these are two separate subtrees of the
same cloned repo. This script keeps them as two separate roots (raw_root for
annotations, audio_root for audio) rather than assuming one is nested in the other.

Output:
  data/slurp/{train,dev,test}.jsonl   -- one JSON object per line:
      {"audio_path": ..., "transcript": ..., "label": ..., "utt_id": ...}
  data/slurp/label_map.json           -- label string -> integer id
"""
import argparse
import json
import os
from pathlib import Path

import yaml
from tqdm import tqdm


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_label(entry, granularity):
    scenario = entry.get("scenario", "unknown")
    action = entry.get("action", "unknown")
    if granularity == "scenario":
        return scenario
    elif granularity == "action":
        return action
    elif granularity == "scenario_action":
        return f"{scenario}_{action}"
    else:
        raise ValueError(f"Unknown granularity: {granularity}")


def find_audio_path(audio_root, recording_file):
    """audio_root is data/slurp/raw/audio -- audio files sit flat inside
    slurp_real/ and slurp_synth/ subfolders directly under it."""
    for sub in ["slurp_real", "slurp_synth"]:
        candidate = audio_root / sub / recording_file
        if candidate.exists():
            return str(candidate)
    return None


def process_split(raw_root, audio_root, split_file, out_path, granularity, split_name):
    n_written, n_skipped = 0, 0
    with open(split_file) as fin, open(out_path, "w") as fout:
        for line in tqdm(fin, desc=f"SLURP {split_name}"):
            entry = json.loads(line)
            label = resolve_label(entry, granularity)
            for rec in entry.get("recordings", []):
                audio_path = find_audio_path(audio_root, rec["file"])
                if audio_path is None:
                    n_skipped += 1
                    continue
                transcript = entry.get("sentence", "")
                out = {
                    "audio_path": audio_path,
                    "transcript": transcript,
                    "label": label,
                    "utt_id": rec["file"],
                }
                fout.write(json.dumps(out) + "\n")
                n_written += 1
    print(f"[{split_name}] wrote {n_written}, skipped {n_skipped} (missing audio)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    slurp_cfg = cfg["datasets"]["slurp"]
    granularity = slurp_cfg["granularity"]

    root = Path(slurp_cfg["root"])
    raw_root = root / "raw" / "dataset" / "slurp"   # annotations live here
    audio_root = root / "raw" / "audio"              # audio lives here (sibling subtree, NOT nested under raw_root)
    out_root = root
    out_root.mkdir(parents=True, exist_ok=True)

    if not audio_root.exists():
        print(f"!! {audio_root} does not exist -- did you run download_audio.sh and move/extract "
              f"the audio there? See README troubleshooting notes.")

    split_map = {"train": "train.jsonl", "dev": "devel.jsonl", "test": "test.jsonl"}
    all_labels = set()

    for split_name, fname in split_map.items():
        split_file = raw_root / fname
        if not split_file.exists():
            print(f"!! Missing {split_file} -- did scripts/01_download_datasets.sh finish? Skipping {split_name}.")
            continue
        out_path = out_root / f"{split_name}.jsonl"
        process_split(raw_root, audio_root, split_file, out_path, granularity, split_name)
        with open(out_path) as f:
            for line in f:
                all_labels.add(json.loads(line)["label"])

    label_map = {lab: i for i, lab in enumerate(sorted(all_labels))}
    with open(out_root / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"SLURP label map ({granularity}-level): {len(label_map)} classes -> data/slurp/label_map.json")


if __name__ == "__main__":
    main()
