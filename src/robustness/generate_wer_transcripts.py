"""
Generates transcript sets at controlled word-error-rate (WER) conditions for the
robustness experiment. Primary method: Whisper decoding at three sizes (natural
WER variation). Supplement: synthetic word-level corruption to hit exact target
WER levels (0/10/20/30%) -- this is what src/features/extract_embeddings.py
consumes for the robustness sweep.

Usage:
  python3 -m src.robustness.generate_wer_transcripts --dataset slurp --split test
"""
import argparse
import json
import random
from pathlib import Path

import jiwer
import whisper
import yaml
from tqdm import tqdm


def whisper_transcribe_all(jsonl_path, whisper_sizes):
    examples = [json.loads(l) for l in open(jsonl_path)]
    results = {}
    for size in whisper_sizes:
        print(f"Loading Whisper {size}...")
        model = whisper.load_model(size)
        hyps, refs = [], []
        for ex in tqdm(examples, desc=f"whisper-{size}"):
            result = model.transcribe(ex["audio_path"], fp16=False)
            hyps.append(result["text"].strip())
            refs.append(ex["transcript"].strip())
        realized_wer = jiwer.wer(refs, hyps)
        results[size] = {"hyps": hyps, "refs": refs, "realized_wer": realized_wer}
        print(f"Whisper {size}: realized WER = {realized_wer:.3f}")
        del model
    return examples, results


COMMON_SUBS = {
    "to": "too", "for": "four", "there": "their", "navigate": "nervate",
    "play": "pray", "music": "muse ick", "call": "cal", "turn": "torn",
}


def corrupt_transcript(text, target_wer, rng):
    words = text.split()
    if not words:
        return text
    n_errors = max(0, round(len(words) * target_wer))
    indices = rng.sample(range(len(words)), min(n_errors, len(words)))
    for idx in indices:
        op = rng.choice(["sub", "del", "ins"])
        w = words[idx].lower()
        if op == "sub":
            words[idx] = COMMON_SUBS.get(w, rng.choice(words) if len(words) > 1 else w)
        elif op == "del":
            words[idx] = ""
        elif op == "ins":
            words[idx] = words[idx] + " " + rng.choice(list(COMMON_SUBS.values()))
    return " ".join(w for w in words if w)


def synthetic_corruption_set(jsonl_path, target_wers, seed=42):
    rng = random.Random(seed)
    examples = [json.loads(l) for l in open(jsonl_path)]
    out = {}
    for wer in target_wers:
        corrupted = [corrupt_transcript(ex["transcript"], wer, rng) for ex in examples]
        refs = [ex["transcript"] for ex in examples]
        realized = jiwer.wer(refs, corrupted)
        out[wer] = {"hyps": corrupted, "refs": refs, "realized_wer": realized}
        print(f"Synthetic target WER {wer:.2f} -> realized {realized:.3f}")
    return examples, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True, choices=["slurp", "fsc"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--skip_whisper", action="store_true",
                     help="Skip the (slower) Whisper pass; only produce synthetic corruption. "
                          "Whisper's own decoded transcripts are not consumed by the head-only "
                          "pipeline, only the synthetic set is, so this is safe to use to save time.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    rcfg = cfg["robustness"]

    jsonl_path = Path(cfg["datasets"][args.dataset]["root"]) / f"{args.split}.jsonl"
    out_dir = Path(cfg["paths"]["results"]) / "robustness" / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_whisper:
        print("== Whisper decoding across sizes (for realized-WER reference only) ==")
        examples, whisper_results = whisper_transcribe_all(jsonl_path, rcfg["whisper_sizes"])
        with open(out_dir / f"{args.dataset}_{args.split}_whisper.json", "w") as f:
            json.dump({size: {"realized_wer": r["realized_wer"], "hyps": r["hyps"]}
                       for size, r in whisper_results.items()}, f, indent=2)

    if rcfg["synthetic_corruption_supplement"]:
        print("== Synthetic corruption at exact target WERs (used by the head-only pipeline) ==")
        _, synth_results = synthetic_corruption_set(jsonl_path, rcfg["wer_targets"])
        with open(out_dir / f"{args.dataset}_{args.split}_synthetic.json", "w") as f:
            json.dump({str(wer): {"realized_wer": r["realized_wer"], "hyps": r["hyps"]}
                       for wer, r in synth_results.items()}, f, indent=2)

    print(f"Transcript sets written to {out_dir}")


if __name__ == "__main__":
    main()
