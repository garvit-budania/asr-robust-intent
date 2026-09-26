"""
One-time frozen feature extraction: runs each audio/text encoder over each
dataset split ONCE (no gradients, no training) and caches pooled embeddings to
disk. This is the core of the efficient approach -- see chat/paper notes for the
reasoning. After this, src/train_head.py trains ONLY small fusion heads over
these cached, fixed-size embeddings, which is what makes training take minutes
instead of hours.

Usage (run once per audio encoder x dataset x split, and once per text encoder
x dataset x split -- see scripts/04_extract_all_features.sh for the full sweep):

  python3 -m src.features.extract_embeddings --dataset slurp \
      --encoder_type audio --encoder_name facebook/wav2vec2-base --split train

  python3 -m src.features.extract_embeddings --dataset slurp \
      --encoder_type text --encoder_name roberta-base --split train

For the ASR-robustness sweep, text embeddings can be extracted from CORRUPTED
transcripts instead of the originals by pointing --corrupted_transcripts_json at
one of the WER transcript files produced by src/robustness/generate_wer_transcripts.py,
and giving a --tag (e.g. wer10) so it doesn't overwrite the clean-transcript cache.
Audio embeddings never need to be re-extracted for the robustness sweep -- audio
is unaffected by ASR corruption, which is exactly the point.
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
from tqdm import tqdm

from src.data.dataset import IntentDataset, load_label_map


def extract_audio(cfg, dataset, split, encoder_name, device):
    ds_root = Path(cfg["datasets"][dataset]["root"])
    label_map = load_label_map(ds_root / "label_map.json")
    feat_extractor = AutoFeatureExtractor.from_pretrained(encoder_name)
    dummy_tokenizer = AutoTokenizer.from_pretrained("roberta-base")  # text side unused here

    max_len_path = Path("data/derived_max_len.json")
    max_len = json.load(open(max_len_path))["max_text_len"] if max_len_path.exists() else 128

    ds = IntentDataset(ds_root / f"{split}.jsonl", label_map, dummy_tokenizer, feat_extractor,
                        max_text_len=max_len, max_audio_seconds=cfg["datasets"][dataset]["max_audio_seconds"],
                        cache_audio=False)  # single pass -- no benefit to caching here
    loader = DataLoader(ds, batch_size=32, num_workers=2, shuffle=False)

    encoder = AutoModel.from_pretrained(encoder_name).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    all_embeddings, all_utt_ids = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"audio:{encoder_name}:{dataset}:{split}"):
            input_values = batch["input_values"].to(device)
            out = encoder(input_values=input_values)
            pooled = out.last_hidden_state.mean(dim=1).cpu()
            all_embeddings.append(pooled)
            all_utt_ids.extend(batch["utt_id"])

    embeddings = torch.cat(all_embeddings, dim=0)
    out_dir = Path("features") / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    short_name = encoder_name.split("/")[-1]
    out_path = out_dir / f"{split}_audio_{short_name}.pt"
    torch.save({"utt_ids": all_utt_ids, "embeddings": embeddings}, out_path)
    print(f"Saved {tuple(embeddings.shape)} -> {out_path}")


def extract_text(cfg, dataset, split, encoder_name, device, corrupted_transcripts_json=None, wer_key=None, tag=None):
    ds_root = Path(cfg["datasets"][dataset]["root"])
    examples = [json.loads(l) for l in open(ds_root / f"{split}.jsonl")]

    transcripts_override = None
    if corrupted_transcripts_json:
        # generate_wer_transcripts.py's synthetic-corruption output holds ALL WER
        # levels in one file, keyed by WER value as a string (e.g. "0.1"). Select
        # the right sub-entry with --wer_key; otherwise assume the file is a
        # single flat {"hyps": [...]} structure.
        data = json.load(open(corrupted_transcripts_json))
        if wer_key is not None:
            data = data[wer_key]
        hyps = data["hyps"] if isinstance(data, dict) and "hyps" in data else data
        transcripts_override = {ex["utt_id"]: hyp for ex, hyp in zip(examples, hyps)}

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    max_len_path = Path("data/derived_max_len.json")
    max_len = json.load(open(max_len_path))["max_text_len"] if max_len_path.exists() else 128

    encoder = AutoModel.from_pretrained(encoder_name).to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    all_embeddings, all_utt_ids = [], []
    batch_size = 64
    with torch.no_grad():
        for i in tqdm(range(0, len(examples), batch_size), desc=f"text:{encoder_name}:{dataset}:{split}"):
            batch_examples = examples[i:i + batch_size]
            texts = [
                transcripts_override[ex["utt_id"]] if transcripts_override else ex["transcript"]
                for ex in batch_examples
            ]
            enc = tokenizer(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
            out = encoder(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu()
            all_embeddings.append(cls)
            all_utt_ids.extend([ex["utt_id"] for ex in batch_examples])

    embeddings = torch.cat(all_embeddings, dim=0)
    out_dir = Path("features") / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    short_name = encoder_name.split("/")[-1]
    suffix = f"_{tag}" if tag else ""
    out_path = out_dir / f"{split}_text_{short_name}{suffix}.pt"
    torch.save({"utt_ids": all_utt_ids, "embeddings": embeddings}, out_path)
    print(f"Saved {tuple(embeddings.shape)} -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True, choices=["slurp", "fsc"])
    ap.add_argument("--encoder_type", required=True, choices=["audio", "text"])
    ap.add_argument("--encoder_name", required=True)
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--corrupted_transcripts_json", default=None,
                     help="For the robustness sweep: path to a corrupted-transcript file. Text only.")
    ap.add_argument("--wer_key", default=None,
                     help="Which WER level to select from the multi-WER json (e.g. '0.1'). Text only.")
    ap.add_argument("--tag", default=None, help="Suffix for the output filename, e.g. wer10.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.encoder_type == "audio":
        extract_audio(cfg, args.dataset, args.split, args.encoder_name, device)
    else:
        extract_text(cfg, args.dataset, args.split, args.encoder_name, device,
                      args.corrupted_transcripts_json, args.wer_key, args.tag)


if __name__ == "__main__":
    main()
