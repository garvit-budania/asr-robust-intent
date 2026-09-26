"""
Trains ONLY the lightweight head/fusion parameters over precomputed, cached
embeddings (from src/features/extract_embeddings.py). No large encoder is loaded
or run here at all -- this is what makes each run take seconds-to-minutes
instead of tens of minutes, and is why 5 seeds across all 6 configs is now
affordable again.

Usage:
  python3 -m src.train_head --dataset slurp --model_type fused \
      --audio_encoder facebook/wav2vec2-base --text_encoder distilbert-base-uncased \
      --seed 42

  python3 -m src.train_head --dataset slurp --model_type text_only \
      --text_encoder roberta-base --seed 42

  python3 -m src.train_head --dataset slurp --model_type audio_only \
      --audio_encoder facebook/hubert-base-ls960 --seed 42
"""
import argparse
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import Dataset, DataLoader

from src.data.dataset import load_label_map
from src.models.head_fusion import HeadOnlyFusionModel, HeadOnlyTextModel, HeadOnlyAudioModel


class CachedEmbeddingDataset(Dataset):
    def __init__(self, dataset, split, label_map, audio_encoder=None, text_encoder=None, text_tag=None):
        feat_dir = Path("features") / dataset
        jsonl_path = Path("data") / dataset / f"{split}.jsonl"
        examples = [json.loads(l) for l in open(jsonl_path)]
        self.labels = {ex["utt_id"]: label_map[ex["label"]] for ex in examples}

        self.audio = None
        if audio_encoder:
            a_short = audio_encoder.split("/")[-1]
            data = torch.load(feat_dir / f"{split}_audio_{a_short}.pt")
            self.audio = dict(zip(data["utt_ids"], data["embeddings"]))

        self.text = None
        if text_encoder:
            t_short = text_encoder.split("/")[-1]
            suffix = f"_{text_tag}" if text_tag else ""
            data = torch.load(feat_dir / f"{split}_text_{t_short}{suffix}.pt")
            self.text = dict(zip(data["utt_ids"], data["embeddings"]))

        keysets = [set(self.labels.keys())]
        if self.audio is not None:
            keysets.append(set(self.audio.keys()))
        if self.text is not None:
            keysets.append(set(self.text.keys()))
        self.utt_ids = sorted(set.intersection(*keysets))
        if not self.utt_ids:
            raise RuntimeError(
                f"No overlapping utt_ids found for {dataset}/{split} -- did feature "
                f"extraction (scripts/04_extract_all_features.sh) finish for this "
                f"config? Check features/{dataset}/ for the expected .pt files."
            )

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        uid = self.utt_ids[idx]
        item = {"label": torch.tensor(self.labels[uid], dtype=torch.long)}
        if self.audio is not None:
            item["audio_embedding"] = self.audio[uid]
        if self.text is not None:
            item["text_embedding"] = self.text[uid]
        return item


def evaluate(model, loader, device, model_type):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if model_type == "fused":
                out = model(batch["text_embedding"], batch["audio_embedding"])
                logits = out["fused_logits"]
            elif model_type == "audio_only":
                out = model(batch["audio_embedding"])
                logits = out["logits"]
            else:
                out = model(batch["text_embedding"])
                logits = out["logits"]
            preds = logits.argmax(-1)
            correct += (preds == batch["label"]).sum().item()
            total += batch["label"].size(0)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True, choices=["slurp", "fsc"])
    ap.add_argument("--model_type", required=True, choices=["fused", "text_only", "audio_only"])
    ap.add_argument("--audio_encoder", default=None)
    ap.add_argument("--text_encoder", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=60,
                     help="Head training is cheap -- generous epoch budget is fine.")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--embedding_level_fusion", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_root = Path(cfg["datasets"][args.dataset]["root"])
    label_map = load_label_map(ds_root / "label_map.json")
    num_labels = len(label_map)

    train_ds = CachedEmbeddingDataset(args.dataset, "train", label_map, args.audio_encoder, args.text_encoder)
    dev_ds = CachedEmbeddingDataset(args.dataset, "dev", label_map, args.audio_encoder, args.text_encoder)
    test_ds = CachedEmbeddingDataset(args.dataset, "test", label_map, args.audio_encoder, args.text_encoder)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    dev_loader = DataLoader(dev_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)

    sample = train_ds[0]
    audio_dim = sample["audio_embedding"].shape[0] if "audio_embedding" in sample else None
    text_dim = sample["text_embedding"].shape[0] if "text_embedding" in sample else None

    if args.model_type == "fused":
        model = HeadOnlyFusionModel(text_dim, audio_dim, num_labels,
                                     embedding_level_fusion=args.embedding_level_fusion)
        cname = f"fused_{args.audio_encoder.split('/')[-1]}_{args.text_encoder.split('/')[-1]}"
        if args.embedding_level_fusion:
            cname += "_embfusion"
    elif args.model_type == "text_only":
        model = HeadOnlyTextModel(text_dim, num_labels)
        cname = f"text_only_{args.text_encoder.split('/')[-1]}"
    else:
        model = HeadOnlyAudioModel(audio_dim, num_labels)
        cname = f"audio_only_{args.audio_encoder.split('/')[-1]}"
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_dev_acc, patience_counter = 0.0, 0
    ckpt_dir = Path(cfg["paths"]["checkpoints"]) / args.dataset / cname / f"seed{args.seed}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for batch in train_loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            optimizer.zero_grad()
            if args.model_type == "fused":
                out = model(batch["text_embedding"], batch["audio_embedding"])
            elif args.model_type == "audio_only":
                out = model(batch["audio_embedding"])
            else:
                out = model(batch["text_embedding"])
            loss, _ = model.compute_loss(out, batch["label"])
            loss.backward()
            optimizer.step()

        dev_acc = evaluate(model, dev_loader, device, args.model_type)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"[{cname} seed{args.seed}] epoch {epoch}: dev_acc={dev_acc:.4f}")

    model.load_state_dict(torch.load(ckpt_dir / "best.pt"))
    test_acc = evaluate(model, test_loader, device, args.model_type)
    print(f"[{cname} seed{args.seed}] FINAL test_acc={test_acc:.4f}")

    metrics = {"config": cname, "dataset": args.dataset, "seed": args.seed,
               "best_dev_acc": best_dev_acc, "test_acc": test_acc}
    with open(ckpt_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
