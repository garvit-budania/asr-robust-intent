"""
Conditional-independence test (Remark 2, Sec. III of the paper): tests whether
the text branch's and audio branch's per-utterance correctness are
independent, via a 2x2 contingency-table chi-square test.

This uses the trained FUSED model's own internal text_branch/audio_branch
(the P_T and P_A referenced in the theory), NOT the separate standalone
text_only/audio_only baseline models -- Remark 2 is specifically about the
two branches as they exist inside the fusion architecture.

This was previously only described as a future check in the paper; this
script is the first actual implementation of it.

Usage:
  python3 -m src.stats.independence_test --dataset slurp \
      --audio_encoder facebook/hubert-base-ls960 --text_encoder distilbert-base-uncased --seed 42
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import chi2_contingency
from torch.utils.data import DataLoader

from src.data.dataset import load_label_map
from src.train_head import CachedEmbeddingDataset
from src.models.head_fusion import HeadOnlyFusionModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True, choices=["slurp", "fsc"])
    ap.add_argument("--audio_encoder", required=True)
    ap.add_argument("--text_encoder", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_root = Path(cfg["datasets"][args.dataset]["root"])
    label_map = load_label_map(ds_root / "label_map.json")
    num_labels = len(label_map)

    test_ds = CachedEmbeddingDataset(args.dataset, "test", label_map, args.audio_encoder, args.text_encoder)
    loader = DataLoader(test_ds, batch_size=256)
    audio_dim = test_ds[0]["audio_embedding"].shape[0]
    text_dim = test_ds[0]["text_embedding"].shape[0]

    a, t = args.audio_encoder.split("/")[-1], args.text_encoder.split("/")[-1]
    cname = f"fused_{a}_{t}"
    ckpt = Path(cfg["paths"]["checkpoints"]) / args.dataset / cname / f"seed{args.seed}" / "best.pt"
    model = HeadOnlyFusionModel(text_dim, audio_dim, num_labels).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    text_correct_list, audio_correct_list = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            text_logits = model.text_branch(batch["text_embedding"])
            audio_logits = model.audio_branch(batch["audio_embedding"])
            text_pred = text_logits.argmax(-1)
            audio_pred = audio_logits.argmax(-1)
            text_correct_list.extend((text_pred == batch["label"]).cpu().numpy().tolist())
            audio_correct_list.extend((audio_pred == batch["label"]).cpu().numpy().tolist())

    text_correct = np.array(text_correct_list, dtype=bool)
    audio_correct = np.array(audio_correct_list, dtype=bool)

    both = int(np.sum(text_correct & audio_correct))
    text_only_correct = int(np.sum(text_correct & ~audio_correct))
    audio_only_correct = int(np.sum(~text_correct & audio_correct))
    neither = int(np.sum(~text_correct & ~audio_correct))
    table = [[both, text_only_correct], [audio_only_correct, neither]]

    chi2, p, dof, expected = chi2_contingency(table, correction=False)

    result = {
        "dataset": args.dataset, "config": cname, "seed": args.seed,
        "contingency_table": {
            "both_correct": both,
            "text_only_correct": text_only_correct,
            "audio_only_correct": audio_only_correct,
            "neither_correct": neither,
        },
        "chi2": float(chi2), "p_value": float(p), "dof": int(dof),
        "n_utterances": len(text_correct),
    }
    out_dir = Path(cfg["paths"]["results"]) / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"independence_{args.dataset}_{cname}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()