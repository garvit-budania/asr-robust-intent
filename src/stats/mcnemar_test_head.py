"""
McNemar's test (head-only version) between the best and second-best fused
configurations, on pooled per-utterance test predictions -- Sec. V-D.

Usage:
  python3 -m src.stats.mcnemar_test_head --dataset slurp \
      --config_a wav2vec2-base_roberta-base --config_b wavlm-base-plus_roberta-base --seed 42
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml
from statsmodels.stats.contingency_tables import mcnemar
from torch.utils.data import DataLoader

from src.data.dataset import load_label_map
from src.train_head import CachedEmbeddingDataset
from src.models.head_fusion import HeadOnlyFusionModel

FULL_NAME = {
    "wav2vec2-base": "facebook/wav2vec2-base",
    "hubert-base-ls960": "facebook/hubert-base-ls960",
    "wavlm-base-plus": "microsoft/wavlm-base-plus",
    "distilbert-base-uncased": "distilbert-base-uncased",
    "roberta-base": "roberta-base",
}


def get_predictions(audio_short, text_short, dataset, seed, cfg, device):
    ds_root = Path(cfg["datasets"][dataset]["root"])
    label_map = load_label_map(ds_root / "label_map.json")
    num_labels = len(label_map)

    test_ds = CachedEmbeddingDataset(dataset, "test", label_map, FULL_NAME[audio_short], FULL_NAME[text_short])
    loader = DataLoader(test_ds, batch_size=256)
    audio_dim = test_ds[0]["audio_embedding"].shape[0]
    text_dim = test_ds[0]["text_embedding"].shape[0]

    cname = f"fused_{audio_short}_{text_short}"
    ckpt = Path(cfg["paths"]["checkpoints"]) / dataset / cname / f"seed{seed}" / "best.pt"
    model = HeadOnlyFusionModel(text_dim, audio_dim, num_labels).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    correct_flags = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch["text_embedding"], batch["audio_embedding"])
            preds = out["fused_logits"].argmax(-1)
            correct_flags.extend((preds == batch["label"]).cpu().numpy().tolist())
    return np.array(correct_flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--config_a", required=True, help="e.g. wav2vec2-base_roberta-base")
    ap.add_argument("--config_b", required=True, help="e.g. wavlm-base-plus_roberta-base")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Safe: none of our audio/text encoder short-names contain an underscore
    # (they all use hyphens), so there is exactly one underscore in each
    # "<audio>_<text>" string -- the one train_head.py inserted as separator.
    audio_a, text_a = args.config_a.split("_", 1)[0], args.config_a.split("_", 1)[1]
    audio_b, text_b = args.config_b.split("_", 1)[0], args.config_b.split("_", 1)[1]
    # The above assumes the audio name has no underscore; verify against FULL_NAME.
    if audio_a not in FULL_NAME:
        # audio short-name itself might legitimately be the whole first token;
        # re-split more carefully by checking known audio names.
        for cand in FULL_NAME:
            if args.config_a.startswith(cand + "_"):
                audio_a, text_a = cand, args.config_a[len(cand) + 1:]
                break
    if audio_b not in FULL_NAME:
        for cand in FULL_NAME:
            if args.config_b.startswith(cand + "_"):
                audio_b, text_b = cand, args.config_b[len(cand) + 1:]
                break

    correct_a = get_predictions(audio_a, text_a, args.dataset, args.seed, cfg, device)
    correct_b = get_predictions(audio_b, text_b, args.dataset, args.seed, cfg, device)
    assert len(correct_a) == len(correct_b), "Prediction arrays must be the same length"

    both_correct = int(np.sum(correct_a & correct_b))
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    both_wrong = int(np.sum(~correct_a & ~correct_b))
    table = [[both_correct, a_only], [b_only, both_wrong]]

    result = mcnemar(table, exact=True)
    out = {
        "dataset": args.dataset, "config_a": args.config_a, "config_b": args.config_b,
        "contingency_table": table,
        "discordant_pairs": {"a_only_correct": a_only, "b_only_correct": b_only},
        "statistic": float(result.statistic), "p_value": float(result.pvalue),
    }
    out_dir = Path(cfg["paths"]["results"]) / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mcnemar_{args.dataset}_{args.config_a}_vs_{args.config_b}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
