"""
Fusion-level ablation: compares an already-trained probability-level fused
config against its embedding-level variant (trained via
`python3 -m src.train_head ... --embedding_level_fusion`).

Usage:
  python3 -m src.ablate_fusion_level --dataset slurp \
      --audio_encoder facebook/wav2vec2-base --text_encoder roberta-base --seed 42
"""
import argparse
import json
from pathlib import Path

import yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--audio_encoder", required=True)
    ap.add_argument("--text_encoder", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    a, t = args.audio_encoder.split("/")[-1], args.text_encoder.split("/")[-1]
    base_name = f"fused_{a}_{t}"
    emb_name = f"{base_name}_embfusion"

    ckpt_root = Path(cfg["paths"]["checkpoints"]) / args.dataset

    prob_metrics_path = ckpt_root / base_name / f"seed{args.seed}" / "metrics.json"
    emb_metrics_path = ckpt_root / emb_name / f"seed{args.seed}" / "metrics.json"

    if not prob_metrics_path.exists():
        raise FileNotFoundError(f"Probability-level fusion not trained yet: {prob_metrics_path}.")
    if not emb_metrics_path.exists():
        raise FileNotFoundError(f"Embedding-level fusion not trained yet: {emb_metrics_path}.")

    prob_metrics = json.load(open(prob_metrics_path))
    emb_metrics = json.load(open(emb_metrics_path))

    result = {
        "dataset": args.dataset, "config": base_name, "seed": args.seed,
        "probability_level_test_acc": prob_metrics["test_acc"],
        "embedding_level_test_acc": emb_metrics["test_acc"],
        "delta_prob_minus_emb": prob_metrics["test_acc"] - emb_metrics["test_acc"],
    }
    out_dir = Path(cfg["paths"]["results"]) / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"fusion_level_{args.dataset}_{base_name}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
