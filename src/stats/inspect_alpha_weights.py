"""
Inspect learned alpha1 (text weight) / alpha2 (audio weight) / alpha3 (fusion-
branch weight) across all trained fused checkpoints. These live ONLY inside
each model's state_dict (key: alpha_weights.logits) -- they are never written
to any results JSON, so this script is the only way to see them.

This directly answers: has the model learned to trust one branch almost
entirely (e.g. alpha1 near 1, alpha2 near 0), which would explain fused
accuracy tracking text-only closely on a dataset where text-only already
dominates (e.g. SLURP, where audio-only alone is far weaker than text-only)?

Usage:
  python3 -m src.stats.inspect_alpha_weights
"""
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

with open("configs/experiment.yaml") as f:
    cfg = yaml.safe_load(f)

ckpt_root = Path(cfg["paths"]["checkpoints"])
rows = []

for ckpt_path in sorted(ckpt_root.glob("*/fused_*/seed*/best.pt")):
    parts = ckpt_path.parts
    # .../checkpoints/<dataset>/<config>/seed<N>/best.pt
    dataset = parts[-4]
    config = parts[-3]
    seed = parts[-2].replace("seed", "")

    sd = torch.load(ckpt_path, map_location="cpu")
    key = "alpha_weights.logits"
    if key not in sd:
        print(f"!! {ckpt_path} has no {key} -- skipping (unexpected checkpoint format).")
        continue

    weights = F.softmax(sd[key], dim=0)
    a1, a2, a3 = weights.tolist()
    rows.append({
        "dataset": dataset, "config": config, "seed": seed,
        "alpha1_text": a1, "alpha2_audio": a2, "alpha3_fusion_branch": a3,
    })
    print(f"{dataset:6s} {config:55s} seed{seed:>4s}  "
          f"alpha1(text)={a1:.4f}  alpha2(audio)={a2:.4f}  alpha3(fusion)={a3:.4f}")

out_dir = Path(cfg["paths"]["results"]) / "stats"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "alpha_weights.json"
with open(out_path, "w") as f:
    json.dump(rows, f, indent=2)
print(f"\nSaved -> {out_path}")

if rows:
    print("\n-- Per-dataset means (across all fused checkpoints found) --")
    for ds in sorted(set(r["dataset"] for r in rows)):
        ds_rows = [r for r in rows if r["dataset"] == ds]
        mean_a1 = statistics.mean(r["alpha1_text"] for r in ds_rows)
        mean_a2 = statistics.mean(r["alpha2_audio"] for r in ds_rows)
        mean_a3 = statistics.mean(r["alpha3_fusion_branch"] for r in ds_rows)
        print(f"{ds}: mean alpha1(text)={mean_a1:.4f}  mean alpha2(audio)={mean_a2:.4f}  "
              f"mean alpha3(fusion)={mean_a3:.4f}  (n={len(ds_rows)} checkpoints)")
else:
    print("No fused checkpoints found under", ckpt_root)