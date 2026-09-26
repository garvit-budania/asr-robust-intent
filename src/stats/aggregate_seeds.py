"""
Scans checkpoints/<dataset>/<config>/seed*/metrics.json and produces mean +/- std
test accuracy per (dataset, config) -- Table II of the paper.

Usage:
  python3 -m src.stats.aggregate_seeds
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

with open("configs/experiment.yaml") as f:
    cfg = yaml.safe_load(f)

ckpt_root = Path(cfg["paths"]["checkpoints"])
results = defaultdict(list)

for metrics_file in ckpt_root.glob("*/*/seed*/metrics.json"):
    m = json.load(open(metrics_file))
    key = (m["dataset"], m["config"])
    results[key].append(m["test_acc"])

rows = []
for (dataset, config), accs in sorted(results.items()):
    rows.append({
        "dataset": dataset, "config": config, "n_seeds": len(accs),
        "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
        "all_accs": accs,
    })

df = pd.DataFrame(rows)
out_dir = Path(cfg["paths"]["results"]) / "stats"
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "seed_aggregated_accuracy.csv", index=False)
print(df[["dataset", "config", "n_seeds", "mean_acc", "std_acc"]].to_string(index=False))
print(f"\nSaved -> {out_dir / 'seed_aggregated_accuracy.csv'}")

if any(r["n_seeds"] < 5 for r in rows):
    print("\n!! NOTE: some configs have fewer than 5 seeds.")
