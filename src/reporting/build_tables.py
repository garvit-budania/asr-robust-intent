"""
Assembles Table II and Table III of the paper from whatever results exist on
disk. Run this LAST. Never fills a cell with a placeholder -- missing results
show as "PENDING" so you can see at a glance what's left to run.

NOTE: the paper no longer includes an edge-deployment table (that section was
removed) -- this script only builds the two tables that matter now: encoder-grid
accuracy and ASR-robustness curve.

Usage:
  python3 src/reporting/build_tables.py
"""
import json
from pathlib import Path

import pandas as pd
import yaml

with open("configs/experiment.yaml") as f:
    cfg = yaml.safe_load(f)

results_dir = Path(cfg["paths"]["results"])
out_path = results_dir / "paper_tables.md"
lines = []

# ---------- Table II: encoder-grid accuracy ----------
lines.append("## Table II: Intent Accuracy (mean +/- std over 5 seeds)\n")
seed_csv = results_dir / "stats" / "seed_aggregated_accuracy.csv"
if seed_csv.exists():
    df = pd.read_csv(seed_csv)
    lines.append("| System | FSC | SLURP |")
    lines.append("|---|---|---|")
    configs_of_interest = [
        "text_only_roberta-base", "audio_only_hubert-base-ls960",
        "fused_wav2vec2-base_distilbert-base-uncased", "fused_wav2vec2-base_roberta-base",
        "fused_hubert-base-ls960_distilbert-base-uncased", "fused_hubert-base-ls960_roberta-base",
        "fused_wavlm-base-plus_distilbert-base-uncased", "fused_wavlm-base-plus_roberta-base",
    ]
    for c in configs_of_interest:
        row = [c]
        for ds in ["fsc", "slurp"]:
            match = df[(df["config"] == c) & (df["dataset"] == ds)]
            if len(match):
                m, s = match.iloc[0]["mean_acc"], match.iloc[0]["std_acc"]
                row.append(f"{m:.3f} +/- {s:.3f}")
            else:
                row.append("PENDING")
        lines.append("| " + " | ".join(row) + " |")
else:
    lines.append("PENDING -- run scripts/05_train_all_heads.sh then src/stats/aggregate_seeds.py\n")

lines.append("\n")

# ---------- Table III: robustness curve ----------
lines.append("## Table III: Accuracy under Transcript Corruption (SLURP)\n")
robustness_json = results_dir / "robustness" / "slurp_robustness_curve.json"
if robustness_json.exists():
    rows = json.load(open(robustness_json))
    lines.append("| WER | Text-only RoBERTa | Best fused | Audio-only floor |")
    lines.append("|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['wer_target']*100:.0f}% | {r['text_only_acc']:.3f} | "
                      f"{r['fused_acc']:.3f} | {r['audio_only_floor']:.3f} |")
else:
    lines.append("PENDING -- run scripts/07_run_robustness_head.sh\n")

lines.append("\n")

# ---------- Ablations summary ----------
lines.append("## Ablation Summary\n")
ablation_dir = results_dir / "ablations"
if ablation_dir.exists() and any(ablation_dir.glob("calibration_*.json")):
    cal_files = list(ablation_dir.glob("calibration_*.json"))
    deltas = [json.load(open(f))["accuracy_delta"] for f in cal_files]
    lines.append(f"- Calibration ablation: {len(cal_files)} runs, "
                  f"mean accuracy delta (calibrated - uncalibrated) = {sum(deltas)/len(deltas):+.4f}")
else:
    lines.append("- Calibration ablation: PENDING")

if ablation_dir.exists() and any(ablation_dir.glob("fusion_level_*.json")):
    fus_files = list(ablation_dir.glob("fusion_level_*.json"))
    deltas = [json.load(open(f))["delta_prob_minus_emb"] for f in fus_files]
    lines.append(f"- Fusion-level ablation: {len(fus_files)} runs, "
                  f"mean (probability-level minus embedding-level) = {sum(deltas)/len(deltas):+.4f}")
else:
    lines.append("- Fusion-level ablation: PENDING")

with open(out_path, "w") as f:
    f.write("\n".join(lines))

print("\n".join(lines))
print(f"\nWritten -> {out_path}")
