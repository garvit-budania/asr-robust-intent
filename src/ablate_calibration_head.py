"""
Calibration ablation (head-only version) -- Sec. IV-D / Step 5. Loads a trained
HeadOnlyFusionModel checkpoint and cached embeddings, fits a temperature on the
audio branch's dev-set logits, and compares fused test accuracy/ECE with vs.
without calibration. Same math as the original src/ablate_calibration.py, just
running on cached embeddings + the head-only model instead of raw audio/text.

Usage:
  python3 -m src.ablate_calibration_head --dataset slurp \
      --audio_encoder facebook/wav2vec2-base --text_encoder roberta-base --seed 42
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import load_label_map
from src.train_head import CachedEmbeddingDataset
from src.models.head_fusion import HeadOnlyFusionModel


def expected_calibration_error(probs, labels, n_bins=15):
    confidences, predictions = probs.max(dim=-1)
    accuracies = predictions.eq(labels)
    ece = torch.zeros(1)
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        in_bin = (confidences > lo) & (confidences <= hi)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin
    return ece.item()


def get_audio_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logits = model.audio_branch(batch["audio_embedding"])
            all_logits.append(logits.cpu())
            all_labels.append(batch["label"].cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


def get_fused_logits(model, loader, device, temp_scaler=None):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            text_logits = model.text_branch(batch["text_embedding"])
            audio_logits = model.audio_branch(batch["audio_embedding"])
            if temp_scaler is not None:
                audio_logits = temp_scaler(audio_logits)

            p_text = F.softmax(text_logits, dim=-1)
            p_audio = F.softmax(audio_logits, dim=-1)
            fusion_in = torch.cat([p_text, p_audio], dim=-1)
            h = F.relu(model.fusion_fc1(fusion_in))
            fused_logits = model.fusion_fc2(h)

            all_logits.append(fused_logits.cpu())
            all_labels.append(batch["label"].cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_root = Path(cfg["datasets"][args.dataset]["root"])
    label_map = load_label_map(ds_root / "label_map.json")
    num_labels = len(label_map)

    dev_ds = CachedEmbeddingDataset(args.dataset, "dev", label_map, args.audio_encoder, args.text_encoder)
    test_ds = CachedEmbeddingDataset(args.dataset, "test", label_map, args.audio_encoder, args.text_encoder)
    dev_loader = DataLoader(dev_ds, batch_size=256)
    test_loader = DataLoader(test_ds, batch_size=256)

    audio_dim = dev_ds[0]["audio_embedding"].shape[0]
    text_dim = dev_ds[0]["text_embedding"].shape[0]

    a, t = args.audio_encoder.split("/")[-1], args.text_encoder.split("/")[-1]
    cname = f"fused_{a}_{t}"
    ckpt_path = Path(cfg["paths"]["checkpoints"]) / args.dataset / cname / f"seed{args.seed}" / "best.pt"
    model = HeadOnlyFusionModel(text_dim, audio_dim, num_labels).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    dev_audio_logits, dev_labels = get_audio_logits(model, dev_loader, device)
    temp_scaler = model.audio_branch.temp_scaler
    fitted_temp = temp_scaler.fit(dev_audio_logits.to(device), dev_labels.to(device))

    uncal_logits, test_labels = get_fused_logits(model, test_loader, device, temp_scaler=None)
    uncal_probs = F.softmax(uncal_logits, dim=-1)
    uncal_acc = (uncal_probs.argmax(-1) == test_labels).float().mean().item()
    uncal_ece = expected_calibration_error(uncal_probs, test_labels)

    cal_logits, _ = get_fused_logits(model, test_loader, device, temp_scaler=temp_scaler)
    cal_probs = F.softmax(cal_logits, dim=-1)
    cal_acc = (cal_probs.argmax(-1) == test_labels).float().mean().item()
    cal_ece = expected_calibration_error(cal_probs, test_labels)

    result = {
        "config": cname, "dataset": args.dataset, "seed": args.seed,
        "fitted_temperature": fitted_temp,
        "uncalibrated": {"fused_acc": uncal_acc, "ece": uncal_ece},
        "calibrated": {"fused_acc": cal_acc, "ece": cal_ece},
        "accuracy_delta": cal_acc - uncal_acc,
    }
    out_dir = Path(cfg["paths"]["results"]) / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"calibration_{args.dataset}_{cname}_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
