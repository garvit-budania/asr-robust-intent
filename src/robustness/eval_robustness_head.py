"""
ASR robustness evaluation (head-only version) -- Sec. V-C / Step 6. Audio
embeddings are reused UNCHANGED across all WER conditions (audio is unaffected
by transcript corruption -- this is the point); text embeddings must be
re-extracted per WER level first via src/features/extract_embeddings.py with
--corrupted_transcripts_json and --wer_key.

FIX vs. previous version: the audio-only floor was being computed by loading
the audio_only_hubert-base-ls960 checkpoint but feeding it embeddings built
from --best_fused_audio (e.g. wavlm-base-plus) -- a mismatched feature space
that produced near-random accuracy. The floor must use the SAME encoder the
audio_only_<name> checkpoint was actually trained with, independent of
whichever encoder the best FUSED config happens to use.

Usage:
  python3 -m src.robustness.eval_robustness_head --dataset slurp \
      --best_fused_audio microsoft/wavlm-base-plus --best_fused_text roberta-base \
      --audio_only_encoder facebook/hubert-base-ls960 --seed 42
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset import load_label_map
from src.train_head import CachedEmbeddingDataset
from src.models.head_fusion import HeadOnlyFusionModel, HeadOnlyTextModel, HeadOnlyAudioModel


def eval_loader(model, loader, device, kind):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            if kind == "fused":
                out = model(batch["text_embedding"], batch["audio_embedding"])
                logits = out["fused_logits"]
            elif kind == "audio_only":
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
    ap.add_argument("--best_fused_audio", default="microsoft/wavlm-base-plus")
    ap.add_argument("--best_fused_text", default="roberta-base")
    ap.add_argument("--audio_only_encoder", default="facebook/hubert-base-ls960",
                     help="MUST match whichever encoder the audio_only_<name> checkpoint "
                          "was actually trained with -- this is independent of "
                          "--best_fused_audio, which can be a different encoder.")
    ap.add_argument("--text_only_encoder", default="roberta-base",
                     help="MUST match whichever encoder the text_only_<name> checkpoint "
                          "was actually trained with.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds_root = Path(cfg["datasets"][args.dataset]["root"])
    label_map = load_label_map(ds_root / "label_map.json")
    num_labels = len(label_map)

    a, t = args.best_fused_audio.split("/")[-1], args.best_fused_text.split("/")[-1]
    cname = f"fused_{a}_{t}"

    # -- Audio-only floor (Corollary 1 epsilon_A proxy) -- WER-invariant by construction.
    # Uses --audio_only_encoder, NOT --best_fused_audio: these can legitimately differ,
    # and using the wrong one here silently produces a meaningless near-random "floor".
    audio_test_ds = CachedEmbeddingDataset(args.dataset, "test", label_map,
                                            audio_encoder=args.audio_only_encoder)
    audio_dim = audio_test_ds[0]["audio_embedding"].shape[0]
    audio_model = HeadOnlyAudioModel(audio_dim, num_labels).to(device)
    audio_ckpt_name = f"audio_only_{args.audio_only_encoder.split('/')[-1]}"
    audio_ckpt = (Path(cfg["paths"]["checkpoints"]) / args.dataset /
                  audio_ckpt_name / f"seed{args.seed}" / "best.pt")
    audio_model.load_state_dict(torch.load(audio_ckpt, map_location=device))
    audio_floor_acc = eval_loader(audio_model, DataLoader(audio_test_ds, batch_size=256), device, "audio_only")
    print(f"Audio-only floor (Corollary 1, using {args.audio_only_encoder}): {audio_floor_acc:.4f}")

    text_only_ckpt_name = f"text_only_{args.text_only_encoder.split('/')[-1]}"
    text_model_ckpt = (Path(cfg["paths"]["checkpoints"]) / args.dataset /
                        text_only_ckpt_name / f"seed{args.seed}" / "best.pt")
    fused_ckpt = Path(cfg["paths"]["checkpoints"]) / args.dataset / cname / f"seed{args.seed}" / "best.pt"

    rcfg = cfg["robustness"]
    rows = []
    for wer in rcfg["wer_targets"]:
        tag = f"wer{int(round(wer * 100))}" if wer > 0 else None

        text_test_ds = CachedEmbeddingDataset(args.dataset, "test", label_map,
                                               text_encoder=args.best_fused_text, text_tag=tag)
        text_dim = text_test_ds[0]["text_embedding"].shape[0]

        combined_ds = CachedEmbeddingDataset(args.dataset, "test", label_map,
                                              audio_encoder=args.best_fused_audio,
                                              text_encoder=args.best_fused_text, text_tag=tag)

        text_model = HeadOnlyTextModel(text_dim, num_labels).to(device)
        text_model.load_state_dict(torch.load(text_model_ckpt, map_location=device))
        text_acc = eval_loader(text_model, DataLoader(text_test_ds, batch_size=256), device, "text_only")

        fused_model = HeadOnlyFusionModel(text_dim, audio_dim, num_labels).to(device)
        fused_model.load_state_dict(torch.load(fused_ckpt, map_location=device))
        fused_acc = eval_loader(fused_model, DataLoader(combined_ds, batch_size=256), device, "fused")

        rows.append({"wer_target": wer, "text_only_acc": text_acc, "fused_acc": fused_acc,
                     "audio_only_floor": audio_floor_acc})
        print(f"WER {wer:.2f}: text_only={text_acc:.4f} fused={fused_acc:.4f} floor={audio_floor_acc:.4f}")

    out_dir = Path(cfg["paths"]["results"]) / "robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.dataset}_robustness_curve.json", "w") as f:
        json.dump(rows, f, indent=2)

    wers = [r["wer_target"] * 100 for r in rows]
    plt.figure(figsize=(7, 5))
    plt.plot(wers, [r["text_only_acc"] for r in rows], marker="o", label="Text-only (RoBERTa)")
    plt.plot(wers, [r["fused_acc"] for r in rows], marker="s", label=f"Fused ({cname})")
    plt.axhline(audio_floor_acc, linestyle="--", color="gray", label="Audio-only floor (Corollary 1)")
    plt.xlabel("Word Error Rate (%)")
    plt.ylabel("Accuracy")
    plt.title(f"ASR Robustness Curve \u2014 {args.dataset.upper()}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{args.dataset}_robustness_curve.png", dpi=150)
    print(f"Saved plot to {out_dir / f'{args.dataset}_robustness_curve.png'}")


if __name__ == "__main__":
    main()
