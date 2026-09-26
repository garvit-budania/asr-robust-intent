"""
Lightweight fusion model that operates on PRECOMPUTED, FROZEN encoder embeddings
rather than raw audio/text. This is the core of the efficient approach: since
the large pretrained encoders never appear inside this model at all, training it
involves only a few thousand parameters and no backprop through wav2vec2/HuBERT/
WavLM/DistilBERT/RoBERTa -- an epoch takes seconds instead of minutes.

Mathematically identical to ThreeBranchFusionModel in src/models/fusion.py: same
three-loss objective, same learned alpha weights, same product-rule-motivated
probability-level fusion (Sec. III of the paper) -- just with the encoder forward
pass factored out and precomputed once (src/features/extract_embeddings.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.fusion import AlphaWeights, TemperatureScaler


class HeadOnlyBranch(nn.Module):
    """A single branch: fixed embedding -> linear head -> logits. Optionally calibrated."""

    def __init__(self, embedding_dim, num_labels):
        super().__init__()
        self.head = nn.Linear(embedding_dim, num_labels)
        self.temp_scaler = TemperatureScaler()

    def forward(self, embedding, apply_temperature=False):
        logits = self.head(embedding)
        if apply_temperature:
            logits = self.temp_scaler(logits)
        return logits


class HeadOnlyFusionModel(nn.Module):
    """Three-branch fusion over precomputed embeddings -- see ThreeBranchFusionModel
    in fusion.py for the full-fine-tuning equivalent this mirrors exactly, Sec. IV
    of the paper."""

    def __init__(self, text_embedding_dim, audio_embedding_dim, num_labels,
                 fusion_hidden=512, alpha1_init=0.3, alpha2_init=0.3,
                 embedding_level_fusion=False):
        super().__init__()
        self.text_branch = HeadOnlyBranch(text_embedding_dim, num_labels)
        self.audio_branch = HeadOnlyBranch(audio_embedding_dim, num_labels)
        self.num_labels = num_labels
        self.embedding_level_fusion = embedding_level_fusion

        fusion_in = (text_embedding_dim + audio_embedding_dim) if embedding_level_fusion else 2 * num_labels
        self.fusion_fc1 = nn.Linear(fusion_in, fusion_hidden)
        self.fusion_fc2 = nn.Linear(fusion_hidden, num_labels)
        self.alpha_weights = AlphaWeights(alpha1_init, alpha2_init)

    def forward(self, text_embedding, audio_embedding, apply_temperature=False):
        text_logits = self.text_branch(text_embedding, apply_temperature)
        audio_logits = self.audio_branch(audio_embedding, apply_temperature)

        p_text = F.softmax(text_logits, dim=-1)
        p_audio = F.softmax(audio_logits, dim=-1)

        fusion_input = torch.cat([text_embedding, audio_embedding], dim=-1) if self.embedding_level_fusion \
            else torch.cat([p_text, p_audio], dim=-1)

        h = F.relu(self.fusion_fc1(fusion_input))
        fused_logits = self.fusion_fc2(h)

        return {"text_logits": text_logits, "audio_logits": audio_logits, "fused_logits": fused_logits}

    def compute_loss(self, outputs, labels):
        loss1 = F.cross_entropy(outputs["text_logits"], labels)
        loss2 = F.cross_entropy(outputs["audio_logits"], labels)
        loss3 = F.cross_entropy(outputs["fused_logits"], labels)
        a1, a2, a3 = self.alpha_weights()
        total = a1 * loss1 + a2 * loss2 + a3 * loss3
        return total, {"loss1": loss1.item(), "loss2": loss2.item(), "loss3": loss3.item(),
                        "alpha1": a1.item(), "alpha2": a2.item(), "alpha3": a3.item()}


class HeadOnlyTextModel(nn.Module):
    def __init__(self, embedding_dim, num_labels):
        super().__init__()
        self.branch = HeadOnlyBranch(embedding_dim, num_labels)

    def forward(self, text_embedding, **kwargs):
        return {"logits": self.branch(text_embedding)}

    def compute_loss(self, outputs, labels):
        loss = F.cross_entropy(outputs["logits"], labels)
        return loss, {"loss": loss.item()}


class HeadOnlyAudioModel(nn.Module):
    def __init__(self, embedding_dim, num_labels):
        super().__init__()
        self.branch = HeadOnlyBranch(embedding_dim, num_labels)

    def forward(self, audio_embedding, **kwargs):
        return {"logits": self.branch(audio_embedding)}

    def compute_loss(self, outputs, labels):
        loss = F.cross_entropy(outputs["logits"], labels)
        return loss, {"loss": loss.item()}
