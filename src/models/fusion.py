"""
Three-branch audio-text fusion model, matching Fig. 1 / Appendix A of the execution plan:

  Text branch:  transcript -> TextEncoder -> [CLS] -> Linear -> softmax -> P_T   (loss1)
  Audio branch: waveform   -> AudioEncoder -> mean-pool -> Linear -> softmax/T -> P_A (loss2)
  Fusion:       concat([P_T, P_A]) -> Linear(512) + ReLU -> Linear(K) -> softmax -> P_F (loss3)

  L = alpha1 * loss1 + alpha2 * loss2 + (1 - alpha1 - alpha2) * loss3
  alpha1, alpha2 are learned (parametrized via unconstrained logits + softmax-simplex
  projection so alpha1, alpha2 in (0,1) and alpha1+alpha2 < 1 always holds).

Also includes:
  - TextOnlyModel / AudioOnlyModel for the baseline comparisons.
  - Temperature scaling module for the calibration ablation (Sec. IV-D).
  - An embedding-level fusion variant for the fusion-level ablation (Sec. IV-C TODO).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class TemperatureScaler(nn.Module):
    """Learns a single scalar temperature to calibrate a branch's logits (Guo et al. 2017)."""

    def __init__(self, init_t=1.5):
        super().__init__()
        self.log_t = nn.Parameter(torch.log(torch.tensor(init_t)))

    def forward(self, logits):
        t = torch.exp(self.log_t)
        return logits / t

    def fit(self, logits, labels, lr=0.01, max_iter=200):
        """Optimize temperature on held-out validation logits (NLL minimization)."""
        optimizer = torch.optim.LBFGS([self.log_t], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        return torch.exp(self.log_t).item()


def _find_transformer_layers(encoder):
    """Locate the list of transformer blocks across different HF model families.
    BERT/RoBERTa-style models expose `.encoder.layer`; DistilBERT exposes
    `.transformer.layer` directly (it has no `.encoder` attribute at all).
    Using hasattr checks at every step avoids AttributeError on models that
    simply don't have a given attribute, instead of assuming one layout."""
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        return encoder.encoder.layer  # BERT, RoBERTa
    if hasattr(encoder, "transformer") and hasattr(encoder.transformer, "layer"):
        return encoder.transformer.layer  # DistilBERT
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layers"):
        return encoder.encoder.layers  # wav2vec2 / HuBERT / WavLM style (audio encoders)
    raise AttributeError(
        f"Don't know how to locate transformer layers for encoder type {type(encoder).__name__}. "
        f"Add a case to _find_transformer_layers() in src/models/fusion.py."
    )


class TextBranch(nn.Module):
    def __init__(self, model_name, num_labels, freeze=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Linear(hidden, num_labels)
        self.temp_scaler = TemperatureScaler()
        if freeze:
            self.freeze_encoder()

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_top_layers(self, n_layers=2):
        layers = _find_transformer_layers(self.encoder)
        for layer in list(layers)[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, input_ids, attention_mask, apply_temperature=False):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]  # [CLS] embedding
        logits = self.head(cls)
        if apply_temperature:
            logits = self.temp_scaler(logits)
        return logits, cls


class AudioBranch(nn.Module):
    def __init__(self, model_name, num_labels, freeze=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.head = nn.Linear(hidden, num_labels)
        self.temp_scaler = TemperatureScaler()
        if freeze:
            self.freeze_encoder()

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_top_layers(self, n_layers=2):
        layers = _find_transformer_layers(self.encoder)
        for layer in list(layers)[-n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, input_values, apply_temperature=False):
        out = self.encoder(input_values=input_values)
        pooled = out.last_hidden_state.mean(dim=1)  # mean-pool over time
        logits = self.head(pooled)
        if apply_temperature:
            logits = self.temp_scaler(logits)
        return logits, pooled


class AlphaWeights(nn.Module):
    """Learned alpha1, alpha2 with alpha1,alpha2 in (0,1) and alpha1+alpha2<1 guaranteed
    via a 3-way softmax over unconstrained logits (third slot = 1-alpha1-alpha2)."""

    def __init__(self, init_a1=0.3, init_a2=0.3):
        super().__init__()
        init_a3 = max(1e-3, 1 - init_a1 - init_a2)
        logits = torch.log(torch.tensor([init_a1, init_a2, init_a3]))
        self.logits = nn.Parameter(logits)

    def forward(self):
        w = F.softmax(self.logits, dim=0)
        return w[0], w[1], w[2]  # alpha1, alpha2, 1-alpha1-alpha2


class ThreeBranchFusionModel(nn.Module):
    """Full model: text branch + audio branch + fusion branch, probability-level fusion."""

    def __init__(self, text_model_name, audio_model_name, num_labels,
                 fusion_hidden=512, alpha1_init=0.3, alpha2_init=0.3,
                 embedding_level_fusion=False):
        super().__init__()
        self.text_branch = TextBranch(text_model_name, num_labels)
        self.audio_branch = AudioBranch(audio_model_name, num_labels)
        self.num_labels = num_labels
        self.embedding_level_fusion = embedding_level_fusion

        if embedding_level_fusion:
            # Ablation variant: concat raw embeddings instead of probability vectors.
            text_hidden = self.text_branch.encoder.config.hidden_size
            audio_hidden = self.audio_branch.encoder.config.hidden_size
            fusion_in = text_hidden + audio_hidden
        else:
            fusion_in = 2 * num_labels  # concat [P_T ; P_A]

        self.fusion_fc1 = nn.Linear(fusion_in, fusion_hidden)
        self.fusion_fc2 = nn.Linear(fusion_hidden, num_labels)
        self.alpha_weights = AlphaWeights(alpha1_init, alpha2_init)

    def unfreeze_top_layers(self, n_layers=2):
        self.text_branch.unfreeze_top_layers(n_layers)
        self.audio_branch.unfreeze_top_layers(n_layers)

    def forward(self, input_ids, attention_mask, input_values, apply_temperature=False):
        text_logits, text_emb = self.text_branch(input_ids, attention_mask, apply_temperature)
        audio_logits, audio_emb = self.audio_branch(input_values, apply_temperature)

        p_text = F.softmax(text_logits, dim=-1)
        p_audio = F.softmax(audio_logits, dim=-1)

        if self.embedding_level_fusion:
            fusion_input = torch.cat([text_emb, audio_emb], dim=-1)
        else:
            fusion_input = torch.cat([p_text, p_audio], dim=-1)

        h = F.relu(self.fusion_fc1(fusion_input))
        fused_logits = self.fusion_fc2(h)

        return {
            "text_logits": text_logits,
            "audio_logits": audio_logits,
            "fused_logits": fused_logits,
            "text_embedding": text_emb,
            "audio_embedding": audio_emb,
            "cosine_sim": F.cosine_similarity(text_emb.mean(-1, keepdim=True).expand_as(text_emb)
                                               if text_emb.shape != audio_emb.shape else text_emb,
                                               audio_emb, dim=-1).mean()
            if text_emb.shape[-1] == audio_emb.shape[-1] else torch.tensor(float("nan")),
        }

    def compute_loss(self, outputs, labels):
        loss1 = F.cross_entropy(outputs["text_logits"], labels)
        loss2 = F.cross_entropy(outputs["audio_logits"], labels)
        loss3 = F.cross_entropy(outputs["fused_logits"], labels)
        a1, a2, a3 = self.alpha_weights()
        total = a1 * loss1 + a2 * loss2 + a3 * loss3
        return total, {"loss1": loss1.item(), "loss2": loss2.item(), "loss3": loss3.item(),
                        "alpha1": a1.item(), "alpha2": a2.item(), "alpha3": a3.item()}


class TextOnlyModel(nn.Module):
    """Text-only baseline (also used as the audio-branch's textual counterpart in
    encoder-objective comparisons: BERT/RoBERTa fine-tuned alone)."""

    def __init__(self, model_name, num_labels):
        super().__init__()
        self.branch = TextBranch(model_name, num_labels, freeze=True)

    def unfreeze_top_layers(self, n_layers=2):
        self.branch.unfreeze_top_layers(n_layers)

    def forward(self, input_ids, attention_mask, **kwargs):
        logits, _ = self.branch(input_ids, attention_mask)
        return {"logits": logits}

    def compute_loss(self, outputs, labels):
        loss = F.cross_entropy(outputs["logits"], labels)
        return loss, {"loss": loss.item()}


class AudioOnlyModel(nn.Module):
    """Audio-only baseline -- also serves as the epsilon_A proxy for Corollary 1's floor."""

    def __init__(self, model_name, num_labels):
        super().__init__()
        self.branch = AudioBranch(model_name, num_labels, freeze=True)

    def unfreeze_top_layers(self, n_layers=2):
        self.branch.unfreeze_top_layers(n_layers)

    def forward(self, input_values, **kwargs):
        logits, _ = self.branch(input_values)
        return {"logits": logits}

    def compute_loss(self, outputs, labels):
        loss = F.cross_entropy(outputs["logits"], labels)
        return loss, {"loss": loss.item()}
