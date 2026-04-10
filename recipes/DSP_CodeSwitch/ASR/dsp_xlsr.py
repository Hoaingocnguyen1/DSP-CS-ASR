"""
DSP XLS-R Code-Switching ASR Model
====================================
Architecture:
  Audio → [LoRA-Adapted XLS-R 53 Large] → features [B, T, 1024]
               ↓
          CausalPromptGenerator (Causal GRU)
            - Reads features stream-by-stream (for real-time LID)
            - Outputs 1: LID logits [B, T, 3] and GRU hidden state (for streaming)
            - Outputs 2: Acoustic Prompt [B, T, 256]
               ↓
  [Feature Injection]
  Final_Features = features + Tanh(Gate) * Linear(Acoustic_Prompt)

Same DSP-CS method as DSP_W2VBERT but with XLS-R backbone for ablation study.
XLS-R outputs 1024-dim features (same as w2v-bert-2.0) so all components are reused.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2
from speechbrain.nnet.adapters import AdaptedModel, LoRA
from speechbrain.nnet.normalization import LayerNorm


class CausalPromptGenerator(nn.Module):
    """
    Causal GRU-based Language ID predictor for streaming code-switching ASR.
    Identical to the one in dsp_model.py — reused for consistency.
    """

    def __init__(self, input_size=1024, hidden_size=256, num_languages=3, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(input_size=hidden_size)
        self.lid_head = nn.Linear(hidden_size, num_languages)

        # === Stability Init ===
        # 1. Orthogonal init for GRU recurrent weights (better gradient flow)
        for name, param in self.gru.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        # 2. Small init for LID head → logits near 0 → loss ≈ ln(3) ≈ 1.1
        nn.init.normal_(self.lid_head.weight, std=0.01)
        nn.init.zeros_(self.lid_head.bias)

    def forward(self, features, hx=None):
        gru_out, hx_new = self.gru(features, hx)
        gru_out = self.norm(self.dropout(gru_out))
        # Clamp GRU output to prevent hidden state explosion
        gru_out = gru_out.clamp(min=-5, max=5)
        lid_logits = self.lid_head(gru_out)
        return lid_logits, gru_out, hx_new


class DSP_XLSR(nn.Module):
    """
    DSP-CS method applied on XLS-R 53 Large backbone.

    Same architecture as DSP_W2VBERT:
      - LoRA adapters injected into all Linear layers
      - CausalPromptGenerator for frame-level LID
      - Acoustic Prompt Injection via learnable Tanh gate

    Arguments
    ---------
    source : str
        HuggingFace model ID for XLS-R backbone.
    save_path : str
        Local path to cache the backbone.
    input_size : int
        Backbone output feature dimension (1024 for XLS-R large).
    hidden_size : int
        GRU hidden size in CausalPromptGenerator.
    num_languages : int
        LID class count (VI=1, EN=2, SIL=0).
    lora_rank : int
        LoRA rank r. Lower = fewer params.
    lora_alpha : float
        LoRA scaling factor α.
    dropout : float
        Dropout in CausalPromptGenerator's GRU.
    """

    def __init__(
        self,
        source="facebook/wav2vec2-large-xlsr-53",
        save_path="pretrained_models",
        input_size=1024,
        hidden_size=256,
        num_languages=3,
        lora_rank=16,
        lora_alpha=1.0,
        lora_layers=None,
        dropout=0.1,
    ):
        super().__init__()
        self.warmup = True  # Stage 1: only train GRU. Set False for Stage 2.

        # 1. Load XLS-R backbone (freeze_feature_extractor=True, freeze=False for LoRA)
        base_backbone = Wav2Vec2(
            source=source,
            save_path=save_path,
            output_norm=True,
            freeze_feature_extractor=True,
            freeze=False,  # LoRA freezes pretrained params internally
        )

        # 2. Wrap with LoRA adapters
        # v3: top-12 layers (12-23) with lower rank for distributed adaptation
        # More layers = broader feature adaptation, lower rank = less per-layer overfitting
        lora_linear_layers = []
        lora_start = 12  # v3: start from layer 12 (was 18 in v1)
        for i in range(lora_start, 24):
            p = f"model.encoder.layers.{i}"
            lora_linear_layers += [
                f"{p}.attention.k_proj",
                f"{p}.attention.v_proj",
                f"{p}.attention.q_proj",
                f"{p}.attention.out_proj",
                f"{p}.feed_forward.intermediate_dense",
                f"{p}.feed_forward.output_dense",
            ]
        if lora_layers is not None:
            lora_linear_layers = lora_layers
        self.backbone = AdaptedModel(
            model_to_adapt=base_backbone,
            adapter_class=LoRA,
            all_linear=False,
            target_layers=lora_linear_layers,
            adapter_kwargs={
                "rank": lora_rank,
                "alpha": lora_alpha,
            },
        )

        # 2.5 Unfreeze LayerNorms (LoRA + LN Trick)
        # Unfreezing just the LayerNorms adds a negligible number of parameters (~30k)
        # but allows the feature distributions to re-calibrate for Code-Switching.
        # This often enables PEFT to match or beat Full Fine-Tuning.
        for name, param in self.backbone.named_parameters():
            if "layer_norm" in name:
                param.requires_grad = True

        # 3. CausalPromptGenerator for streaming LID
        self.prompt_generator = CausalPromptGenerator(
            input_size=input_size,
            hidden_size=hidden_size,
            num_languages=num_languages,
            dropout=dropout,
        )

        # 4. LID-Conditioned Gate (ORIGINAL v3 architecture)
        #    gate = gate_net(lid_probs)  → [B, T, 1024]
        #    adapted = features + gate * prompt_proj(prompt) + scale * lang_embed(lid)
        #    Gate is CONDITIONED on LID (not a scalar!), so injection adapts per-frame
        self.gate_net = nn.Sequential(
            nn.Linear(num_languages, hidden_size),   # 3 → 256
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),       # 256 → 1024
        )
        # Init gate_net to output ~0 at start (safe identity)
        nn.init.normal_(self.gate_net[2].weight, std=0.01)
        nn.init.zeros_(self.gate_net[2].bias)

        self.prompt_proj = nn.Linear(hidden_size, input_size)  # 256 → 1024
        nn.init.normal_(self.prompt_proj.weight, std=0.01)
        nn.init.zeros_(self.prompt_proj.bias)

        # Language embedding: adds language-specific bias to features
        self.lang_embed = nn.Embedding(num_languages, input_size)  # 3 × 1024
        self.lang_emb_scale = nn.Parameter(torch.tensor(0.0))  # Learnable scale, init 0

        self.prompt_dropout = nn.Dropout(p=dropout)

    def forward(self, wav, wav_lens=None, hx=None):
        # 1. LoRA-adapted feature extraction [B, T, 1024]
        adapted_features = self.backbone(wav, wav_lens)

        # 2. Causal LID prediction & Acoustic Prompt
        gru_input = adapted_features.detach() if self.warmup else adapted_features
        lid_logits, prompt, hx_new = self.prompt_generator(gru_input, hx)

        # 3. LID-Conditioned Gate Injection (v3 architecture)
        lid_probs = F.softmax(lid_logits.detach(), dim=-1)  # [B, T, 3]
        gate = torch.tanh(self.gate_net(lid_probs))             # [B, T, 1024]
        prompt_d = self.prompt_dropout(prompt)
        injection = gate * self.prompt_proj(prompt_d)         # [B, T, 1024]

        # Language embedding: adds language-specific bias
        lid_hard = lid_probs.argmax(dim=-1)                   # [B, T]
        lang_bias = self.lang_emb_scale * self.lang_embed(lid_hard)

        adapted_features = adapted_features + injection + lang_bias

        return adapted_features, lid_logits, hx_new, None


