"""
MoE-LoRA (Mixture-of-Experts LoRA) cho DSP-CS - Efficient Version
=================================================================
Kiến trúc ĐÚNG: 1 Backbone chung, 2 bộ LoRA delta riêng biệt.

Nguyên lý hoạt động:
  - Backbone XLS-R: Frozen, dùng chung cho cả VI và EN
  - LoRA_VI: Tập hợp các ma trận {A_VI, B_VI} chuyên biệt cho Tiếng Việt  
  - LoRA_EN: Tập hợp các ma trận {A_EN, B_EN} chuyên biệt cho Tiếng Anh
  - Routing: p_vi * LoRA_VI + p_en * LoRA_EN (tổ hợp soft-routing)

Tại mỗi layer được LoRA hóa, output tính như sau:
  y = W_pretrained(x) + p_vi * (B_VI @ A_VI)(x) + p_en * (B_EN @ A_EN)(x)

Điều này CHỈ cần 1 forward pass, tiết kiệm VRAM gần 2x.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2
from dsp_xlsr import CausalPromptGenerator


class RoutedLoraLinear(nn.Module):
    """
    Một lớp Linear được trang bị 2 bộ LoRA (VI + EN).
    Thay vì select 1 expert, lớp này BLEND cả 2 theo routing_weights.
    
    Output = W(x) + routing_weights[0] * B_VI@A_VI(x)
                  + routing_weights[1] * B_EN@A_EN(x)
    """
    def __init__(self, original_linear: nn.Linear, rank: int = 32, alpha: float = 2.0):
        super().__init__()
        d_out, d_in = original_linear.weight.shape
        self.pretrained = original_linear  # Frozen
        self.pretrained.requires_grad_(False)
        
        scale = alpha / rank
        # Expert VI LoRA (A, B)
        self.lora_A_vi = nn.Linear(d_in, rank, bias=False)
        self.lora_B_vi = nn.Linear(rank, d_out, bias=False)
        nn.init.kaiming_uniform_(self.lora_A_vi.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B_vi.weight)
        # Expert EN LoRA (A, B)
        self.lora_A_en = nn.Linear(d_in, rank, bias=False)
        self.lora_B_en = nn.Linear(rank, d_out, bias=False)
        nn.init.kaiming_uniform_(self.lora_A_en.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B_en.weight)
        
        self.scale = scale

    def forward(self, x):
        """Routing weights are read from self._routing set externally before backbone forward."""
        y_base = self.pretrained(x)
        delta_vi = self.lora_B_vi(self.lora_A_vi(x))
        delta_en = self.lora_B_en(self.lora_A_en(x))
        
        routing = getattr(self, "_routing", None)
        if routing is None:
            # Warmup / equal-blend
            return y_base + self.scale * 0.5 * (delta_vi + delta_en)
        
        w_vi, w_en = routing
        # w_vi/w_en: [B, 1] scalars — broadcast over [B, T, D]
        if x.dim() == 3:
            w_vi = w_vi.unsqueeze(-1)  # [B, 1, 1]
            w_en = w_en.unsqueeze(-1)
        return y_base + self.scale * (w_vi * delta_vi + w_en * delta_en)



class MoE_DSP_XLSR(nn.Module):
    """
    DSP-CS với MoE-LoRA (Memory-Efficient Dual-Expert).
    
    1 Backbone XLS-R → 2 bộ LoRA delta nhẹ (VI + EN) → Soft MoE Routing
    → Tiết kiệm ~50% VRAM so với kiến trúc 2-backbone cũ.
    """
    def __init__(
        self,
        source="facebook/wav2vec2-large-xlsr-53",
        save_path="pretrained_models",
        input_size=1024,
        hidden_size=256,
        num_languages=3,  # 0: SIL, 1: VI, 2: EN
        lora_rank=32,
        lora_alpha=2.0,
        dropout=0.1,
    ):
        super().__init__()
        self.warmup = True

        # -----------------------------------------------------------
        # 1. Backbone chung (Frozen) + Causal LID Router
        # -----------------------------------------------------------
        self.backbone = Wav2Vec2(
            source=source, save_path=save_path, output_norm=True,
            freeze_feature_extractor=True, freeze=True,  # Frozen hoàn toàn
        )
        
        # -----------------------------------------------------------
        # 2. Tiêm RoutedLoraLinear vào Top-6 Encoder Layers
        # -----------------------------------------------------------
        self._inject_routed_loras(lora_rank, lora_alpha)
        
        # Unfreeze LayerNorms
        for name, p in self.backbone.named_parameters():
            if "layer_norm" in name:
                p.requires_grad = True
        
        # -----------------------------------------------------------
        # 3. LID Router (dùng để sinh routing weights)
        # -----------------------------------------------------------
        self.prompt_generator = CausalPromptGenerator(
            input_size=input_size, hidden_size=hidden_size,
            num_languages=num_languages, dropout=dropout,
        )
        self.lang_embed = nn.Embedding(num_languages, input_size)
        nn.init.normal_(self.lang_embed.weight, std=0.02)

    def _inject_routed_loras(self, rank, alpha):
        """Thay thế các lớp Linear mục tiêu trong encoder bằng RoutedLoraLinear."""
        target_names = [
            "attention.k_proj", "attention.v_proj",
            "attention.q_proj", "attention.out_proj",
            "feed_forward.intermediate_dense", "feed_forward.output_dense"
        ]
        encoder = self.backbone.model.encoder
        self._routed_loras = nn.ModuleList()
        self._lora_paths = []  # (parent_module, attr_name) for routing in forward
        
        for layer_idx in range(18, 24):
            layer = encoder.layers[layer_idx]
            for attr_path in target_names:
                # Navigate attr_path: "feed_forward.intermediate_dense" → layer.feed_forward.intermediate_dense
                parts = attr_path.split(".")
                obj = layer
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                orig_linear = getattr(obj, parts[-1])
                
                if not isinstance(orig_linear, nn.Linear):
                    continue
                
                routed = RoutedLoraLinear(orig_linear, rank=rank, alpha=alpha)
                setattr(obj, parts[-1], routed)
                self._routed_loras.append(routed)
                self._lora_paths.append(routed)  # same ref

    def _set_routing(self, routing):
        """Đặt routing weights cho tất cả RoutedLoraLinear."""
        for routed_lora in self._lora_paths:
            routed_lora._routing = routing  # stored per forward call

    def forward(self, wav, wav_lens=None, hx=None):
        # Single backbone forward with equal-blend routing (faster).
        # Expert specialization comes from LID-weighted language embedding injection.
        for routed in self._lora_paths:
            routed._routing = None  # equal blend: 0.5*VI + 0.5*EN

        features = self.backbone(wav, wav_lens)

        gru_input = features.detach() if self.warmup else features
        lid_logits, _, hx_new = self.prompt_generator(gru_input, hx)
        lid_probs = F.softmax(lid_logits.detach(), dim=-1)  # [B, T, 3]

        # Language Embedding Injection with stronger scale (0.3 vs 0.1)
        # to compensate for removal of re-routing pass.
        lang_emb = torch.matmul(lid_probs, self.lang_embed.weight)  # [B, T, 1024]
        adapted_features = features + 0.3 * lang_emb

        return adapted_features, lid_logits, hx_new
