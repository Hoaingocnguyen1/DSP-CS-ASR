"""
DSP-CS với Cross-Attention Language Injection (Hướng 5)
=======================================================
Định hướng: Thay vì cộng trực tiếp `x = x + lambda * lang_emb` một cách cứng nhắc,
chúng ta dùng Multi-Head Cross-Attention (MHCA).
Đặc trưng âm học (x) đóng vai trò là Query (cần tìm hiểu).
Ma trận nhúng ngôn ngữ đóng vai trò là Key và Value (nguồn tri thức).
Mô hình sẽ tự động trích xuất đúng lượng thông tin ngôn ngữ cần thiết.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2
from speechbrain.nnet.adapters import AdaptedModel, LoRA

from dsp_xlsr import CausalPromptGenerator 

class CrossAttention_DSP_XLSR(nn.Module):
    def __init__(
        self,
        source="facebook/wav2vec2-large-xlsr-53",
        save_path="pretrained_models",
        input_size=1024,
        hidden_size=256,
        num_languages=3,
        lora_rank=64,
        lora_alpha=2.0,
        dropout=0.1,
        num_heads=4, # 4 Heads cho Cross-Attention
    ):
        super().__init__()
        self.warmup = True

        # 1. Khởi tạo Backbone
        base_backbone = Wav2Vec2(
            source=source, save_path=save_path, output_norm=True,
            freeze_feature_extractor=True, freeze=False,
        )
        
        top6_linear_layers = []
        for i in range(18, 24):
            p = f"model.encoder.layers.{i}"
            top6_linear_layers += [
                f"{p}.attention.k_proj", f"{p}.attention.v_proj",
                f"{p}.attention.q_proj", f"{p}.attention.out_proj",
                f"{p}.feed_forward.intermediate_dense", f"{p}.feed_forward.output_dense"
            ]

        self.backbone = AdaptedModel(
            model_to_adapt=base_backbone, adapter_class=LoRA, all_linear=False,
            target_layers=top6_linear_layers,
            adapter_kwargs={"rank": lora_rank, "alpha": lora_alpha},
        )
        for name, param in self.backbone.named_parameters():
            if "layer_norm" in name:
                param.requires_grad = True

        # 2. Causal Prompt & Gate (như cũ)
        self.prompt_generator = CausalPromptGenerator(
            input_size=input_size, hidden_size=hidden_size,
            num_languages=num_languages, dropout=dropout,
        )
        self.prompt_proj = nn.Linear(hidden_size, input_size)
        self.prompt_dropout = nn.Dropout(p=dropout)
        self.gate_net = nn.Sequential(
            nn.Linear(num_languages, input_size // 4), nn.ReLU(),
            nn.Linear(input_size // 4, input_size), nn.Tanh(),
        )

        # 3. [Khác biệt cốt lõi] Mạng Multi-Head Cross-Attention (MHCA) cho Ngôn ngữ
        # Thay vì chỉ 1 ma trận Embedding đơn giản
        self.lang_dictionary = nn.Parameter(torch.randn(num_languages, input_size))
        
        # Batch_first = True để dễ xử lý dạng [B, T, D]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=input_size, num_heads=num_heads, 
            dropout=dropout, batch_first=True
        )
        self.layer_norm_attn = nn.LayerNorm(input_size)

    def forward(self, wav, wav_lens=None, hx=None):
        adapted_features = self.backbone(wav, wav_lens)
        
        gru_input = adapted_features.detach() if self.warmup else adapted_features
        lid_logits, prompt, hx_new = self.prompt_generator(gru_input, hx)
        
        # Tiêm Prompt vào Gate (như ở chương 4)
        lid_probs = F.softmax(lid_logits.detach(), dim=-1)
        gate = self.gate_net(lid_probs)
        prompt_projected = self.prompt_dropout(self.prompt_proj(prompt))
        x_prime = adapted_features + gate * prompt_projected

        # ----------------------------------------------------
        # Cross-Attention Injection (Hướng 5)
        # ----------------------------------------------------
        # x_prime đóng vai trò là Query (chủ thể đi hỏi)
        # lang_dictionary đóng vai trò là Key và Value (từ điển ngôn ngữ học)
        
        B, T, D = x_prime.shape
        # Expand từ điển để phủ hết kích thước Batch
        lang_dict_batched = self.lang_dictionary.unsqueeze(0).expand(B, -1, -1) # [B, 3, 1024]
        
        # Ở Cross Attention: Query = Đặc trưng âm thanh, Key/Value = Từ điển Ngôn Ngữ
        # Để mô hình tự lấy được lượng thông tin nó cần cho từng frame.
        lang_context, _ = self.cross_attn(
            query=x_prime, 
            key=lang_dict_batched, 
            value=lang_dict_batched
        )
        
        # Trộn luồng (Residual connection) theo kiểu Transformer chuẩn
        x_final = self.layer_norm_attn(x_prime + lang_context)

        return x_final, lid_logits, hx_new
