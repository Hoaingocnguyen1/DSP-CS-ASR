"""
AdapterFusion cho DSP-CS (Hướng 6)
==================================
Định hướng: Khởi tạo 2 adapters (1 cho Việt, 1 cho Anh) TƯƠNG TỰ MoE, NHƯNG
ở AdapterFusion chúng ta không dùng LID xác suất để "Cân điện tử" (Routing).
Thay vào đó, thiết lập một mạng Fusion tự động tính toán Attention weights để
quết định trộn đặc trưng từ LoRA VI và LoRA EN một cách tuyến tính.

Note: Phương pháp này nổi tiếng trên báo Interspeech 2022/2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2
from speechbrain.nnet.adapters import AdaptedModel, LoRA

from dsp_xlsr import CausalPromptGenerator 

class AdapterFusion_DSP_XLSR(nn.Module):
    def __init__(
        self,
        source="facebook/wav2vec2-large-xlsr-53",
        save_path="pretrained_models",
        input_size=1024,
        hidden_size=256,
        num_languages=3,
        lora_rank=32,
        lora_alpha=2.0,
        dropout=0.1,
    ):
        super().__init__()
        self.warmup = True

        base_backbone_vi = Wav2Vec2(
            source=source, save_path=save_path, output_norm=True,
            freeze_feature_extractor=True, freeze=False,
        )
        base_backbone_en = Wav2Vec2(
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

        self.expert_vi = AdaptedModel(
            model_to_adapt=base_backbone_vi, adapter_class=LoRA, all_linear=False,
            target_layers=top6_linear_layers, adapter_kwargs={"rank": lora_rank, "alpha": lora_alpha},
        )
        self.expert_en = AdaptedModel(
            model_to_adapt=base_backbone_en, adapter_class=LoRA, all_linear=False,
            target_layers=top6_linear_layers, adapter_kwargs={"rank": lora_rank, "alpha": lora_alpha},
        )

        for name, param in self.expert_vi.named_parameters():
            if "layer_norm" in name: param.requires_grad = True
        for name, param in self.expert_en.named_parameters():
            if "layer_norm" in name: param.requires_grad = True

        self.prompt_generator = CausalPromptGenerator(
            input_size=input_size, hidden_size=hidden_size, num_languages=num_languages, dropout=dropout,
        )

        # ----------------------------------------------------
        # FUSION LAYER (Hướng 6)
        # ----------------------------------------------------
        # Nhận vào 2 luồng Features [B, T, 1024] => Tính 2 trọng số Weight [B, T, 1]
        self.fusion_query = nn.Linear(input_size, input_size)
        self.fusion_key = nn.Linear(input_size, input_size)
        
        # Khởi tạo ma trận trộn ổn định
        nn.init.xavier_uniform_(self.fusion_query.weight)
        nn.init.xavier_uniform_(self.fusion_key.weight)

    def forward(self, wav, wav_lens=None, hx=None):
        out_vi = self.expert_vi(wav, wav_lens)
        out_en = self.expert_en(wav, wav_lens)
        
        # Xếp 2 luồng chồng lên nhau [B, T, 2, D]
        adapters_out = torch.stack([out_vi, out_en], dim=2) 
        B, T, N_adapters, D = adapters_out.shape
        
        # Lấy trung bình làm ngữ cảnh tham chiếu (Context / Query)
        context = adapters_out.mean(dim=2) # [B, T, D]
        
        # 1. Tính Q và K cho Fusion
        # Q = Context [B, T, 1, D]
        # K = Từng adapter [B, T, 2, D]
        Q = self.fusion_query(context).unsqueeze(2) 
        K = self.fusion_key(adapters_out)           
        
        # 2. Tính Dot-Product Attention: [B, T, 1, D] @ [B, T, D, 2] => [B, T, 1, 2]
        # Thể hiện mức độ "khớp" của mỗi adapter đối với toàn cục Frame đó
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (D ** 0.5)
        
        # Softmax tự động cân bằng xem Frame này ưu tiên xài LoRA nào
        attn_weights = F.softmax(attn_scores, dim=-1) # [B, T, 1, 2]
        
        # 3. Trộn luồng: Sum(Weight * Adapter_Value)
        # [B, T, 1, 2] @ [B, T, 2, D] => [B, T, 1, D] => [B, T, D]
        x_fusion = torch.matmul(attn_weights, adapters_out).squeeze(2)

        # Vẫn dùng Causal LID để lấy ra ngôn ngữ cho nhánh CTC Loss
        gru_input = x_fusion.detach() if self.warmup else x_fusion
        lid_logits, _, hx_new = self.prompt_generator(gru_input, hx)
        
        return x_fusion, lid_logits, hx_new
