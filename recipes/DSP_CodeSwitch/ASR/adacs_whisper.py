import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.integrations.huggingface.whisper import Whisper
from speechbrain.nnet.adapters import AdaptedModel, LoRA
from speechbrain.nnet.normalization import LayerNorm

class BiasAttentionModule(nn.Module):
    """
    Bias Attention Module (BAM) from AdaCS paper.
    Uses a Learned Context Memory Bank to simulate the Bias List encoding.
    """
    def __init__(self, hidden_size=768, num_biases=500, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_biases = num_biases
        
        # 1. Bias Encoder (Simulated via Learnable Memory Bank for speed & efficiency)
        self.bias_memory = nn.Parameter(torch.Tensor(num_biases, hidden_size))
        nn.init.normal_(self.bias_memory, std=0.02)
        
        # 2. Rank & Selection (Learned projection for Q/K matching)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        
        # 3. Cross-Attention Mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, encoder_hidden_states):
        """
        encoder_hidden_states: [B, T, D]
        """
        B, T, D = encoder_hidden_states.shape
        
        # Q: from Audio Encoder, K/V: from Bias Memory
        queries = self.query_proj(encoder_hidden_states) # [B, T, D]
        
        # Expand bias memory for the batch
        keys = self.key_proj(self.bias_memory).unsqueeze(0).expand(B, -1, -1) # [B, N_bias, D]
        values = self.bias_memory.unsqueeze(0).expand(B, -1, -1) # [B, N_bias, D]
        
        # Cross Attention
        attn_out, attn_weights = self.attention(
            query=queries,
            key=keys,
            value=values,
            need_weights=True
        )
        
        # Residual connection + Norm
        out = self.norm(encoder_hidden_states + self.dropout(attn_out))
        return out, attn_weights


class AdaCS_Whisper(nn.Module):
    """
    AdaCS applied to Whisper Small backbone.
    Supports both Full-Finetune or LoRA adaptation.
    """
    def __init__(
        self,
        source="openai/whisper-small",
        save_path="pretrained_models",
        use_lora=False,
        lora_rank=16,
        lora_alpha=32.0,
        num_biases=500,
        language="vi",
    ):
        super().__init__()
        self.use_lora = use_lora
        
        # 1. Load Whisper backbone
        self.whisper = Whisper(
            source=source,
            save_path=save_path,
            encoder_only=False,
            freeze=False,
            freeze_encoder=False,
            language=language,
            task="transcribe",
        )

        # 2. Freeze Configuration
        if self.use_lora:
            # Freeze all first
            for param in self.whisper.parameters():
                param.requires_grad = False
                
            # Apply LoRA
            lora_target_layers = []
            for i in range(12):
                p = f"layers.{i}.self_attn"
                lora_target_layers += [f"{p}.k_proj", f"{p}.v_proj", f"{p}.q_proj", f"{p}.out_proj"]

            self.whisper.model.encoder = AdaptedModel(
                model_to_adapt=self.whisper.model.encoder,
                adapter_class=LoRA,
                all_linear=False,
                target_layers=lora_target_layers,
                adapter_kwargs={"rank": lora_rank, "alpha": lora_alpha},
            )

            # Unfreeze LayerNorms and Decoder
            for name, param in self.whisper.model.encoder.named_parameters():
                if "layer_norm" in name:
                    param.requires_grad = True
            for param in self.whisper.model.decoder.parameters():
                param.requires_grad = True
        else:
            # FULL FINETUNE
            for param in self.whisper.parameters():
                param.requires_grad = True

        # 3. AdaCS Bias Attention Module (BAM)
        # Integrated after the Whisper Encoder
        self.bam = BiasAttentionModule(hidden_size=768, num_biases=num_biases)

    @property
    def tokenizer(self):
        return self.whisper.tokenizer
        
    @property
    def model(self):
        return self.whisper.model
        
    @property
    def bos(self): return self.whisper.bos
    @property
    def eos(self): return self.whisper.eos
    @property
    def bos_lm(self): return self.whisper.bos_lm
    @property
    def bos_prev(self): return self.whisper.bos_prev
    @property
    def no_speech(self): return self.whisper.no_speech
    @property
    def transcribe(self): return self.whisper.transcribe
    @property
    def translate(self): return self.whisper.translate
    @property
    def non_speech_tokens(self): return self.whisper.non_speech_tokens
    @property
    def config(self): return self.whisper.config
    @property
    def get_suppress_tokens(self): return self.whisper.get_suppress_tokens

    def _get_mel(self, wav):
        return self.whisper._get_mel(wav)

    def forward_encoder(self, mel):
        # 1. Base Encoder
        encoder_out = self.whisper.forward_encoder(mel)
        # 2. Add Bias Attention Module normalization
        encoder_out_adapted, _ = self.bam(encoder_out)
        return encoder_out_adapted

    def forward_decoder(self, encoder_out, decoder_input_ids, use_cache=True, past_key_values=None):
        return self.whisper.forward_decoder(
            encoder_out,
            decoder_input_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

    def set_task(self, task):
        self.whisper.set_task(task)
