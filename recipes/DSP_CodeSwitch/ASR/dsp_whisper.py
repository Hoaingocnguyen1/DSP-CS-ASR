"""
DSP-LAA Whisper Code-Switching ASR Model (V4)
===============================================
Architecture:
  Audio → [Whisper Encoder + LoRA] → encoder_out [B, 1500, 768]
               ↓
         CausalPromptGenerator (Causal GRU)
           - LID logits [B, 1500, 3]
           - lid_probs  [B, 1500, 3]  (softmax)
               ↓
         [Frame-level Soft-Routed Language-Aware Adapters]
           vi_adapter(encoder_out) × p_vi  +
           en_adapter(encoder_out) × p_en
               ↓
         encoder_out = encoder_out + mixed_adaptation
               ↓
         [Whisper Decoder] → logits

FSR-LAA (Frame-level Soft-Routed Language-Aware Adapters):
  - Extends traditional LAA from utterance-level to frame-level routing
  - Uses CausalPromptGenerator (GRU) for streaming-compatible LID
  - Soft routing via LID probabilities enables intra-sentential CS handling
  - Each adapter is a lightweight bottleneck (768 → 64 → 768)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.integrations.huggingface.whisper import Whisper
from speechbrain.nnet.normalization import LayerNorm


class CausalPromptGenerator(nn.Module):
    """
    Causal GRU-based Language ID predictor for streaming code-switching ASR.
    Reused from dsp_model.py — same architecture, adapted for Whisper dims.
    """

    def __init__(self, input_size=768, hidden_size=256, num_languages=3, dropout=0.1):
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

        # Stability init
        for name, param in self.gru.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        nn.init.normal_(self.lid_head.weight, std=0.01)
        nn.init.zeros_(self.lid_head.bias)

    def forward(self, features, hx=None):
        gru_out, hx_new = self.gru(features, hx)
        gru_out = self.norm(self.dropout(gru_out))
        gru_out = gru_out.clamp(min=-5, max=5)
        lid_logits = self.lid_head(gru_out)
        return lid_logits, gru_out, hx_new


class LanguageAdapter(nn.Module):
    """
    Lightweight bottleneck adapter for a specific language.
    Architecture: LayerNorm → Down-project → GELU → Dropout → Up-project
    
    Initialized near-zero so at the start of training the model
    behaves as if the adapter is not there (safe identity start).
    
    Arguments
    ---------
    input_size : int
        Input/output dimension (768 for whisper-small).
    bottleneck_size : int
        Hidden bottleneck dimension (default 64, ~0.1M params per adapter).
    dropout : float
        Dropout rate inside the adapter.
    """

    def __init__(self, input_size=768, bottleneck_size=64, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.down_proj = nn.Linear(input_size, bottleneck_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.up_proj = nn.Linear(bottleneck_size, input_size)

        # Safe identity start: adapter outputs ~0 initially
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        """Returns the adaptation delta (NOT x + delta)."""
        h = self.layer_norm(x)
        h = self.down_proj(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.up_proj(h)
        return h


class DSP_Whisper(nn.Module):
    """
    DSP-LAA: Frame-level Soft-Routed Language-Aware Adapters on Whisper.

    Combines:
      - Whisper encoder-decoder (SpeechBrain integration)
      - LoRA adapters on encoder attention layers (shared adaptation)
      - CausalPromptGenerator for per-frame LID probabilities
      - 2 Language-Aware Adapters (VI, EN) soft-routed by LID probs

    Arguments
    ---------
    source : str
        HuggingFace model ID (e.g., "openai/whisper-small").
    save_path : str
        Local cache path for the backbone.
    input_size : int
        Whisper encoder output dim (768 for whisper-small).
    hidden_size : int
        GRU hidden size in CausalPromptGenerator.
    num_languages : int
        LID class count (SIL=0, VI=1, EN=2).
    adapter_size : int
        Bottleneck dimension for Language-Aware Adapters.
    lora_rank : int
        LoRA rank for encoder adaptation.
    lora_alpha : float
        LoRA scaling factor.
    dropout : float
        Dropout rate.
    language : str
        Whisper language token (e.g., "vi" for Vietnamese).
    """

    def __init__(
        self,
        source="openai/whisper-small",
        save_path="pretrained_models",
        input_size=768,
        hidden_size=256,
        num_languages=3,
        adapter_size=64,
        lora_rank=16,
        lora_alpha=32.0,
        dropout=0.1,
        language="vi",
    ):
        super().__init__()
        self.warmup = True
        self.input_size = input_size

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

        # 2. Freeze all Whisper params first, then selectively unfreeze
        for param in self.whisper.parameters():
            param.requires_grad = False

        # 3. Apply LoRA to encoder attention layers (shared adaptation)
        # Whisper-small encoder has 12 layers, each with self_attn (k,v,q,out_proj)
        from speechbrain.nnet.adapters import AdaptedModel, LoRA

        lora_target_layers = []
        for i in range(12):  # whisper-small has 12 encoder layers
            p = f"layers.{i}.self_attn"
            lora_target_layers += [
                f"{p}.k_proj",
                f"{p}.v_proj",
                f"{p}.q_proj",
                f"{p}.out_proj",
            ]

        self.whisper.model.encoder = AdaptedModel(
            model_to_adapt=self.whisper.model.encoder,
            adapter_class=LoRA,
            all_linear=False,
            target_layers=lora_target_layers,
            adapter_kwargs={
                "rank": lora_rank,
                "alpha": lora_alpha,
            },
        )

        # 4. Unfreeze encoder LayerNorms (LoRA + LN Trick)
        for name, param in self.whisper.model.encoder.named_parameters():
            if "layer_norm" in name:
                param.requires_grad = True

        # 5. Unfreeze decoder (fine-tune decoder fully for seq2seq)
        for param in self.whisper.model.decoder.parameters():
            param.requires_grad = True

        # 6. CausalPromptGenerator (LID Router)
        self.prompt_generator = CausalPromptGenerator(
            input_size=input_size,
            hidden_size=hidden_size,
            num_languages=num_languages,
            dropout=dropout,
        )

        # 7. Language-Aware Adapters (FSR-LAA)
        # Each adapter: 768 → adapter_size → 768 (bottleneck)
        self.vi_adapter = LanguageAdapter(
            input_size=input_size,
            bottleneck_size=adapter_size,
            dropout=dropout,
        )
        self.en_adapter = LanguageAdapter(
            input_size=input_size,
            bottleneck_size=adapter_size,
            dropout=dropout,
        )

    @property
    def model(self):
        return self.whisper.model

    @property
    def tokenizer(self):
        return self.whisper.tokenizer

    @property
    def bos(self):
        return self.whisper.bos

    @property
    def eos(self):
        return self.whisper.eos

    @property
    def bos_lm(self):
        return self.whisper.bos_lm

    @property
    def bos_prev(self):
        return self.whisper.bos_prev

    @property
    def no_speech(self):
        return self.whisper.no_speech

    @property
    def transcribe(self):
        return self.whisper.transcribe

    @property
    def translate(self):
        return self.whisper.translate

    @property
    def non_speech_tokens(self):
        return self.whisper.non_speech_tokens

    @property
    def config(self):
        return self.whisper.config

    @property
    def get_suppress_tokens(self):
        return self.whisper.get_suppress_tokens

    def set_task(self, task):
        self.whisper.set_task(task)

    def forward(self, wav, decoder_input_ids):
        """
        Arguments
        ---------
        wav : torch.Tensor
            Raw waveform [batch, time_samples]. 16kHz mono.
        decoder_input_ids : torch.Tensor
            Token IDs for decoder input [batch, seq_len].

        Returns
        -------
        logits : torch.Tensor [batch, seq_len, vocab_size]
        lid_logits : torch.Tensor [batch, 1500, num_languages]
        """
        # 1. Whisper encoder (with LoRA shared adaptation)
        mel = self.whisper._get_mel(wav)
        encoder_out = self.whisper.forward_encoder(mel)  # [B, 1500, 768]

        # 2. CausalPromptGenerator → LID probabilities
        gru_input = encoder_out.detach() if self.warmup else encoder_out
        lid_logits, _, _ = self.prompt_generator(gru_input)
        lid_probs = torch.softmax(lid_logits, dim=-1)  # [B, 1500, 3]

        # 3. FSR-LAA: Frame-level Soft-Routed Language-Aware Adapters
        # lid_probs channels: [SIL=0, VI=1, EN=2]
        p_vi = lid_probs[:, :, 1:2]  # [B, 1500, 1]
        p_en = lid_probs[:, :, 2:3]  # [B, 1500, 1]

        vi_adapt = self.vi_adapter(encoder_out)  # [B, 1500, 768]
        en_adapt = self.en_adapter(encoder_out)  # [B, 1500, 768]

        # Soft mix: SIL frames get no adaptation (p_vi + p_en ≈ 0)
        adaptation = p_vi * vi_adapt + p_en * en_adapt
        encoder_out = encoder_out + adaptation

        # 4. Whisper decoder (Bypass wrapper for eLAL cross_attention)
        output_states = self.whisper.model.decoder(
            encoder_hidden_states=encoder_out,
            input_ids=decoder_input_ids,
            output_attentions=True,
            use_cache=False,
            return_dict=True
        )
        logits = (
            output_states.last_hidden_state
            @ torch.transpose(self.whisper.model.decoder.embed_tokens.weight.to(encoder_out.dtype), 0, 1)
        ).float()
        
        # Lấy cross-attention cho LAL [B, num_heads, tgt_len, src_len]
        cross_attn = output_states.cross_attentions[-1]

        return logits, lid_logits, cross_attn

    def get_encoder_out(self, wav):
        """Get encoder output + LAA adaptation (for beam search)."""
        mel = self.whisper._get_mel(wav)
        encoder_out = self.whisper.forward_encoder(mel)

        gru_input = encoder_out.detach() if self.warmup else encoder_out
        lid_logits, _, _ = self.prompt_generator(gru_input)
        lid_probs = torch.softmax(lid_logits, dim=-1)

        p_vi = lid_probs[:, :, 1:2]
        p_en = lid_probs[:, :, 2:3]

        vi_adapt = self.vi_adapter(encoder_out)
        en_adapt = self.en_adapter(encoder_out)

        adaptation = p_vi * vi_adapt + p_en * en_adapt
        encoder_out = encoder_out + adaptation

        return encoder_out, lid_logits

    def decode_step(self, encoder_out, decoder_input_ids, past_key_values=None):
        """Single decoder step (for autoregressive generation)."""
        logits, attn, past_kv = self.whisper.forward_decoder(
            encoder_out, decoder_input_ids,
            use_cache=True, past_key_values=past_key_values,
        )
        return logits, attn, past_kv

    def forward_decoder(
        self, encoder_out, decoder_input_ids, use_cache=True, past_key_values=None
    ):
        """Expose Whisper decoder API for SpeechBrain searchers."""
        return self.whisper.forward_decoder(
            encoder_out,
            decoder_input_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )
