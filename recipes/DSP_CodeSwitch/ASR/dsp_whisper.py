"""
DSP Whisper Code-Switching ASR Model
=====================================
Architecture:
  Audio → [Whisper Encoder + LoRA] → encoder_out [B, 1500, 768]
               ↓
         CausalPromptGenerator (Causal GRU)
           - LID logits [B, 1500, 3]
           - Acoustic Prompt [B, 1500, 256]
               ↓
         [Feature Injection]
         encoder_out = encoder_out + Tanh(Gate) * Linear(Prompt)
               ↓
         [Whisper Decoder] → logits

DSP-CS method applied to Whisper Small encoder-decoder backbone.
Injection point: between encoder and decoder outputs.
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


class DSP_Whisper(nn.Module):
    """
    DSP-CS method applied to Whisper Small backbone.

    Combines:
      - Whisper encoder-decoder (SpeechBrain integration)
      - LoRA adapters on encoder attention layers
      - CausalPromptGenerator for per-frame LID
      - Tanh gate injection between encoder and decoder

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
        LID class count (VI=1, EN=2, SIL=0).
    lora_rank : int
        LoRA rank for encoder adaptation.
    lora_alpha : float
        LoRA scaling factor.
    dropout : float
        Dropout in CausalPromptGenerator.
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

        # 3. Apply LoRA to encoder attention layers
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

        # 6. CausalPromptGenerator
        self.prompt_generator = CausalPromptGenerator(
            input_size=input_size,
            hidden_size=hidden_size,
            num_languages=num_languages,
            dropout=dropout,
        )

        # 7. Tanh Gate Injection (V1 architecture)
        self.prompt_proj = nn.Linear(hidden_size, input_size)
        nn.init.xavier_uniform_(self.prompt_proj.weight)
        nn.init.zeros_(self.prompt_proj.bias)
        self.prompt_dropout = nn.Dropout(p=dropout)
        self.gate = nn.Parameter(torch.zeros(input_size))

    def forward(self, wav, decoder_input_ids, wav_lens=None):
        """
        Arguments
        ---------
        wav : torch.Tensor
            Raw waveform [batch, time_samples]. 16kHz mono.
        decoder_input_ids : torch.Tensor
            Token IDs for decoder input [batch, seq_len].
        wav_lens : torch.Tensor, optional
            Relative lengths (not used by Whisper, kept for API compat).

        Returns
        -------
        logits : torch.Tensor [batch, seq_len, vocab_size]
        lid_logits : torch.Tensor [batch, 1500, num_languages]
        """
        # 1. Mel spectrogram + Whisper encoder
        mel = self.whisper._get_mel(wav)
        encoder_out = self.whisper.forward_encoder(mel)  # [B, 1500, 768]

        # 2. CausalPromptGenerator: LID + Acoustic Prompt
        gru_input = encoder_out.detach() if self.warmup else encoder_out
        lid_logits, prompt, _ = self.prompt_generator(gru_input)

        # 3. Tanh Gate Injection (V1 method)
        # injection = tanh(gate) * prompt_proj(dropout(prompt))
        # At init: gate=0 → tanh(0)=0 → no injection (safe identity)
        prompt_projected = self.prompt_dropout(self.prompt_proj(prompt))
        injection = torch.tanh(self.gate) * prompt_projected
        encoder_out = encoder_out + injection

        # 4. Whisper decoder
        logits, attn, _ = self.whisper.forward_decoder(
            encoder_out, decoder_input_ids
        )

        return logits, lid_logits

    def get_encoder_out(self, wav):
        """Get encoder output + DSP injection (for beam search)."""
        mel = self.whisper._get_mel(wav)
        encoder_out = self.whisper.forward_encoder(mel)

        gru_input = encoder_out.detach() if self.warmup else encoder_out
        lid_logits, prompt, _ = self.prompt_generator(gru_input)

        prompt_projected = self.prompt_dropout(self.prompt_proj(prompt))
        injection = torch.tanh(self.gate) * prompt_projected
        encoder_out = encoder_out + injection

        return encoder_out, lid_logits

    def decode_step(self, encoder_out, decoder_input_ids, past_key_values=None):
        """Single decoder step (for autoregressive generation)."""
        logits, attn, past_kv = self.whisper.forward_decoder(
            encoder_out, decoder_input_ids,
            use_cache=True, past_key_values=past_key_values,
        )
        return logits, attn, past_kv
