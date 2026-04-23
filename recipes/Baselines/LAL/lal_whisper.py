"""
LAL Whisper Baseline (Liu et al., 2024)
========================================
Re-implementation of "Aligning Speech to Languages to Enhance 
Code-switching Speech Recognition" (arXiv:2403.05887)

Architecture:
  Audio → [Whisper Encoder + LoRA] → encoder_out [B, 1500, 768]
               ↓
         nn.Linear(768, 3) → lid_logits (frame-level LID)
               ↓
         L_LAL = CrossEntropy(lid_logits, pseudo_labels)
         (pseudo_labels from cross-attention alignment)
               ↓
         [Whisper Decoder] → logits
               ↓
         L = (1-β)·L_seq + β·L_LAL

Key differences from DSP-CS-ASR V5 (eLAL):
  - Linear LID head (not GRU)
  - CrossEntropy loss (not Focal Loss)
  - No confidence masking
  - No language adapters (no Soft-Routing)
  - Fixed β (not scheduled)
"""

import torch
import torch.nn as nn
from speechbrain.integrations.huggingface.whisper import Whisper


class LAL_Whisper(nn.Module):
    """
    Whisper + LoRA + Language Alignment Loss (LAL).

    Faithful re-implementation of Liu et al. (2024) on Whisper-small.
    Only a single linear layer is used for frame-level LID prediction.
    No language-specific adapters. No GRU. No Focal Loss.
    """

    def __init__(
        self,
        source="openai/whisper-small",
        save_path="pretrained_models",
        input_size=768,
        num_languages=3,
        lora_rank=16,
        lora_alpha=32.0,
        language="vi",
    ):
        super().__init__()
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

        # 2. Freeze all, then unfreeze selectively
        for param in self.whisper.parameters():
            param.requires_grad = False

        # 3. LoRA on encoder (same as V5 for fair comparison)
        from speechbrain.nnet.adapters import AdaptedModel, LoRA

        lora_target_layers = []
        for i in range(12):
            p = f"layers.{i}.self_attn"
            lora_target_layers += [
                f"{p}.k_proj", f"{p}.v_proj",
                f"{p}.q_proj", f"{p}.out_proj",
            ]

        self.whisper.model.encoder = AdaptedModel(
            model_to_adapt=self.whisper.model.encoder,
            adapter_class=LoRA,
            all_linear=False,
            target_layers=lora_target_layers,
            adapter_kwargs={"rank": lora_rank, "alpha": lora_alpha},
        )

        # 4. Unfreeze encoder LayerNorms
        for name, param in self.whisper.model.encoder.named_parameters():
            if "layer_norm" in name:
                param.requires_grad = True

        # 5. Unfreeze decoder
        for param in self.whisper.model.decoder.parameters():
            param.requires_grad = True

        # 6. LAL: Simple Linear LID head (Paper 4 exact)
        self.lid_head = nn.Linear(input_size, num_languages)
        nn.init.normal_(self.lid_head.weight, std=0.01)
        nn.init.zeros_(self.lid_head.bias)

    # --- Properties (same interface as DSP_Whisper) ---
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
        Returns
        -------
        logits : [B, seq_len, vocab]
        lid_logits : [B, 1500, num_languages]
        cross_attn : [B, heads, tgt_len, src_len]
        """
        # 1. Encoder
        mel = self.whisper._get_mel(wav)
        encoder_out = self.whisper.forward_encoder(mel)  # [B, 1500, 768]

        # 2. LAL: Linear LID head (no GRU!)
        lid_logits = self.lid_head(encoder_out)  # [B, 1500, 3]

        # 3. Decoder (NO adaptation on encoder_out — pure Whisper)
        output_states = self.whisper.model.decoder(
            encoder_hidden_states=encoder_out,
            input_ids=decoder_input_ids,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )
        logits = (
            output_states.last_hidden_state
            @ torch.transpose(
                self.whisper.model.decoder.embed_tokens.weight.to(encoder_out.dtype), 0, 1
            )
        ).float()

        cross_attn = output_states.cross_attentions[-1]

        return logits, lid_logits, cross_attn

    def get_encoder_out(self, wav):
        """For beam search inference."""
        mel = self.whisper._get_mel(wav)
        encoder_out = self.whisper.forward_encoder(mel)
        lid_logits = self.lid_head(encoder_out)
        return encoder_out, lid_logits

    def forward_decoder(self, encoder_out, decoder_input_ids,
                        use_cache=True, past_key_values=None):
        """Expose Whisper decoder API for SpeechBrain searchers."""
        return self.whisper.forward_decoder(
            encoder_out, decoder_input_ids,
            use_cache=use_cache, past_key_values=past_key_values,
        )
