"""
AG Whisper Baseline (Aditya et al., ICASSP 2024)
==================================================
Re-implementation of "Attention-Guided Adaptation for 
Code-Switching Speech Recognition"

Architecture:
  Audio → [Whisper Encoder + Adapter (φ_e)] → encoder_out
               ↓
         [Whisper Decoder + Adapter (φ_d)] → output + self-attention maps
               ↓
         Head Selection: top K heads with LID token attention pattern
               ↓
         AG Loss: L_AG = Σ||A_selected - G(y)||²
               ↓
         L = L_CE + γ·L_AG  (γ=0.01)

Training: 2-stage
  Stage 1: Train encoder adapter (φ_e) only → L_CE (8 epochs)
  Stage 2: Train encoder + decoder adapter → L_CE + γ·L_AG (8 epochs)

Key: Decoder uses BILINGUAL prompt: <|sot|><|vi|><|en|><|transcribe|><|notimestamps|>
"""

import torch
import torch.nn as nn
from speechbrain.integrations.huggingface.whisper import Whisper


class LanguageAdapter(nn.Module):
    """Serial bottleneck adapter (Houlsby et al., 2019)."""

    def __init__(self, input_size=768, bottleneck_size=192, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.down_proj = nn.Linear(input_size, bottleneck_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.up_proj = nn.Linear(bottleneck_size, input_size)

        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        h = self.layer_norm(x)
        h = self.down_proj(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.up_proj(h)
        return h


class AG_Whisper(nn.Module):
    """
    Whisper + Encoder/Decoder Adapters + Bilingual LID Prompt.
    
    The AG Loss is computed externally in the training script,
    because it requires access to decoder self-attention maps
    and ground-truth language labels.
    
    Key difference from paper: We use LoRA instead of serial adapters
    for encoder to be fair with V5 comparison. Decoder gets serial adapters.
    """

    def __init__(
        self,
        source="openai/whisper-small",
        save_path="pretrained_models",
        input_size=768,
        adapter_size=192,
        lora_rank=16,
        lora_alpha=32.0,
        dropout=0.1,
        language="vi",
        num_decoder_layers=12,
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
            attn_implementation="eager",  # Force eager attention to fix SDPA cache bug
        )

        # 2. Freeze all
        for param in self.whisper.parameters():
            param.requires_grad = False

        # 3. LoRA on encoder (same as V5)
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

        # 6. Decoder adapters (serial, after each decoder layer)
        # These are applied externally via hooks
        self.dec_adapters = nn.ModuleList([
            LanguageAdapter(input_size, adapter_size, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # 7. Register hooks on decoder layers to inject adapters
        self._adapter_hooks = []
        for i, layer in enumerate(self.whisper.model.decoder.layers):
            hook = layer.register_forward_hook(self._make_adapter_hook(i))
            self._adapter_hooks.append(hook)

        # 8. Build bilingual prompt: [sot, vi, en, transcribe, notimestamps]
        tokenizer = self.whisper.tokenizer
        self._vi_token = tokenizer.convert_tokens_to_ids("<|vi|>")
        self._en_token = tokenizer.convert_tokens_to_ids("<|en|>")
        
        # Original prefix: [sot, vi, transcribe, notimestamps]
        # We add en after vi: [sot, vi, en, transcribe, notimestamps]
        orig_prefix = tokenizer.prefix_tokens  # [sot, vi, transcribe, notimestamps]
        self._bilingual_prefix = [
            orig_prefix[0],     # sot
            self._vi_token,     # vi
            self._en_token,     # en
            orig_prefix[2],     # transcribe
            orig_prefix[3],     # notimestamps
        ]
        
        # Positions of LID tokens in bilingual prefix (0-indexed)
        self._lid_positions = [1, 2]  # vi at pos 1, en at pos 2

    def _make_adapter_hook(self, layer_idx):
        """Create a forward hook that injects decoder adapter."""
        adapter = self.dec_adapters[layer_idx]
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
                h = h + adapter(h)
                return (h,) + output[1:]
            else:
                h = output
                h = h + adapter(h)
                return h
        return hook

    # --- Properties ---
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

    @property
    def bilingual_prefix(self):
        return self._bilingual_prefix

    @property
    def lid_positions(self):
        return self._lid_positions

    def forward(self, wav, decoder_input_ids):
        """
        Returns
        -------
        logits : [B, seq_len, vocab]
        self_attns : list of [B, heads, seq_len, seq_len] (12 layers)
        """
        # 1. Encoder
        mel = self.whisper._get_mel(wav)
        encoder_out = self.whisper.forward_encoder(mel)

        # 2. Decoder with self-attention extraction
        output = self.whisper.model.decoder(
            encoder_hidden_states=encoder_out,
            input_ids=decoder_input_ids,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

        logits = (
            output.last_hidden_state
            @ torch.transpose(
                self.whisper.model.decoder.embed_tokens.weight.to(encoder_out.dtype), 0, 1
            )
        ).float()

        # Collect self-attention maps from all decoder layers
        self_attns = list(output.attentions)  # list of [B, H, tgt, tgt]

        return logits, self_attns

    def get_encoder_out(self, wav):
        """For beam search inference."""
        mel = self.whisper._get_mel(wav)
        return self.whisper.forward_encoder(mel)

    def forward_decoder(self, encoder_out, decoder_input_ids,
                        use_cache=True, past_key_values=None):
        """Standard decoder forward for beam search."""
        return self.whisper.forward_decoder(
            encoder_out, decoder_input_ids,
            use_cache=use_cache, past_key_values=past_key_values,
        )
