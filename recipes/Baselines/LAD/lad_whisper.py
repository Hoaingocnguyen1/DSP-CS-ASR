"""
LAD Whisper Baseline (Zhao et al., ICASSP 2025)
=================================================
Re-implementation of "Adapting Whisper for Code-Switching through
Encoding Refining and Language-Aware Decoding"

Architecture (Proper — adapters INSIDE each decoder layer):
  Audio → [Whisper Encoder + LoRA] → encoder_out [B, 1500, 768]
               ↓
         Encoding Refiner (GRU substitute for LSTM)
               ↓ (CTC Loss on encoder)
         encoder_out_refined
               ↓
         [Whisper Decoder — MODIFIED layers]
           Per layer:
             h + p_vi → SharedSelfAttn → h_vi_sa
             h + p_en → SharedSelfAttn → h_en_sa
             h_vi_sa  → SharedCrossAttn → h_vi_ca → vi_adapter → adapted_vi
             h_en_sa  → SharedCrossAttn → h_en_ca → en_adapter → adapted_en
               ↓
         Fusion at output:
           w_vi, w_en = sigmoid(Linear(h_vi)), sigmoid(Linear(h_en))
           h_fused = w_vi * h_vi + w_en * h_en
               ↓
         logits [B, seq_len, vocab]

Key differences from simplified V9:
  - Adapters inside EACH decoder layer (not just at output)
  - Single decoder forward (not running decoder twice)
  - Encoding Refiner (GRU + CTC) on encoder output
  - Learned fusion (sigmoid-gated) instead of GRU lid_probs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.integrations.huggingface.whisper import Whisper
from speechbrain.nnet.normalization import LayerNorm


class LanguageAdapter(nn.Module):
    """Lightweight bottleneck adapter for LAD per decoder layer."""

    def __init__(self, input_size=768, bottleneck_size=64, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(input_size)
        self.down_proj = nn.Linear(input_size, bottleneck_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.up_proj = nn.Linear(bottleneck_size, input_size)

        # Near-zero init for safe identity start
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        """Returns adaptation delta (caller adds residual)."""
        h = self.layer_norm(x)
        h = self.down_proj(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.up_proj(h)
        return h


class LADDecoderLayerWrapper(nn.Module):
    """Wraps a single WhisperDecoderLayer to add dual-path LAD adapters.
    
    For each layer:
    1. Run shared self-attention with VI language prompt embedding
    2. Run shared self-attention with EN language prompt embedding
    3. Run shared cross-attention for both paths
    4. Apply language-specific adapters to each path
    5. Run shared MLP for both paths
    
    The wrapper replaces the original layer's forward method.
    """

    def __init__(self, original_layer, adapter_size=64, input_size=768, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        
        # Language-specific adapters (after cross-attention)
        self.vi_adapter = LanguageAdapter(input_size, adapter_size, dropout)
        self.en_adapter = LanguageAdapter(input_size, adapter_size, dropout)

    def forward(
        self,
        hidden_states,          # h_vi, h_en concatenated as tuple or stacked
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=True,
        **kwargs,
    ):
        """Dual-path forward through wrapped decoder layer.
        
        We run the original layer's sub-modules manually to inject adapters.
        """
        layer = self.original_layer
        
        # If not in LAD mode (inference/beam search), use original layer directly
        if not isinstance(hidden_states, tuple):
            return layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
        
        # LAD mode: unpack dual paths
        h_vi, h_en = hidden_states
        
        # === Self-Attention (shared weights, different inputs) ===
        # VI path
        residual_vi = h_vi
        h_vi = layer.self_attn_layer_norm(h_vi)
        outputs_vi = layer.self_attn(
            hidden_states=h_vi,
            attention_mask=attention_mask,
            past_key_value=None,
            output_attentions=False,
        )
        h_vi = residual_vi + outputs_vi[0]
        
        # EN path
        residual_en = h_en
        h_en = layer.self_attn_layer_norm(h_en)
        outputs_en = layer.self_attn(
            hidden_states=h_en,
            attention_mask=attention_mask,
            past_key_value=None,
            output_attentions=False,
        )
        h_en = residual_en + outputs_en[0]
        
        # === Cross-Attention (shared weights, shared encoder) ===
        cross_attn_weights_vi = None
        cross_attn_weights_en = None
        
        # VI path
        residual_vi = h_vi
        h_vi = layer.encoder_attn_layer_norm(h_vi)
        outputs_vi = layer.encoder_attn(
            hidden_states=h_vi,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=None,
            output_attentions=output_attentions,
        )
        h_vi = residual_vi + outputs_vi[0]
        if output_attentions and len(outputs_vi) > 1:
            cross_attn_weights_vi = outputs_vi[1]
        
        # EN path
        residual_en = h_en
        h_en = layer.encoder_attn_layer_norm(h_en)
        outputs_en = layer.encoder_attn(
            hidden_states=h_en,
            key_value_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            layer_head_mask=cross_attn_layer_head_mask,
            past_key_value=None,
            output_attentions=output_attentions,
        )
        h_en = residual_en + outputs_en[0]
        if output_attentions and len(outputs_en) > 1:
            cross_attn_weights_en = outputs_en[1]
        
        # === Language-Specific Adapters (after cross-attention) ===
        h_vi = h_vi + self.vi_adapter(h_vi)
        h_en = h_en + self.en_adapter(h_en)
        
        # === Feed-Forward (shared weights) ===
        # VI path
        residual_vi = h_vi
        h_vi = layer.final_layer_norm(h_vi)
        h_vi = layer.activation_fn(layer.fc1(h_vi))
        h_vi = layer.fc2(h_vi)
        h_vi = residual_vi + h_vi
        
        # EN path
        residual_en = h_en
        h_en = layer.final_layer_norm(h_en)
        h_en = layer.activation_fn(layer.fc1(h_en))
        h_en = layer.fc2(h_en)
        h_en = residual_en + h_en
        
        outputs = ((h_vi, h_en),)
        if output_attentions:
            outputs += (cross_attn_weights_vi,)
        
        return outputs


class LAD_Whisper(nn.Module):
    """
    LAD: Language-Aware Decoding for Code-Switching ASR.
    Proper re-implementation of Zhao et al. (ICASSP 2025).
    
    Adapters are injected INSIDE each decoder layer (not just at output).
    """

    def __init__(
        self,
        source="openai/whisper-small",
        save_path="pretrained_models",
        input_size=768,
        num_languages=3,
        adapter_size=64,
        lora_rank=16,
        lora_alpha=32.0,
        dropout=0.1,
        language="vi",
        num_decoder_layers=12,
    ):
        super().__init__()
        self.input_size = input_size
        self.num_decoder_layers = num_decoder_layers

        # 1. Load Whisper backbone
        self.whisper = Whisper(
            source=source,
            save_path=save_path,
            encoder_only=False,
            freeze=False,
            freeze_encoder=False,
            language=language,
            task="transcribe",
            attn_implementation="eager",
        )

        # 2. Freeze all
        for param in self.whisper.parameters():
            param.requires_grad = False

        # 3. LoRA on encoder
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

        # 5. Unfreeze decoder LayerNorms only (Adapters will be added next and are trainable by default)
        for name, param in self.whisper.model.decoder.named_parameters():
            if "layer_norm" in name:
                param.requires_grad = True

        # 6. Wrap each decoder layer with LAD adapters
        self.lad_wrappers = nn.ModuleList()
        decoder = self.whisper.model.decoder
        for i in range(num_decoder_layers):
            wrapper = LADDecoderLayerWrapper(
                original_layer=decoder.layers[i],
                adapter_size=adapter_size,
                input_size=input_size,
                dropout=dropout,
            )
            self.lad_wrappers.append(wrapper)

        # 7. Encoding Refiner (GRU as substitute for paper's LSTM)
        self.encoding_refiner = nn.GRU(
            input_size=input_size,
            hidden_size=input_size // 2,
            batch_first=True,
            bidirectional=True,
            num_layers=2,
            dropout=dropout,
        )
        self.refiner_proj = nn.Linear(input_size, input_size)
        
        # 8. CTC head for Encoding Refiner auxiliary loss
        tokenizer = self.whisper.tokenizer
        vocab_size = tokenizer.get_vocab().__len__() if hasattr(tokenizer, 'get_vocab') else 51865
        self.ctc_head = nn.Linear(input_size, vocab_size)

        # 9. Language prompt embeddings (learnable, added to decoder input)
        self.vi_prompt_emb = nn.Parameter(torch.randn(1, 1, input_size) * 0.01)
        self.en_prompt_emb = nn.Parameter(torch.randn(1, 1, input_size) * 0.01)

        # 10. LID Module for Fusion (supervised via token-level loss)
        self.lid_head = nn.Linear(input_size, num_languages)
        nn.init.xavier_uniform_(self.lid_head.weight)
        nn.init.zeros_(self.lid_head.bias)

        # Cache language token IDs for prompt swapping
        self._vi_lang_token = tokenizer.convert_tokens_to_ids("<|vi|>")
        self._en_lang_token = tokenizer.convert_tokens_to_ids("<|en|>")

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

    def _run_lad_decoder(self, encoder_out, decoder_input_ids):
        """Run decoder with dual-path LAD through wrapped layers.
        
        Returns
        -------
        h_fused : [B, seq_len, 768]
        cross_attn : attention weights from last layer (VI path)
        ctc_logits : [B, T_enc, vocab] for CTC auxiliary loss
        """
        decoder = self.whisper.model.decoder
        
        # Token embeddings + positional encoding
        input_embeds = decoder.embed_tokens(decoder_input_ids)
        # Scale by sqrt(d_model) if Whisper uses it
        # hidden_states = input_embeds * (self.input_size ** 0.5)  # Whisper doesn't scale
        hidden_states = input_embeds
        
        # Add positional embeddings
        positions = decoder.embed_positions(decoder_input_ids)
        hidden_states = hidden_states + positions
        hidden_states = F.dropout(hidden_states, p=decoder.dropout, training=self.training)
        
        # Create causal attention mask
        seq_len = decoder_input_ids.shape[1]
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=decoder_input_ids.device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)
        
        # Split into VI and EN paths with language prompt embeddings
        h_vi = hidden_states + self.vi_prompt_emb.expand(hidden_states.size(0), hidden_states.size(1), -1)
        h_en = hidden_states + self.en_prompt_emb.expand(hidden_states.size(0), hidden_states.size(1), -1)
        
        # Process through wrapped decoder layers
        cross_attn_last = None
        for i, wrapper in enumerate(self.lad_wrappers):
            outputs = wrapper(
                (h_vi, h_en),
                attention_mask=causal_mask,
                encoder_hidden_states=encoder_out,
                output_attentions=(i == self.num_decoder_layers - 1),
                use_cache=False,
            )
            h_vi, h_en = outputs[0]
            if len(outputs) > 1 and outputs[1] is not None:
                cross_attn_last = outputs[1]
        
        # Layer norm
        h_vi = decoder.layer_norm(h_vi)
        h_en = decoder.layer_norm(h_en)
        
        # Fusion: Language-Aware Decoding based on LID probabilities
        # We compute LID logits by combining paths loosely (or just from one, paper says built upon cross attn)
        # Using a simple addition is symmetric and captures both contexts for LID prediction
        lid_logits = self.lid_head(h_vi + h_en)  # [B, seq, num_languages]
        lid_probs = F.softmax(lid_logits, dim=-1)  # [B, seq, num_languages]
        
        # lid_probs channels: [SIL=0, VI=1, EN=2] based on dataio_prepare lid_map
        w_vi = lid_probs[:, :, 1:2]  # [B, seq, 1] - Prob of VI
        w_en = lid_probs[:, :, 2:3]  # [B, seq, 1] - Prob of EN
        
        h_fused = w_vi * h_vi + w_en * h_en
        
        # CTC logits from encoder (Encoding Refiner)
        ctc_logits = self.ctc_head(encoder_out)
        
        return h_fused, cross_attn_last, ctc_logits, lid_logits

    def forward(self, wav, decoder_input_ids):
        """
        Returns
        -------
        logits : [B, seq_len, vocab]
        ctc_logits : [B, T_enc, vocab] for CTC loss
        cross_attn : attention weights
        """
        # 1. Encoder
        mel = self.whisper._get_mel(wav)
        encoder_out = self.whisper.forward_encoder(mel)  # [B, 1500, 768]

        # 2. Encoding Refiner
        refined, _ = self.encoding_refiner(encoder_out)
        encoder_out = encoder_out + self.refiner_proj(refined)  # Residual connection

        # 3. LAD Decoder
        h_fused, cross_attn, ctc_logits, lid_logits = self._run_lad_decoder(encoder_out, decoder_input_ids)

        # 4. Project to vocabulary
        logits = (
            h_fused
            @ torch.transpose(
                self.whisper.model.decoder.embed_tokens.weight.to(encoder_out.dtype), 0, 1
            )
        ).float()

        return logits, ctc_logits, cross_attn, lid_logits

    def get_encoder_out(self, wav):
        """For beam search inference (standard decoder, no LAD)."""
        mel = self.whisper._get_mel(wav)
        encoder_out = self.whisper.forward_encoder(mel)
        # Apply refiner
        refined, _ = self.encoding_refiner(encoder_out)
        encoder_out = encoder_out + self.refiner_proj(refined)
        return encoder_out

    def forward_decoder(self, encoder_out, decoder_input_ids,
                        use_cache=True, past_key_values=None):
        """Standard decoder forward for beam search (no LAD)."""
        return self.whisper.forward_decoder(
            encoder_out, decoder_input_ids,
            use_cache=use_cache, past_key_values=past_key_values,
        )
