"""
DSP W2V-BERT Code-Switching ASR Model
======================================
Architecture:
  Audio → [LoRA-Adapted W2V-BERT-2.0] → features [B, T, 1024]
               ↓
          CausalPromptGenerator (Causal GRU)
            - Reads features stream-by-stream (for real-time LID)
            - Outputs LID logits [B, T, 3] and GRU hidden state (for streaming)

Adaptation Strategy:
  - LoRA (Low-Rank Adaptation) via SpeechBrain's AdaptedModel is injected into ALL
    Linear layers of the W2V-BERT backbone — the industry-standard PEFT approach.
  - LoRA rank=16 means ~0.5% of backbone params are trainable.
  - Reference: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., ICLR 2022)

SpeechBrain Dependency:
  - This file depends on `speechbrain` installed as a pip package (NOT embedded source).
  - Install: pip install speechbrain>=1.0.0
  - No need to clone the SpeechBrain repo; all imports are from the installed package.
"""

import torch
import torch.nn as nn
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from speechbrain.nnet.adapters import AdaptedModel, LoRA
from speechbrain.nnet.normalization import LayerNorm


class CausalPromptGenerator(nn.Module):
    """
    Causal GRU-based Language ID (LID) predictor for streaming code-switching ASR.

    Sole responsibility:
      1. Predict per-frame language labels (VI / EN / SIL)
      2. Maintain causal hidden state for streaming chunk-by-chunk inference

    Arguments
    ---------
    input_size : int
        Backbone feature dimension (1024 for w2v-bert-2.0).
    hidden_size : int
        GRU hidden dimension (256).
    num_languages : int
        LID class count — 3: VI=1, EN=2, SIL=0.
    dropout : float
        Dropout probability after GRU.
    """

    def __init__(self, input_size=1024, hidden_size=256, num_languages=3, dropout=0.1):
        super().__init__()

        # Causal GRU: bidirectional=False → no future chunk seen (streaming safe)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=False,
        )

        # Dropout → LayerNorm (from SpeechBrain LibriSpeech seq2seq recipe)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(input_size=hidden_size)

        # LID head: [B, T, hidden] → [B, T, num_languages]
        self.lid_head = nn.Linear(hidden_size, num_languages)

    def forward(self, features, hx=None):
        """
        Arguments
        ---------
        features : torch.Tensor [batch, time, input_size]
        hx : torch.Tensor [1, batch, hidden_size], optional

        Returns
        -------
        lid_logits : torch.Tensor [batch, time, num_languages]
        hx_new : torch.Tensor [1, batch, hidden_size]
        """
        gru_out, hx_new = self.gru(features, hx)
        gru_out = self.norm(self.dropout(gru_out))
        lid_logits = self.lid_head(gru_out)
        return lid_logits, hx_new


class DSP_W2VBERT(nn.Module):
    """
    Semi-Supervised Causal Streaming Code-Switching ASR model.

    Combines:
      - W2V-BERT-2.0 backbone with LoRA PEFT adapters (via SpeechBrain AdaptedModel)
      - CausalPromptGenerator for per-frame LID + streaming GRU state

    Arguments
    ---------
    source : str
        HuggingFace model ID.
    save_path : str
        Local cache path for backbone weights.
    input_size : int
        Backbone output dim (1024).
    hidden_size : int
        GRU hidden size (256).
    num_languages : int
        LID class count (3).
    lora_rank : int
        LoRA rank r. Recommended: 8–32.
    lora_alpha : float
        LoRA scaling = alpha / rank.
    dropout : float
        Dropout in CausalPromptGenerator.
    """

    def __init__(
        self,
        source="facebook/w2v-bert-2.0",
        save_path="pretrained_models",
        input_size=1024,
        hidden_size=256,
        num_languages=3,
        lora_rank=16,
        lora_alpha=1.0,
        dropout=0.1,
    ):
        super().__init__()

        # 1. Load backbone (freeze=False — AdaptedModel handles freezing internally)
        base_backbone = HuggingFaceWav2Vec2(
            source=source,
            save_path=save_path,
            freeze=False,
        )

        # 2. Wrap with LoRA via SpeechBrain AdaptedModel
        #    - Freezes all backbone params (requires_grad=False)
        #    - Replaces each Linear with LoRA(Linear) → adds trainable rank-r matrices
        #    - Checkpointable: saves only LoRA delta weights
        self.backbone = AdaptedModel(
            model_to_adapt=base_backbone,
            adapter_class=LoRA,
            all_linear=True,
            adapter_kwargs={"rank": lora_rank, "alpha": lora_alpha},
        )

        # 3. Causal LID predictor + streaming state
        self.prompt_generator = CausalPromptGenerator(
            input_size=input_size,
            hidden_size=hidden_size,
            num_languages=num_languages,
            dropout=dropout,
        )

    def forward(self, wav, wav_lens=None, hx=None):
        """
        Arguments
        ---------
        wav : torch.Tensor [batch, time_samples]  16kHz mono
        wav_lens : torch.Tensor [batch]  relative lengths in [0, 1]
        hx : torch.Tensor [1, batch, hidden_size]  GRU state (None = first chunk)

        Returns
        -------
        adapted_features : torch.Tensor [batch, time_frames, 1024]
        lid_logits : torch.Tensor [batch, time_frames, num_languages]
        hx_new : torch.Tensor [1, batch, hidden_size]
        """
        # LoRA-adapted acoustic feature extraction
        adapted_features = self.backbone(wav, wav_lens)

        # Streaming LID prediction
        lid_logits, hx_new = self.prompt_generator(adapted_features, hx)

        return adapted_features, lid_logits, hx_new
