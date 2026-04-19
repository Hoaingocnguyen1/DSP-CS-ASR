import torch
import torch.nn as nn
from speechbrain.integrations.huggingface.whisper import Whisper
from speechbrain.nnet.adapters import AdaptedModel, LoRA

class Vanilla_LoRA_Whisper(nn.Module):
    """
    Standard LoRA applied to Whisper Small backbone.
    This serves as the pure baseline to compare against DSP-CS.
    - Whisper encoder-decoder
    - LoRA adapters on encoder attention layers
    - Fully fine-tuned decoder
    - Constant injection mapping (None)
    """

    def __init__(
        self,
        source="openai/whisper-small",
        save_path="pretrained_models",
        lora_rank=16,
        lora_alpha=32.0,
        language="vi",
    ):
        super().__init__()

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

        # 2. Freeze all Whisper params first
        for param in self.whisper.parameters():
            param.requires_grad = False

        # 3. Apply LoRA to encoder attention layers EXACTLY like DSP_Whisper
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

        # 4. Unfreeze encoder LayerNorms
        for name, param in self.whisper.model.encoder.named_parameters():
            if "layer_norm" in name:
                param.requires_grad = True

        # 5. Unfreeze decoder fully (exact match to DSP_Whisper strategy)
        for param in self.whisper.model.decoder.parameters():
            param.requires_grad = True

    @property
    def tokenizer(self):
        return self.whisper.tokenizer
        
    @property
    def model(self):
        return self.whisper.model
        
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

    def _get_mel(self, wav):
        return self.whisper._get_mel(wav)

    def forward_encoder(self, mel):
        return self.whisper.forward_encoder(mel)

    def forward_decoder(self, encoder_out, decoder_input_ids, use_cache=True, past_key_values=None):
        return self.whisper.forward_decoder(
            encoder_out,
            decoder_input_ids,
            use_cache=use_cache,
            past_key_values=past_key_values,
        )

    def set_task(self, task):
        self.whisper.set_task(task)
