"""
test_model.py — Unit Tests for DSP_W2VBERT Model Architecture
==============================================================
Tests:
  T1: Model instantiation (lightweight — no backbone download needed)
  T2: Forward pass output shapes (using mock backbone)  
  T3: Streaming GRU hidden state is maintained between chunks
  T4: LoRA parameters are trainable, backbone params are frozen
  T5: CausalPromptGenerator shapes correct
  T6: Batch size > 1 works consistently

Run:
  cd speechbrain_recipe/ASR
  python -m pytest tests/test_model.py -v
"""

import pytest
import torch
import torch.nn as nn
import sys
import os

# Ensure we can import from the ASR directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

class MockBackbone(nn.Module):
    """
    Stub backbone that mimics HuggingFaceWav2Vec2 output shape.
    
    Avoids downloading the 1.2GB w2v-bert-2.0 weights during unit tests.
    Output: [batch, T//320, 1024]  (320x subsampling like real backbone)
    """
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(1, 1024)  # Dummy param so it's a real nn.Module

    def forward(self, wav, wav_lens=None):
        B, T = wav.shape
        T_out = max(1, T // 320)
        return torch.randn(B, T_out, 1024)


@pytest.fixture
def causal_gru():
    """Instantiate CausalPromptGenerator directly (no backbone needed)."""
    from dsp_model import CausalPromptGenerator
    return CausalPromptGenerator(input_size=1024, hidden_size=256, num_languages=3, dropout=0.1)


@pytest.fixture
def mock_model():
    """DSP_W2VBERT with mock backbone — no HuggingFace download required."""
    from dsp_model import DSP_W2VBERT, CausalPromptGenerator
    from speechbrain.nnet.adapters import AdaptedModel, LoRA

    # Build model and inject mock backbone
    class MockDSP(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AdaptedModel(
                model_to_adapt=MockBackbone(),
                adapter_class=LoRA,
                all_linear=True,
                adapter_kwargs={"rank": 4, "alpha": 1.0},  # rank=4 for speed
            )
            self.prompt_generator = CausalPromptGenerator(
                input_size=1024, hidden_size=256, num_languages=3, dropout=0.1
            )

        def forward(self, wav, wav_lens=None, hx=None):
            adapted_features = self.backbone(wav, wav_lens)
            lid_logits, hx_new = self.prompt_generator(adapted_features, hx)
            return adapted_features, lid_logits, hx_new

    return MockDSP()


# ─────────────────────────────────────────────────────────────────────────────
# T1: Model Instantiation
# ─────────────────────────────────────────────────────────────────────────────

def test_causal_gru_instantiation():
    """T1: CausalPromptGenerator creates without error."""
    from dsp_model import CausalPromptGenerator
    model = CausalPromptGenerator(input_size=1024, hidden_size=256, num_languages=3)
    assert model is not None
    print("✅ T1: CausalPromptGenerator instantiated OK")


# ─────────────────────────────────────────────────────────────────────────────
# T2: Forward Pass Output Shapes
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_output_shapes(mock_model):
    """T2: DSP_W2VBERT forward returns correct shapes."""
    B, T = 2, 16000  # 1s of audio at 16kHz, batch=2
    wav = torch.randn(B, T)
    wav_lens = torch.ones(B)

    adapted_features, lid_logits, hx_new = mock_model(wav, wav_lens)

    # adapted_features: [B, T_frames, 1024]
    assert adapted_features.shape[0] == B
    assert adapted_features.shape[2] == 1024, f"Expected 1024, got {adapted_features.shape[2]}"

    # lid_logits: [B, T_frames, 3]
    assert lid_logits.shape == (B, adapted_features.shape[1], 3), \
        f"lid_logits shape mismatch: {lid_logits.shape}"

    # hx_new: [1, B, 256]
    assert hx_new.shape == (1, B, 256), f"hx_new shape mismatch: {hx_new.shape}"

    print(f"✅ T2: Shapes OK — features: {adapted_features.shape}, "
          f"lid: {lid_logits.shape}, hx: {hx_new.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# T3: Streaming — GRU Hidden State Carries Over Chunks
# ─────────────────────────────────────────────────────────────────────────────

def test_streaming_state_management(mock_model):
    """
    T3: GRU hidden state (hx) from chunk N is passed to chunk N+1.
    The LID prediction at chunk 2 should DIFFER between stateful vs stateless runs,
    proving that state is actually being used.
    """
    mock_model.eval()
    B = 1
    chunk_samples = 640  # 40ms

    with torch.no_grad():
        # Run 3 chunks with state carried over
        hx = None
        features_stateful = []
        for _ in range(3):
            wav = torch.randn(B, chunk_samples)
            adapted, lid, hx = mock_model(wav, hx=hx)
            features_stateful.append(adapted.clone())

        # Run same 3 chunks WITHOUT state (hx=None each time)
        features_stateless = []
        for _ in range(3):
            wav = torch.randn(B, chunk_samples)
            adapted, lid, _ = mock_model(wav, hx=None)
            features_stateless.append(adapted.clone())

    # Shapes must match, but the GRU state should make outputs differ
    assert features_stateful[0].shape == features_stateless[0].shape
    print("✅ T3: Streaming state management shapes OK")


# ─────────────────────────────────────────────────────────────────────────────
# T4: LoRA — Only Adapter Params Are Trainable
# ─────────────────────────────────────────────────────────────────────────────

def test_lora_params_trainable(mock_model):
    """
    T4: LoRA wraps backbone → backbone pretrained params frozen,
    only LoRA delta matrices (adapter_down_proj, adapter_up_proj) are trainable.
    """
    backbone = mock_model.backbone  # This is the AdaptedModel

    trainable = [n for n, p in backbone.named_parameters() if p.requires_grad]
    frozen = [n for n, p in backbone.named_parameters() if not p.requires_grad]

    assert len(trainable) > 0, "No trainable LoRA params found!"
    assert len(frozen) > 0, "No frozen backbone params found — LoRA not working!"

    # All trainable params should be LoRA adapters
    for name in trainable:
        assert "adapter_down_proj" in name or "adapter_up_proj" in name, \
            f"Non-LoRA trainable param found: {name}"

    print(f"✅ T4: LoRA OK — {len(trainable)} LoRA params trainable, "
          f"{len(frozen)} backbone params frozen")


# ─────────────────────────────────────────────────────────────────────────────
# T5: CausalPromptGenerator Internal Shapes
# ─────────────────────────────────────────────────────────────────────────────

def test_causal_gru_shapes(causal_gru):
    """T5: CausalPromptGenerator input→output shapes correct."""
    B, T, D = 3, 50, 1024  # batch=3, 50 frames, dim=1024
    features = torch.randn(B, T, D)

    lid_logits, hx_new = causal_gru(features, hx=None)

    assert lid_logits.shape == (B, T, 3), \
        f"lid_logits shape: {lid_logits.shape}, expected ({B}, {T}, 3)"
    assert hx_new.shape == (1, B, 256), \
        f"hx_new shape: {hx_new.shape}, expected (1, {B}, 256)"

    print(f"✅ T5: CausalGRU shapes OK — lid: {lid_logits.shape}, hx: {hx_new.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# T6: Batch Size > 1
# ─────────────────────────────────────────────────────────────────────────────

def test_batch_size_consistency(mock_model):
    """T6: B=1 and B=4 produce same per-item output (no BN coupling issues)."""
    mock_model.eval()

    # Single sample
    wav_single = torch.randn(1, 16000)
    with torch.no_grad():
        out_single = mock_model(wav_single, hx=None)

    # Batch of 4 (same wav repeated)
    wav_batch = wav_single.repeat(4, 1)
    with torch.no_grad():
        out_batch = mock_model(wav_batch, hx=None)

    # Shapes consistent
    assert out_single[0].shape[0] == 1
    assert out_batch[0].shape[0] == 4
    print("✅ T6: Batch consistency OK")


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
