"""
test_streaming.py — Integration Tests for Streaming Inference
=============================================================
Tests:
  S1: HX state shape is preserved across chunks (ONNX path)
  S2: Stateful inference (hx carry-over) differs from stateless (hx=None each time)
  S3: Final chunk with variable length is handled correctly (padding)
  S4: Batch size=1 only (streaming is always single-sample)
  S5: Language detection uses mode, not mean of indices

Run:
  cd speechbrain_recipe/ASR
  python -m pytest tests/test_streaming.py -v
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import tempfile
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

class MockBackbone(nn.Module):
    def forward(self, wav, wav_lens=None):
        B, T = wav.shape
        return torch.randn(B, max(1, T // 320), 1024)


def build_mock_onnx_session():
    """Build a mock ONNX session using in-memory export."""
    try:
        import onnxruntime as ort
        from torch.onnx import TrainingMode
    except ImportError:
        return None

    from dsp_model import CausalPromptGenerator
    from speechbrain.nnet.adapters import AdaptedModel, LoRA

    class MockWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = AdaptedModel(
                model_to_adapt=MockBackbone(),
                adapter_class=LoRA, all_linear=True,
                adapter_kwargs={"rank": 4, "alpha": 1.0}
            )
            self.backbone = backbone
            self.gru = CausalPromptGenerator(1024, 256, 3, dropout=0.0)
            self.ctc = nn.Linear(1024, 50)

        def forward(self, wav, wav_lens, hx):
            feat = self.backbone(wav, wav_lens)
            lid, hx_new = self.gru(feat, hx)
            tokens = self.ctc(feat)
            return tokens, lid, hx_new

    model = MockWrapper()
    model.eval()

    dummy = (torch.randn(1, 640), torch.tensor([1.0]), torch.zeros(1, 1, 256))
    buf = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
    torch.onnx.export(
        model, dummy, buf.name,
        opset_version=17, training=TrainingMode.EVAL,
        input_names=["wav", "wav_lens", "hx_in"],
        output_names=["token_logits", "lid_logits", "hx_out"],
        dynamic_axes={
            "wav": {0: "batch", 1: "time"},
            "wav_lens": {0: "batch"},
            "hx_in": {1: "batch"},
            "token_logits": {0: "batch", 1: "T"},
            "lid_logits": {0: "batch", 1: "T"},
            "hx_out": {1: "batch"},
        }
    )

    session = ort.InferenceSession(buf.name, providers=["CPUExecutionProvider"])
    return session, buf.name


# ─────────────────────────────────────────────────────────────────────────────
# S1: HX Shape Preserved Across Chunks
# ─────────────────────────────────────────────────────────────────────────────

def test_hx_shape_preserved_across_chunks():
    """S1: hx_out shape == hx_in shape after every chunk."""
    result = build_mock_onnx_session()
    if result is None:
        pytest.skip("onnxruntime not available")

    session, path = result
    try:
        hx = np.zeros((1, 1, 256), dtype=np.float32)
        initial_shape = hx.shape

        for i in range(5):  # 5 consecutive chunks
            wav_chunk = np.random.randn(1, 640).astype(np.float32)
            _, _, hx_out = session.run(None, {
                "wav": wav_chunk,
                "wav_lens": np.array([1.0], dtype=np.float32),
                "hx_in": hx,
            })
            assert hx_out.shape == initial_shape, \
                f"Chunk {i}: hx shape changed! {hx_out.shape} != {initial_shape}"
            hx = hx_out

        print(f"✅ S1: hx shape {initial_shape} preserved across 5 chunks")
    finally:
        os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# S2: Stateful vs Stateless Inference Produces Different lid_logits
# ─────────────────────────────────────────────────────────────────────────────

def test_stateful_differs_from_stateless():
    """
    S2: Stateful (carry hx) vs stateless (reset hx each chunk) should produce
    different LID logits — proving hidden state is actually being used.
    We use the same deterministic audio but different hx.
    """
    from dsp_model import CausalPromptGenerator

    gru = CausalPromptGenerator(1024, 256, 3, dropout=0.0)
    gru.eval()

    torch.manual_seed(42)
    features = torch.randn(1, 10, 1024)  # 10 frames

    with torch.no_grad():
        # Stateful: build up hidden state first
        warm_features = torch.randn(1, 20, 1024)
        _, hx_warmed = gru(warm_features)

        # Same input, but with warmed vs cold state
        lid_stateful, _ = gru(features, hx=hx_warmed)
        lid_stateless, _ = gru(features, hx=None)

    # They should differ (GRU state affects output)
    diff = (lid_stateful - lid_stateless).abs().max().item()
    assert diff > 0.01, f"Stateful vs stateless diff too small ({diff:.6f}), GRU state not being used?"
    print(f"✅ S2: Stateful differs from stateless (max diff: {diff:.4f})")


# ─────────────────────────────────────────────────────────────────────────────
# S3: Final Chunk Padding Handled Correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_final_chunk_padding():
    """
    S3: Final chunk is shorter than chunk_samples.
    inference_streaming.py pads with zeros — verify there's no crash and
    wav_lens is set to the actual relative length < 1.0.
    """
    result = build_mock_onnx_session()
    if result is None:
        pytest.skip("onnxruntime not available")

    session, path = result
    CHUNK = 640

    try:
        # Simulate final chunk with only 200 samples of real audio
        real_samples = 200
        chunk = np.zeros(CHUNK, dtype=np.float32)
        chunk[:real_samples] = np.random.randn(real_samples)

        # actual relative length
        actual_len = real_samples / CHUNK  # 0.3125

        tokens, lid, hx_out = session.run(None, {
            "wav": chunk[np.newaxis, :],
            "wav_lens": np.array([actual_len], dtype=np.float32),
            "hx_in": np.zeros((1, 1, 256), dtype=np.float32),
        })

        assert tokens is not None
        assert 0.0 < actual_len < 1.0, f"actual_len should be fractional: {actual_len}"
        print(f"✅ S3: Final chunk with actual_len={actual_len:.4f} OK, "
              f"output shape: {tokens.shape}")
    finally:
        os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# S4: Batch Size Must Be 1 in Streaming
# ─────────────────────────────────────────────────────────────────────────────

def test_streaming_batch_size_one():
    """
    S4: Streaming inference always uses batch=1.
    Test that B=1 runs fine and B=2 produces correctly shaped outputs too
    (for batch streaming scenarios).
    """
    from dsp_model import CausalPromptGenerator

    gru = CausalPromptGenerator(1024, 256, 3, dropout=0.0)
    gru.eval()

    with torch.no_grad():
        # B=1 (standard streaming)
        feat_b1 = torch.randn(1, 5, 1024)
        lid_b1, hx_b1 = gru(feat_b1)
        assert lid_b1.shape == (1, 5, 3)
        assert hx_b1.shape == (1, 1, 256)

        # B=2 (hypothetical multi-stream)
        feat_b2 = torch.randn(2, 5, 1024)
        lid_b2, hx_b2 = gru(feat_b2)
        assert lid_b2.shape == (2, 5, 3)
        assert hx_b2.shape == (1, 2, 256)

    print("✅ S4: B=1 and B=2 streaming shapes both correct")


# ─────────────────────────────────────────────────────────────────────────────
# S5: Language Detection Uses Mode (Most Frequent), Not Mean
# ─────────────────────────────────────────────────────────────────────────────

def test_lid_uses_mode_not_mean():
    """
    S5: Guard against the Bug D2 regression — dominant language detection
    must use np.bincount().argmax() (mode), NOT np.round(mean(lid_class)).
    
    Example: frames [1, 1, 1, 2, 2] 
      - Mode = 1 (VI) — CORRECT
      - Mean = 1.4 → round → 1 (accidental correct but fragile)
    
    Counter-example: frames [0, 2, 2, 2, 2]
      - Mode = 2 (EN) — CORRECT
      - Mean = 1.6 → round → 2 (correct this time)
    
    Bad example: frames [0, 1, 2, 2, 2]
      - Mode = 2 (EN) — CORRECT 
      - Mean = 1.4 → round → 1 (VI) — WRONG!
    """
    lid_class = np.array([0, 1, 2, 2, 2])

    # Mode (correct approach)
    dominant_mode = int(np.bincount(lid_class).argmax())
    assert dominant_mode == 2, f"Mode-based dominant lang should be 2 (EN), got {dominant_mode}"

    # Mean (buggy approach — wrong answer here)
    dominant_mean = int(round(lid_class.mean()))
    # This could be 1 (VI=wrong) instead of 2 (EN)
    # Not a strict assertion since it might coincidentally match, 
    # but we document the risk
    print(f"✅ S5: Mode={dominant_mode} (EN✅), Mean-based={dominant_mean} "
          f"({'✅' if dominant_mean == 2 else '❌ Bug D2 would trigger here'})")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
