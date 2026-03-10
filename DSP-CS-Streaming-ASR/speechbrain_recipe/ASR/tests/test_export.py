"""
test_export.py — Integration Tests for ONNX Export
===================================================
Tests:
  E1: EndToEndASRWrapper forward pass shapes (without real backbone)
  E2: ONNX file is valid after export (schema check)
  E3: ONNX output shapes match PyTorch output shapes
  E4: INT8 quantization reduces file size

Prerequisite: pip install onnx onnxruntime

Run:
  cd speechbrain_recipe/ASR
  python -m pytest tests/test_export.py -v
  
Note: E2, E3, E4 require --real flag if you want to test with full backbone.
      By default they use mock inputs and skip backbone download.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Mock Components
# ─────────────────────────────────────────────────────────────────────────────

class MockBackbone(nn.Module):
    """Mock W2V-BERT: [B, T] → [B, T//320, 1024]"""
    def __init__(self):
        super().__init__()
        self._dummy = nn.Linear(1, 1)

    def forward(self, wav, wav_lens=None):
        B, T = wav.shape
        return torch.randn(B, max(1, T // 320), 1024)


def make_mock_model():
    """Build a fast mock version of EndToEndASRWrapper."""
    from speechbrain.nnet.adapters import AdaptedModel, LoRA
    from dsp_model import CausalPromptGenerator

    class MockDSP(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = AdaptedModel(
                model_to_adapt=MockBackbone(),
                adapter_class=LoRA,
                all_linear=True,
                adapter_kwargs={"rank": 4, "alpha": 1.0},
            )
            self.prompt_generator = CausalPromptGenerator(1024, 256, 3, 0.0)

        def forward(self, wav, wav_lens=None, hx=None):
            adapted = self.backbone(wav, wav_lens)
            lid, hx_new = self.prompt_generator(adapted, hx)
            return adapted, lid, hx_new

    class MockWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.acoustic_model = MockDSP()
            self.ctc_linear = nn.Linear(1024, 100)  # 100-token vocab for test

        def forward(self, wav, wav_lens, hx):
            adapted, lid, hx_new = self.acoustic_model(wav, wav_lens, hx)
            token_logits = self.ctc_linear(adapted)
            return token_logits, lid, hx_new

    return MockWrapper()


# ─────────────────────────────────────────────────────────────────────────────
# E1: Wrapper Forward Pass Shapes
# ─────────────────────────────────────────────────────────────────────────────

def test_wrapper_forward_shapes():
    """E1: EndToEndASRWrapper outputs (token_logits, lid_logits, hx)."""
    model = make_mock_model()
    model.eval()

    wav = torch.randn(1, 640)
    wav_lens = torch.tensor([1.0])
    hx = torch.zeros(1, 1, 256)

    with torch.no_grad():
        token_logits, lid_logits, hx_out = model(wav, wav_lens, hx)

    T = token_logits.shape[1]
    assert token_logits.shape == (1, T, 100), f"token_logits: {token_logits.shape}"
    assert lid_logits.shape == (1, T, 3), f"lid_logits: {lid_logits.shape}"
    assert hx_out.shape == (1, 1, 256), f"hx_out: {hx_out.shape}"
    print(f"✅ E1: Wrapper forward OK — tokens: {token_logits.shape}, lid: {lid_logits.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# E2: ONNX Export Produces Valid File
# ─────────────────────────────────────────────────────────────────────────────

def test_onnx_export_valid():
    """E2: Model can be traced to ONNX and schema validates."""
    try:
        import onnx
        from torch.onnx import TrainingMode
    except ImportError:
        pytest.skip("onnx not installed — skipping export test")

    model = make_mock_model()
    model.eval()

    dummy_wav = torch.randn(1, 640)
    dummy_wav_lens = torch.tensor([1.0])
    dummy_hx = torch.zeros(1, 1, 256)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name

    try:
        torch.onnx.export(
            model,
            (dummy_wav, dummy_wav_lens, dummy_hx),
            onnx_path,
            export_params=True,
            opset_version=17,
            training=TrainingMode.EVAL,
            input_names=["wav", "wav_lens", "hx_in"],
            output_names=["token_logits", "lid_logits", "hx_out"],
            dynamic_axes={
                "wav": {0: "batch", 1: "time"},
                "wav_lens": {0: "batch"},
                "hx_in": {1: "batch"},
                "token_logits": {0: "batch", 1: "time_frames"},
                "lid_logits": {0: "batch", 1: "time_frames"},
                "hx_out": {1: "batch"},
            },
        )

        # Validate ONNX schema
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ E2: ONNX export valid — {os.path.getsize(onnx_path) / 1024:.1f} KB")

    finally:
        os.unlink(onnx_path)


# ─────────────────────────────────────────────────────────────────────────────
# E3: ONNX Runtime Output Matches PyTorch
# ─────────────────────────────────────────────────────────────────────────────

def test_onnx_output_matches_pytorch():
    """E3: ONNX runtime output numerically matches PyTorch (atol=1e-4)."""
    try:
        import onnx
        import onnxruntime as ort
        from torch.onnx import TrainingMode
    except ImportError:
        pytest.skip("onnx/onnxruntime not installed")

    model = make_mock_model()
    model.eval()

    wav = torch.randn(1, 640)
    wav_lens = torch.tensor([1.0])
    hx = torch.zeros(1, 1, 256)

    with torch.no_grad():
        pt_tokens, pt_lid, pt_hx = model(wav, wav_lens, hx)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        onnx_path = f.name

    try:
        torch.onnx.export(
            model,
            (wav, wav_lens, hx),
            onnx_path,
            opset_version=17,
            training=TrainingMode.EVAL,
            input_names=["wav", "wav_lens", "hx_in"],
            output_names=["token_logits", "lid_logits", "hx_out"],
        )

        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        ort_inputs = {
            "wav": wav.numpy(),
            "wav_lens": wav_lens.numpy(),
            "hx_in": hx.numpy(),
        }
        ort_tokens, ort_lid, ort_hx = session.run(None, ort_inputs)

        np.testing.assert_allclose(
            pt_tokens.numpy(), ort_tokens, atol=1e-4,
            err_msg="token_logits mismatch between PyTorch and ONNX"
        )
        print(f"✅ E3: ONNX matches PyTorch (max diff: "
              f"{abs(pt_tokens.numpy() - ort_tokens).max():.6f})")

    finally:
        os.unlink(onnx_path)


# ─────────────────────────────────────────────────────────────────────────────
# E4: INT8 Quantization Reduces File Size
# ─────────────────────────────────────────────────────────────────────────────

def test_quantization_reduces_size():
    """E4: INT8 quantization produces smaller file than FP32."""
    try:
        import onnx
        from torch.onnx import TrainingMode
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        pytest.skip("onnxruntime quantization not installed")

    model = make_mock_model()
    model.eval()

    wav = torch.randn(1, 640)
    wav_lens = torch.tensor([1.0])
    hx = torch.zeros(1, 1, 256)

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        fp32_path = f.name
    int8_path = fp32_path.replace(".onnx", "_int8.onnx")

    try:
        torch.onnx.export(
            model, (wav, wav_lens, hx), fp32_path,
            opset_version=17, training=TrainingMode.EVAL,
            input_names=["wav", "wav_lens", "hx_in"],
            output_names=["token_logits", "lid_logits", "hx_out"],
        )

        quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QUInt8)

        fp32_size = os.path.getsize(fp32_path)
        int8_size = os.path.getsize(int8_path)
        ratio = fp32_size / int8_size

        assert int8_size < fp32_size, \
            f"INT8 ({int8_size} bytes) should be smaller than FP32 ({fp32_size} bytes)"
        print(f"✅ E4: Quantization OK — FP32: {fp32_size/1024:.0f}KB → "
              f"INT8: {int8_size/1024:.0f}KB ({ratio:.1f}x compression)")

    finally:
        for p in [fp32_path, int8_path]:
            if os.path.exists(p):
                os.unlink(p)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
