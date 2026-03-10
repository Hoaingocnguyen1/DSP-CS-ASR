#!/usr/bin/env python3
"""
ONNX Export and Quantization Script
===================================
Exports the trained DSP_W2VBERT model (W2V-BERT-2.0 Backbone + Causal GRU) 
to ONNX format and applies INT8 Quantization.

This allows the model to run efficiently on **CPU-only** devices 
(like Raspberry Pi, Web Browsers, or low-cost servers) with minimal latency,
fulfilling the strict streaming requirements of the thesis.

Usage:
  python export_onnx.py --hparams train.yaml
"""

import os
import sys
import torch
from torch.onnx import TrainingMode
import argparse
from hyperpyyaml import load_hyperpyyaml
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

# Import your custom model
from dsp_model import DSP_W2VBERT

class EndToEndASRWrapper(torch.nn.Module):
    """
    Wraps the acoustic model (DSP_W2VBERT) and a CTC linear decoder layer
    so the ONNX graph outputs direct probability logits.
    
    IMPORTANT: Uses ctc_lin (input_size=1024 → vocab), NOT seq_lin (512 → vocab).
    The adapted_features from DSP_W2VBERT are [B, T, 1024], so a 512-dim decoder
    linear would cause a shape mismatch crash at runtime.
    """
    def __init__(self, acoustic_model, ctc_linear):
        super().__init__()
        self.acoustic_model = acoustic_model
        self.ctc_linear = ctc_linear

    def forward(self, wav, wav_lens, hx):
        # 1. Acoustic feature extraction & prompt adaptation
        adapted_features, lid_logits, hx_out = self.acoustic_model(wav, wav_lens, hx)
        
        # 2. CTC-style projection: [B, T, 1024] → [B, T, vocab_size]
        # Uses ctc_lin (1024 → vocab), not seq_lin (512 → vocab)
        token_logits = self.ctc_linear(adapted_features)
        
        return token_logits, lid_logits, hx_out


def export_to_onnx(model, output_path="dsp_model.onnx"):
    print("Preparing model for ONNX tracing...")
    model.eval()
    
    # Create dummy inputs
    # For streaming, we pass a 40ms chunk (e.g. 640 samples at 16kHz)
    dummy_wav = torch.randn(1, 640)
    dummy_wav_lens = torch.tensor([1.0])
    
    # Hidden state hx for GRU: [num_layers, batch_size, hidden_size]
    # dsp_model uses a 1-layer GRU with hidden_size 256
    dummy_hx = torch.zeros(1, 1, 256)

    print(f"Exporting to {output_path}...")
    
    # Using opset_version 17 to support newer operations in w2v-bert-2.0
    torch.onnx.export(
        model, 
        (dummy_wav, dummy_wav_lens, dummy_hx), 
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        # Easy Fix #1: Force EVAL mode so Dropout is frozen (p=0) in graph.
        # Without this, dropout may be traced with train=True, adding randomness.
        training=TrainingMode.EVAL,
        input_names=['wav', 'wav_lens', 'hx_in'],
        output_names=['token_logits', 'lid_logits', 'hx_out'],
        dynamic_axes={
            'wav': {0: 'batch_size', 1: 'time'},
            'wav_lens': {0: 'batch_size'},
            'hx_in': {1: 'batch_size'},
            'token_logits': {0: 'batch_size', 1: 'time_features'},
            'lid_logits': {0: 'batch_size', 1: 'time_features'},
            'hx_out': {1: 'batch_size'}
        }
    )
    
    print("✅ ONNX Export with Streaming States Successful!")
    return output_path

def quantize_onnx_model(onnx_path, quantized_path="dsp_model_int8.onnx"):
    print(f"\nApplying INT8 Dynamic Quantization to {onnx_path}...")
    try:
        quantize_dynamic(
            model_input=onnx_path,
            model_output=quantized_path,
            weight_type=QuantType.QUInt8,
            optimize_model=True
        )
        print(f"✅ Quantization Successful! Saved to {quantized_path}")
        
        # Print size savings
        orig_size = os.path.getsize(onnx_path) / (1024 * 1024)
        quant_size = os.path.getsize(quantized_path) / (1024 * 1024)
        print(f"Original Size:  {orig_size:.2f} MB")
        print(f"Quantized Size: {quant_size:.2f} MB (Compression: {orig_size/quant_size:.2f}x)")
        
    except Exception as e:
        print(f"❌ Quantization failed. Ensure 'onnxruntime' is installed. Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", type=str, default="train.yaml", help="Path to hyperparameter YAML file")
    parser.add_argument("--onnx_path", type=str, default="dsp_model.onnx", help="Output path for FP32 ONNX")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to the SpeechBrain checkpoint directory (e.g. results/DSP_W2VBERT_CodeSwitch/2025/save)")
    parser.add_argument("--quantize", action="store_true", default=True, help="Whether to apply INT8 quantization")
    args = parser.parse_args()

    print(f"Initializing End-to-End Wrapper from {args.hparams}...")
    
    with open(args.hparams) as fin:
        hparams = load_hyperpyyaml(fin)

    import speechbrain as sb
    acoustic_model = hparams["dsp_w2vbert"]
    # Medium Fix #4: Use ctc_lin (input_size=1024 → vocab) NOT seq_lin (512 → vocab).
    # EndToEndASRWrapper feeds adapted_features [B,T,1024] directly to the linear layer.
    # Using seq_lin (512-dim input) would crash with a dimension mismatch.
    ctc_linear = hparams["ctc_lin"]
    
    # BUG FIX #3: Load trained weights from checkpoint.
    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=args.checkpoint_dir,
        recoverables={
            "dsp_w2vbert": acoustic_model,
            "ctc_lin": ctc_linear,  # Medium Fix #4: load ctc_lin not seq_lin
        }
    )
    # Recover best checkpoint (min WER)
    ckpt = checkpointer.recover_if_possible()
    if ckpt is None:
        raise RuntimeError(
            f"No checkpoint found in {args.checkpoint_dir}. "
            "Please train first or check the --checkpoint_dir path."
        )
    print(f"✅ Loaded checkpoint from: {args.checkpoint_dir}")
    
    model = EndToEndASRWrapper(acoustic_model, ctc_linear)
    
    # Export Flow
    fp32_onnx = export_to_onnx(model, args.onnx_path)
    
    if args.quantize:
        quantized_onnx = args.onnx_path.replace(".onnx", "_int8.onnx")
        quantize_onnx_model(fp32_onnx, quantized_onnx)
    
    print("\n🏁 Finished. The quantized model can now be run on CPU via onnxruntime.InferenceSession!")

