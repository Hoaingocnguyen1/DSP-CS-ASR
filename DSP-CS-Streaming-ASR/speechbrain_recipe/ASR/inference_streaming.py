#!/usr/bin/env python3
"""
Streaming Inference Demo (CPU-Only)
===================================
Simulates a real-time streaming environment by reading an audio file 
chunk by chunk (e.g., 40ms) and passing it through the quantized ONNX model.

The hidden state of the Causal Prompt Generator (GRU) is continuously 
updated and passed along, preventing border artifacts.

Requirements:
  pip install onnxruntime torchaudio numpy
"""

import os
import argparse
import numpy as np
import onnxruntime as ort
import torchaudio

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_model", type=str, default="dsp_model_int8.onnx", help="Path to Quantized ONNX model")
    parser.add_argument("--wav_file", type=str, required=True, help="Path to a 16kHz test WAV file")
    parser.add_argument("--chunk_ms", type=int, default=40, help="Simulated streaming chunk size in milliseconds")
    args = parser.parse_args()

    if not os.path.exists(args.onnx_model):
        print(f"Error: ONNX model {args.onnx_model} not found. Please run export_onnx.py first.")
        return

    # Load ONNX Runtime Session (CPU Provider)
    print(f"Loading ONNX Model: {args.onnx_model} on CPU...")
    # NOTE: Set threads to 1 or 2 for lightweight embedded devices
    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = 2
    session = ort.InferenceSession(args.onnx_model, sess_opt, providers=['CPUExecutionProvider'])

    # Load Audio
    waveform, sample_rate = torchaudio.load(args.wav_file)
    if sample_rate != 16000:
        import torchaudio.transforms as T
        print(f"Resampling from {sample_rate} to 16000...")
        waveform = T.Resample(sample_rate, 16000)(waveform)
    
    # Warning #13 fix: squeeze(0) to safely handle stereo → mono conversion
    # waveform.squeeze() fails on stereo [2, N] → slices channels instead of time
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio_np = waveform.squeeze(0).numpy()  # [N]
    
    # Calculate chunk size in samples
    chunk_samples = int(16000 * (args.chunk_ms / 1000.0))
    total_samples = len(audio_np)
    
    print(f"\n🎧 Starting Streaming Inference Simulation...")
    print(f"Audio Length: {total_samples/16000:.2f}s | Chunk Size: {args.chunk_ms}ms ({chunk_samples} samples)\n")

    # STATE MANAGEMENT: Initialize Hidden State for Causal GRU
    # Shape matches export: [num_layers=1, batch_size=1, hidden_size=256]
    hx = np.zeros((1, 1, 256), dtype=np.float32)


    for start_idx in range(0, total_samples, chunk_samples):
        end_idx = min(start_idx + chunk_samples, total_samples)
        chunk = audio_np[start_idx:end_idx]
        
        # Pad final chunk if it's smaller than chunk_samples (optional but safer for some backbones)
        if len(chunk) < chunk_samples:
            chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')
            
        # Format inputs for ONNX
        # Warning #15 fix: Pass actual relative length, not hardcoded 1.0,
        # so the backbone correctly masks padding in the final (shorter) chunk.
        actual_len = (end_idx - start_idx) / chunk_samples  # relative: 0.0 - 1.0
        ort_inputs = {
            'wav': np.expand_dims(chunk, axis=0).astype(np.float32),  # [1, chunk_size]
            'wav_lens': np.array([min(actual_len, 1.0)], dtype=np.float32),
            'hx_in': hx
        }
        
        # Run Inference
        # Warning #14 fix: renamed `adapted_features` → `token_logits` to reflect
        # the ONNX graph now outputs vocabulary logits (EndToEndASRWrapper), not raw features.
        token_logits, lid_logits, hx_out = session.run(None, ort_inputs)
        
        # UPDATE STATE: Carry over the hidden state to the next chunk!
        hx = hx_out
        
        # Simple print progress
        lid_class = lid_logits.argmax(axis=-1).flatten()  # [T] class indices
        lid_labels = {0: 'SIL', 1: 'VI', 2: 'EN'}
        # Use mode (most frequent class) for dominant language, not mean of class indices
        dominant_lid = int(np.bincount(lid_class).argmax())
        dominant_lang = lid_labels.get(dominant_lid, '?')
        print(f"Chunk [{start_idx/16000:.2f}s - {end_idx/16000:.2f}s] "
              f"→ token_logits: {token_logits.shape}, "
              f"dominant_lang: {dominant_lang}")

    print("\n✅ Streaming Inference Complete.")
    print("Notice how the `hx_in` state was passed chunk-to-chunk seamlessly!")

if __name__ == "__main__":
    main()
