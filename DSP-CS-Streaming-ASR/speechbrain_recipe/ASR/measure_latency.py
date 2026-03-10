#!/usr/bin/env python3
"""
Streaming Inference Latency Measurement
=========================================
Measures the real-time factor (RTF) and per-chunk latency (ms/chunk)
for the DSP_W2VBERT model in streaming (causal, chunk-by-chunk) mode.

This script is used to populate the Latency column in the thesis
(Table 3 — Proposed Method). Baseline models are offline (no streaming),
so latency is only reported for our proposed method.

Usage:
    python speechbrain_recipe/ASR/measure_latency.py \
        --hparams  speechbrain/recipes/DSP_CodeSwitch/ASR/train.yaml \
        --wav      path/to/test_audio.wav \
        --chunk_ms 40 \
        --n_trials 100

Reported Metrics:
    - Chunk latency (ms): time to process a single 40ms chunk
    - RTF (Real-Time Factor): processing_time / audio_duration < 1.0 = real-time capable
    - GRU hidden state overhead: size of state tensor that must be passed between chunks
"""

import argparse
import time
import torch
import torchaudio
import numpy as np
from pathlib import Path


def load_model(hparams_file: str, device: str):
    """Load the DSP_W2VBERT model from a SpeechBrain hparams yaml."""
    from hyperpyyaml import load_hyperpyyaml
    import sys
    sys.path.append(str(Path(hparams_file).parent))   # Add recipe dir for dsp_model import
    from dsp_model import DSP_W2VBERT

    with open(hparams_file, encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f)

    model = hparams["modules"]["dsp_w2vbert"]
    model.to(device).eval()
    print(f"✅ Loaded DSP_W2VBERT (device={device})")
    return model


def load_audio(wav_path: str, target_sr: int = 16000):
    """Load and resample audio to 16kHz mono."""
    wav, sr = torchaudio.load(wav_path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)  # stereo -> mono
    return wav.squeeze(0), target_sr   # [samples]


def stream_chunks(wav: torch.Tensor, sample_rate: int, chunk_ms: int):
    """Split audio into fixed-size chunks for streaming simulation."""
    chunk_samples = int(sample_rate * chunk_ms / 1000)
    total = wav.shape[0]
    chunks = []
    start = 0
    while start < total:
        end = min(start + chunk_samples, total)
        chunk = wav[start:end]
        # Zero-pad the last chunk if smaller
        if chunk.shape[0] < chunk_samples:
            chunk = torch.nn.functional.pad(chunk, (0, chunk_samples - chunk.shape[0]))
        chunks.append(chunk)
        start = end
    return chunks


@torch.no_grad()
def benchmark(model, wav_path: str, chunk_ms: int, n_trials: int, device: str):
    """Run streaming inference benchmark."""
    wav, sr = load_audio(wav_path)
    total_audio_duration_s = wav.shape[0] / sr
    chunks = stream_chunks(wav, sr, chunk_ms)
    chunk_samples = int(sr * chunk_ms / 1000)

    print(f"\n📊 Audio: {wav_path}")
    print(f"   Duration     : {total_audio_duration_s:.2f}s")
    print(f"   Chunk size   : {chunk_ms}ms ({chunk_samples} samples)")
    print(f"   Total chunks : {len(chunks)}")
    print(f"   Trials       : {n_trials}")
    print(f"   Device       : {device}")

    # --- WARMUP (1 pass to pre-compile CUDA kernels if any) ---
    hx = None
    for chunk in chunks:
        wav_in = chunk.unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, T] for SpeechBrain
        wav_in = wav_in.squeeze(1)                           # [1, T]
        wav_lens = torch.tensor([1.0]).to(device)
        _, _, hx = model(wav_in, wav_lens, hx)
    print("\nWarmup complete. Starting benchmark...")

    # --- BENCHMARK ---
    chunk_times = []
    for trial in range(n_trials):
        hx = None  # reset state each trial (simulate fresh audio)
        trial_times = []

        for chunk in chunks:
            wav_in = chunk.unsqueeze(0).to(device)   # [1, T]
            wav_lens = torch.tensor([1.0]).to(device)

            if device == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()

            _, _, hx = model(wav_in, wav_lens, hx)

            if device == "cuda":
                torch.cuda.synchronize()
            t_end = time.perf_counter()

            trial_times.append((t_end - t_start) * 1000)  # ms

        chunk_times.append(np.mean(trial_times))

    # --- RESULTS ---
    mean_chunk_ms = np.mean(chunk_times)
    std_chunk_ms  = np.std(chunk_times)
    total_proc_ms = mean_chunk_ms * len(chunks)
    rtf = total_proc_ms / (total_audio_duration_s * 1000)

    # GRU state overhead
    hx_size_bytes = hx.element_size() * hx.nelement()

    print("\n" + "="*55)
    print("  Streaming Inference Latency Report")
    print("="*55)
    print(f"  Chunk latency (mean) : {mean_chunk_ms:.2f} ± {std_chunk_ms:.2f} ms")
    print(f"  Audio duration       : {total_audio_duration_s*1000:.1f} ms")
    print(f"  Processing time      : {total_proc_ms:.1f} ms ({n_trials} trial avg)")
    print(f"  RTF                  : {rtf:.4f}  ({'✅ Real-time' if rtf < 1.0 else '❌ Not real-time'})")
    print(f"  GRU state (bytes)    : {hx_size_bytes} bytes per stream")
    print("="*55)
    print(f"\n📝 THESIS NOTE: Chunk latency = {mean_chunk_ms:.1f}ms, RTF = {rtf:.4f}")

    return {
        "chunk_ms": chunk_ms,
        "mean_chunk_latency_ms": round(mean_chunk_ms, 2),
        "std_ms": round(std_chunk_ms, 2),
        "rtf": round(rtf, 4),
        "gru_state_bytes": hx_size_bytes,
        "n_trials": n_trials,
        "n_chunks": len(chunks),
        "audio_duration_s": round(total_audio_duration_s, 2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hparams", type=str,
                        default="speechbrain/recipes/DSP_CodeSwitch/ASR/train.yaml",
                        help="Path to train.yaml")
    parser.add_argument("--wav", type=str, default=None,
                        help="Path to a test WAV file. If not provided, uses a synthetic signal.")
    parser.add_argument("--chunk_ms", type=int, default=40,
                        help="Chunk size in milliseconds (default: 40ms)")
    parser.add_argument("--n_trials", type=int, default=50,
                        help="Number of repeated trials for stable averaging")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.wav is None:
        # Generate a 5-second synthetic signal for benchmarking (no real audio needed)
        print("No WAV file provided. Using a 5-second synthetic signal...")
        sr = 16000
        synthetic = torch.randn(sr * 5)   # 5 seconds of noise at 16kHz
        tmp_wav = "/tmp/synthetic_bench.wav"
        torchaudio.save(tmp_wav, synthetic.unsqueeze(0), sr)
        args.wav = tmp_wav

    model = load_model(args.hparams, args.device)
    results = benchmark(model, args.wav, args.chunk_ms, args.n_trials, args.device)

    # Save to JSON for thesis
    import json
    out_path = Path("latency_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
