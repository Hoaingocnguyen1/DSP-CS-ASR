#!/usr/bin/env python3
"""
Whisper Teacher Pseudo-labeling Pipeline
========================================
Uses OpenAI's Whisper (Large-v3) to generate pseudo-labels for unlabeled 
Code-Switched speech audio (e.g. from YouTube or Podcasts).

Output is strictly formatted for SpeechBrain:
{
  "pseudo_00001": {
    "wav": "{data_root}/unlabeled_audio/clip_00001.wav",
    "length": 5.2,
    "words": "tôi cần một cái dashboard mới",
    "lid_tokens": "VI VI VI VI EN VI"
  }
}
"""

import argparse
import json
import re
import os
from pathlib import Path
import torch
import torchaudio
from transformers import pipeline

# ---------------------------------------------------------------------
# Vietnamese character detection (from codeswitch-builder)
# ---------------------------------------------------------------------
VI_DIACRITICS = (
    "àáạảãâầấậẩẫăằắặẳẵ" "èéẹẻẽêềếệểễ" "ìíịỉĩ" "òóọỏõôồốộổỗơờớợởỡ" "ùúụủũưừứựửữ" "ỳýỵỷỹđ"
)
VI_REGEX = re.compile(f"[{VI_DIACRITICS}]")

def clean_transcript(text: str) -> str:
    """Ensure transcript is plain lowercase, no language tags."""
    text = text.lower().strip()
    text = re.sub(r"</?(?:en|vi|lang)>", "", text)
    text = re.sub(r"\[/?(?:en|vi|lang)\]", "", text)
    text = re.sub(r"[,\.\?\!]", "", text) # Remove punctuation for clean ASR
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_token(token: str) -> str:
    return re.sub(rf"[^\w{VI_DIACRITICS}]", "", token.lower())

def is_vietnamese(token: str) -> bool:
    return bool(VI_REGEX.search(token))

def is_probable_english(token: str) -> bool:
    if not token.isascii():
        return False
    if len(token) <= 2:
        return False  # avoid misclassifying short VI words
    return token.isalpha()

def classify_tokens(sentence: str):
    """Classify tokens into VI/EN mapping to parallel tokens in transcript."""
    labels = []
    cleaned_sent = clean_transcript(sentence)
    for raw in cleaned_sent.split():
        token = clean_token(raw)
        if not token:
            labels.append("VI") 
            continue
        if is_vietnamese(token):
            labels.append("VI")
        elif is_probable_english(token):
            labels.append("EN")
        else:
            labels.append("VI")
    return " ".join(labels)

def get_duration(wav_path: str) -> float:
    try:
        info = torchaudio.info(wav_path)
        return round(info.num_frames / info.sample_rate, 2)
    except:
        return 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing unlabeled WAV files")
    parser.add_argument("--output_json", type=str, required=True, help="Path to save the JSON manifest")
    parser.add_argument("--model", type=str, default="openai/whisper-large-v3", help="Whisper model to use")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"Error: Directory {audio_dir} does not exist.")
        return

    wav_files = list(audio_dir.glob("*.wav"))
    if not wav_files:
        print(f"No WAV files found in {audio_dir}")
        return

    print(f"Loading Whisper model: {args.model} on {args.device}...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )

    print(f"Loading SpeechBrain VAD model...")
    from speechbrain.inference.VAD import VAD
    vad_model = VAD.from_hparams(
        source="speechbrain/vad-crdnn-libriparty", 
        savedir="pretrained_models/vad-crdnn-libriparty",
        run_opts={"device": args.device}
    )

    dataset_dict = {}
    print(f"Running VAD and Knowledge Distillation on {len(wav_files)} files...")

    # Create a directory to save the segmented short audio chunks
    segmented_dir = audio_dir / "segmented_audio"
    segmented_dir.mkdir(exist_ok=True)

    segment_idx = 0

    for i, wav_path in enumerate(wav_files):
        print(f"Processing {wav_path.name} ({i + 1}/{len(wav_files)})...")
        # 1. Run VAD to get boundaries
        try:
            prob_chunks = vad_model.get_speech_prob_file(str(wav_path))
            prob_th = vad_model.apply_threshold(prob_chunks, activation_th=0.5, deactivation_th=0.25).float()
            boundaries = vad_model.get_boundaries(prob_th)
            # Merge close segments (e.g., less than 0.25s apart) to keep contextual sentences together
            boundaries = vad_model.merge_close_segments(boundaries, close_th=0.250)
            # Remove segments that are incredibly short
            boundaries = vad_model.remove_short_segments(boundaries, len_th=0.5)
        except Exception as e:
            print(f"VAD failed for {wav_path.name}: {e}")
            continue
            
        if boundaries is None or len(boundaries) == 0:
            print(f"No speech found in {wav_path.name}, skipping.")
            continue
            
        # Load audio for slicing
        waveform, sample_rate = torchaudio.load(str(wav_path))
        
        # Easy Fix #3: Batch all chunks from this file into one pipeline call.
        # Previously pipe() was called once per chunk - ignoring batch_size entirely.
        chunk_paths = []
        chunk_meta = []  # (chunk_name, duration) for each saved chunk

        for b in boundaries:
            start_s, end_s = b[0].item(), b[1].item()
            start_sample = int(start_s * sample_rate)
            end_sample = int(end_s * sample_rate)
            chunk_waveform = waveform[:, start_sample:end_sample]
            duration = end_s - start_s
            chunk_name = f"{wav_path.stem}_seg_{segment_idx:06d}.wav"
            chunk_path = segmented_dir / chunk_name
            torchaudio.save(str(chunk_path), chunk_waveform, sample_rate)
            chunk_paths.append(str(chunk_path))
            chunk_meta.append((chunk_name, duration))
            segment_idx += 1  # Reserve the index

        # Run Whisper on all chunks at once (uses batch_size GPU parallelism)
        try:
            results = pipe(chunk_paths, batch_size=args.batch_size)
        except Exception as e:
            print(f"Whisper batch inference failed for {wav_path.name}: {e}")
            continue

        # Re-align chunk indices: segment_idx was already advanced, so back-compute
        base_idx = segment_idx - len(chunk_meta)
        for k, (result, (chunk_name, duration)) in enumerate(zip(results, chunk_meta)):
            raw_text = result["text"]
            transcript = clean_transcript(raw_text)
            if not transcript:
                continue
            lid_tokens = classify_tokens(transcript)
            idx_str = f"pseudo_{base_idx + k:06d}"
            dataset_dict[idx_str] = {
                "wav": f"{{data_root}}/segmented_audio/{chunk_name}",
                "length": round(duration, 2),
                "words": transcript,
                "lid_tokens": lid_tokens
            }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset_dict, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ Created pseudo-labels for {segment_idx} VAD-segmented chunks.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    main()
