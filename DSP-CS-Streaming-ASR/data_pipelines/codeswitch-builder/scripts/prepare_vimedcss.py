#!/usr/bin/env python3
"""
ViMedCSS Dataset Preparation
=============================
This script downloads 'tensorxt/ViMedCSS', extracts the audio to the local disk,
and formats it into SpeechBrain's required JSON format.

It also acts as the Pseudo-Labeler for the LID head. It uses the text and a pre-trained
Whisper model to generate word-level timestamp alignments to find where the English
words are spoken.
"""

import os
import json
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import torch
import torchaudio
import torchaudio.functional as F_audio
import soundfile as sf
import re

# Optional: whisper_timestamped provides forced-alignment word timestamps.
# Install with: pip install whisper-timestamped
try:
    import whisper_timestamped as whisper
    HAS_WHISPER_TS = True
except ImportError:
    HAS_WHISPER_TS = False
    whisper = None
# ---------------------------------------------------------------------
# Vietnamese character detection (from codeswitch-builder)
# ---------------------------------------------------------------------
VI_DIACRITICS = (
    "àáạảãâầấậẩẫăằắặẳẵ" "èéẹẻẽêềếệểễ" "ìíịỉĩ" "òóọỏõôồốộổỗơờớợởỡ" "ùúụủũưừứựửữ" "ỳýỵỷỹđ"
)
VI_REGEX = re.compile(f"[{VI_DIACRITICS}]")

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

def classify_word(raw_word: str) -> str:
    """Classify a single word into VI or EN."""
    token = clean_token(raw_word)
    if not token:
        return "VI"
    if is_vietnamese(token):
        return "VI"
    elif is_probable_english(token):
        return "EN"
    else:
        return "VI"

def clean_text(text: str) -> str:
    """Normalize a transcript for consistent tokenization and LID labeling."""
    text = text.lower().strip()
    # Remove punctuation but keep letters and Vietnamese diacritics
    text = re.sub(r"[!\"#\$%&\'\(\)\*\+,\./:;<=>\?@\[\\\]\^_`{\|}~\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def prepare_split(dataset, split_name, output_dir, use_pseudo_labels=False, whisper_model_size="tiny"):
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = split_dir / "wavs"
    audio_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / f"{split_name}.json"
    
    if use_pseudo_labels and HAS_WHISPER_TS:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model (size={whisper_model_size}) on {device} for Force Alignment...")
        model = whisper.load_model(whisper_model_size, device=device)
    
    sb_dict = {}
    
    print(f"Processing {split_name} split...")
    for i, item in enumerate(tqdm(dataset)):
        # HF datasets Audio structure
        audio_arr = item['audio']['array']
        sr = item['audio']['sampling_rate']
        raw_text = item['sentence'] if 'sentence' in item else item['text']
        # Normalize text: lowercase, remove punctuation (consistent with ASR tokenizer)
        text = clean_text(raw_text)
        
        utt_id = f"vimedcss_{split_name}_{i:05d}"
        wav_path = audio_dir / f"{utt_id}.wav"
        
        # Resample to 16kHz (using torchaudio, avoids librosa heavy dependency)
        if sr != 16000:
            waveform = torch.from_numpy(audio_arr).float().unsqueeze(0)
            waveform = F_audio.resample(waveform, orig_freq=sr, new_freq=16000)
            audio_arr = waveform.squeeze(0).numpy()
            sr = 16000
            
        sf.write(str(wav_path), audio_arr, sr)
        duration = len(audio_arr) / sr
        
        # Base metadata
        sb_dict[utt_id] = {
            "wav": str(wav_path.resolve()),
            "length": duration,
            "words": text
        }
        
        # --- PSEUDO LABELING GENERATION ---
        # Only needed for training split (Warmup Phase 1)
        if use_pseudo_labels and HAS_WHISPER_TS:
            try:
                # Force align the known text with the audio
                result = whisper.transcribe(model, str(wav_path), language="vi", initial_prompt=text)
                
                # Extract word-level timestamps
                lid_sequence = []
                if "segments" in result:
                    for segment in result["segments"]:
                        if "words" in segment:
                            for word_info in segment["words"]:
                                w = word_info["text"]
                                # Assign language label based on project heuristic
                                lang = classify_word(w)
                                lid_sequence.append(lang)
                
                if not lid_sequence:
                    # Fallback if alignment fails
                    lid_sequence = ["VI"] * len(text.split())
                    
                sb_dict[utt_id]["lid_tokens"] = " ".join(lid_sequence)
                
            except Exception as e:
                print(f"Failed alignment for {utt_id}: {e}")
                sb_dict[utt_id]["lid_tokens"] = " ".join(["VI"] * len(text.split())) # default fallback

    # Write SpeechBrain JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(sb_dict, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Saved {len(sb_dict)} utterances to {json_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data/vimedcss")
    parser.add_argument("--pseudo_label", action="store_true", help="Generate word-level LID labels using Whisper")
    parser.add_argument("--model_size", type=str, default="tiny", choices=["tiny", "base", "small"],
                        help="Whisper model size to use for forced-alignment (tiny=fastest, small=most accurate)")
    args = parser.parse_args()
    
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading tensorxt/ViMedCSS from Hugging Face...")
    try:
        ds = load_dataset("tensorxt/ViMedCSS")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Process all available splits
    for split in ds.keys():
        print(f"Found split: {split} (Size: {len(ds[split])})")
        # Generate pseudo labels only for training if requested
        do_pseudo = args.pseudo_label and split == "train"
        prepare_split(ds[split], split, out_dir, use_pseudo_labels=do_pseudo, whisper_model_size=args.model_size)

if __name__ == "__main__":
    main()
