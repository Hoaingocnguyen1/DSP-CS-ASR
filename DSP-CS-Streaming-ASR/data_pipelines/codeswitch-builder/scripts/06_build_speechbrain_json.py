#!/usr/bin/env python3
"""
Step 6 — Build SpeechBrain-Compatible JSON
==========================================
Generate train / valid / test JSON splits from synthesized audio,
specifically tailored for the Causal Signal-Driven Code-Switching ASR thesis.

SpeechBrain format required:
{
  "cs_00001": {
    "wav": "{data_root}/outputs/audio/cs_00001.wav",
    "length": 3.52,
    "words": "tôi cần join meeting lúc 3 giờ",
    "lid_tokens": "VI VI EN EN VI VI VI"
  }
}

Rules:
  • No language tags in 'words' (clean ASR supervision)
  • 'lid_tokens' contains token-level Language ID for Prompt Generator warming
  • Duration computed via torchaudio
"""

import argparse
import json
import random
import re
from pathlib import Path

import torchaudio

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
AUDIO_DIR = REPO_ROOT / "outputs" / "audio"
OUTPUT_DIR = REPO_ROOT / "outputs"

CS_TEXT_PATH = PROCESSED_DIR / "cs_text.txt"

# ---------------------------------------------------------------------
# Vietnamese character detection (from 05_compute_stats.py)
# ---------------------------------------------------------------------
VI_DIACRITICS = (
    "àáạảãâầấậẩẫăằắặẳẵ" "èéẹẻẽêềếệểễ" "ìíịỉĩ" "òóọỏõôồốộổỗơờớợởỡ" "ùúụủũưừứựửữ" "ỳýỵỷỹđ"
)
VI_REGEX = re.compile(f"[{VI_DIACRITICS}]")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_duration(wav_path: Path) -> float | None:
    """Return duration in seconds using torchaudio, or None on error."""
    try:
        info = torchaudio.info(str(wav_path))
        return round(info.num_frames / info.sample_rate, 2)
    except Exception as e:
        print(f"  [WARN] Cannot read {wav_path.name}: {e}")
        return None

def clean_transcript(text: str) -> str:
    """Ensure transcript is plain lowercase, no language tags."""
    text = text.lower().strip()
    text = re.sub(r"</?(?:en|vi|lang)>", "", text)
    text = re.sub(r"\[/?(?:en|vi|lang)\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_token(token: str) -> str:
    """Remove punctuation but keep Vietnamese chars."""
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
    
    # We need to map labels 1:1 with the cleaned tokens of the sentence
    cleaned_sent = clean_transcript(sentence)
    for raw in cleaned_sent.split():
        token = clean_token(raw)
        if not token: # In case of punctuation only, fallback to VI or ignore
            labels.append("VI") 
            continue

        if is_vietnamese(token):
            labels.append("VI")
        elif is_probable_english(token):
            labels.append("EN")
        else:
            labels.append("VI")

    return " ".join(labels)

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_json(
    cs_text_path: Path = CS_TEXT_PATH,
    audio_dir: Path = AUDIO_DIR,
    output_dir: Path = OUTPUT_DIR,
    train_ratio: float = 0.80,
    valid_ratio: float = 0.10,
    seed: int = 42,
):
    """Build train/valid/test JSON files for SpeechBrain."""
    # Load transcripts
    with open(cs_text_path, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    # Match audio files
    dataset_dict = {}

    for idx, sent in enumerate(sentences, start=1):
        idx_str = f"cs_{idx:05d}"
        wav_name = f"{idx_str}.wav"
        wav_path = audio_dir / wav_name

        if not wav_path.exists():
            continue

        duration = get_duration(wav_path)
        if duration is None:
            continue

        transcript = clean_transcript(sent)
        if not transcript:
            continue
            
        lid_tokens = classify_tokens(sent)

        # Use relative path with {data_root} macro for SpeechBrain portability
        rel_wav = wav_path.relative_to(REPO_ROOT).as_posix()
        sb_wav_path = f"{{data_root}}/{rel_wav}"

        dataset_dict[idx_str] = {
            "wav": sb_wav_path,
            "length": duration,
            "words": transcript,
            "lid_tokens": lid_tokens
        }

    if not dataset_dict:
        print("ERROR: No valid (audio + transcript) pairs found.")
        return

    print(f"Found {len(dataset_dict)} valid entries.")

    # --- Shuffle & split -----------------------------------------------------
    keys = list(dataset_dict.keys())
    random.seed(seed)
    random.shuffle(keys)

    n = len(keys)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_keys = keys[:n_train]
    valid_keys = keys[n_train : n_train + n_valid]
    test_keys = keys[n_train + n_valid :]

    # --- Write JSONs ---------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        "train": train_keys,
        "valid": valid_keys,
        "test": test_keys
    }

    for name, subset_keys in splits.items():
        json_path = output_dir / f"{name}.json"
        
        subset_dict = {k: dataset_dict[k] for k in subset_keys}
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(subset_dict, f, indent=2, ensure_ascii=False)
            
        print(f"  {name}.json : {len(subset_keys)} entries → {json_path}")

    print(
        f"\n  Split: train={len(train_keys)}, valid={len(valid_keys)}, test={len(test_keys)}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Step 6: Build SpeechBrain JSON files")
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--valid-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not CS_TEXT_PATH.exists():
        print(f"ERROR: {CS_TEXT_PATH} not found. Run 03_generate_cs_text.py first.")
        return

    if not AUDIO_DIR.exists() or not any(AUDIO_DIR.glob("*.wav")):
        print(f"ERROR: No WAV files in {AUDIO_DIR}. Run 04_synthesize_audio.py first.")
        return

    build_json(
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
    print("\n✅  Step 6 complete — SpeechBrain JSON files ready in outputs/")

if __name__ == "__main__":
    main()
