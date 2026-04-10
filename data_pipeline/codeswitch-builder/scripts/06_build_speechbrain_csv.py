#!/usr/bin/env python3
"""
Step 6 — Build SpeechBrain-Compatible CSV
==========================================
Generate train / valid / test CSV splits from synthesized audio.

SpeechBrain format:
  ID,wav,duration,transcript

Input:
  data/processed/cs_text.txt
  outputs/audio/cs_00001.wav …

Output:
  outputs/train.csv   (80 %)
  outputs/valid.csv   (10 %)
  outputs/test.csv    (10 %)

Rules:
  • No language tags in transcript
  • Plain lowercase text only
  • Duration computed via torchaudio
"""

import argparse
import csv
import random
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
    # Remove any accidental language tags like <en>, <vi>, [EN], [VI]
    import re

    text = re.sub(r"</?(?:en|vi|lang)>", "", text)
    text = re.sub(r"\[/?(?:en|vi|lang)\]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def build_csv(
    cs_text_path: Path = CS_TEXT_PATH,
    audio_dir: Path = AUDIO_DIR,
    output_dir: Path = OUTPUT_DIR,
    train_ratio: float = 0.80,
    valid_ratio: float = 0.10,
    seed: int = 42,
):
    """Build train/valid/test CSV files for SpeechBrain."""
    # Load transcripts
    with open(cs_text_path, encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]

    # Match audio files
    entries: list[dict] = []

    for idx, sent in enumerate(sentences, start=1):
        wav_name = f"cs_{idx:05d}.wav"
        wav_path = audio_dir / wav_name

        if not wav_path.exists():
            continue

        duration = get_duration(wav_path)
        if duration is None:
            continue

        transcript = clean_transcript(sent)
        if not transcript:
            continue

        # Use relative path from repo root for portability
        rel_wav = wav_path.relative_to(REPO_ROOT).as_posix()

        entries.append(
            {
                "ID": f"cs_{idx:05d}",
                "wav": rel_wav,
                "duration": duration,
                "transcript": transcript,
            }
        )

    if not entries:
        print("ERROR: No valid (audio + transcript) pairs found.")
        return

    print(f"Found {len(entries)} valid entries.")

    # --- Shuffle & split -----------------------------------------------------
    random.seed(seed)
    random.shuffle(entries)

    n = len(entries)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_set = entries[:n_train]
    valid_set = entries[n_train : n_train + n_valid]
    test_set = entries[n_train + n_valid :]

    # --- Write CSVs ----------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, subset in [
        ("train", train_set),
        ("valid", valid_set),
        ("test", test_set),
    ]:
        csv_path = output_dir / f"{name}.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["ID", "wav", "duration", "transcript"]
            )
            writer.writeheader()
            writer.writerows(subset)
        print(f"  {name}.csv : {len(subset)} entries → {csv_path}")

    print(
        f"\n  Split: train={len(train_set)}, valid={len(valid_set)}, test={len(test_set)}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Step 6: Build SpeechBrain CSV files")
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

    build_csv(
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
    print("\n✅  Step 6 complete — SpeechBrain CSV files ready in outputs/")


if __name__ == "__main__":
    main()
