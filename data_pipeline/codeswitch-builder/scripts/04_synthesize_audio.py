#!/usr/bin/env python3
"""
Step 4 — Clean Speech Synthesis
================================================

Reads the mixed code-switched text from Step 3 (cs_text_mixed.json)
and synthesizes audio using viXTTS.

Features:
- Reads rich JSON output (vi, en, cs, lid, level, en_ratio)
- Saves all metadata into the output manifest.jsonl for ASR training
- Automatic resume functionality (checks existing manifest IDs)
- 16kHz Mono conversion + amplitude normalization
"""

import argparse
import json
from pathlib import Path

import torch
import torchaudio
from tqdm import tqdm

# -----------------------------------------------------
# Paths
# -----------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
AUDIO_DIR = REPO_ROOT / "outputs" / "audio"
MANIFEST_PATH = REPO_ROOT / "outputs" / "manifest.jsonl"

# Đầu vào là file JSON từ bước mix (Step 3)
CS_MIXED_JSON = PROCESSED_DIR / "cs_text_mixed.json"

TARGET_SR = 16_000
MAX_DURATION_SEC = 15.0


# -----------------------------------------------------
# TTS Wrapper
# -----------------------------------------------------

class VieNeuRemoteSynthesizer:
    def __init__(self, api_base="http://127.0.0.1:23333/v1", model_name="pnnbao-ump/VieNeu-TTS"):
        from vieneu import Vieneu

        print(f"Loading VieNeu Remote on {api_base} ...")
        self.tts = Vieneu(
            mode='remote', 
            api_base=api_base, 
            model_name=model_name
        )
        self.resampler = None

        # Fetch available voice presets from the remote server
        self.available_voices = self.tts.list_preset_voices()
        if self.available_voices:
            # Using the first active voice by default
            _, self.voice_id = self.available_voices[0]
            self.voice_data = self.tts.get_preset_voice(self.voice_id)
            print(f"🔊 Selected voice: {self.voice_id}")
        else:
            self.voice_data = None
            print(f"🔊 Selected voice: Default")

    def synthesize(self, text: str, output_path: Path):
        try:
            # Synthesize audio with the Remote API
            audio_spec = self.tts.infer(text=text, voice=self.voice_data)
            
            # Save raw returned audio back to file
            self.tts.save(audio_spec, str(output_path))

            # Load to apply Torchaudio transformations (Mono + 16kHz + Normalize)
            waveform, sr = torchaudio.load(str(output_path))

            # Convert stereo to mono if needed
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample to 16kHz
            if sr != TARGET_SR:
                if self.resampler is None:
                    self.resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=TARGET_SR
                    )
                waveform = self.resampler(waveform)

            # Normalize amplitude (chống clipping)
            waveform = waveform / (waveform.abs().max() + 1e-6)

            torchaudio.save(str(output_path), waveform, TARGET_SR)

            duration = waveform.shape[-1] / TARGET_SR
            return duration

        except Exception as e:
            print(f"[WARN] Failed: {text[:40]} ... {e}")
            return None


# -----------------------------------------------------
# Main
# -----------------------------------------------------

def run(args):
    if not CS_MIXED_JSON.exists():
        raise FileNotFoundError(f"{CS_MIXED_JSON.name} not found. Run Step 3 (mix) first.")

    with open(CS_MIXED_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.max_samples:
        # Giới hạn số lượng chạy
        data = {k: v for i, (k, v) in enumerate(data.items()) if i < args.max_samples}

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Đọc manifest cũ để Resume
    processed_ids = set()
    if MANIFEST_PATH.exists() and not args.overwrite:
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    processed_ids.add(entry.get("id"))
                except json.JSONDecodeError:
                    pass
        print(f"Resuming... {len(processed_ids)} already synthesized.")
    elif args.overwrite and MANIFEST_PATH.exists():
        MANIFEST_PATH.unlink()
        print("Overwrite flag set. Starting fresh.")

    # Lọc ra các ID cần chạy (chưa tổng hợp)
    pending_items = [(k, v) for k, v in data.items() if k not in processed_ids]

    if not pending_items:
        print("Everything is up to date!")
        return

    print(f"Pending: {len(pending_items)} sentences to synthesize")

    tts = VieNeuRemoteSynthesizer(api_base=args.api_base, model_name=args.api_model_name)
    skipped = 0
    durations = []

    with open(MANIFEST_PATH, "a" if not args.overwrite else "w", encoding="utf-8") as manifest:
        pbar = tqdm(pending_items, desc="TTS Synthesis")
        for sentence_id, metadata in pbar:
            # Câu CS thực tế cần sinh ra âm thanh
            text_to_synthesize = metadata["cs"]

            filename = f"{sentence_id}.wav"
            out_path = AUDIO_DIR / filename

            duration = tts.synthesize(text_to_synthesize, out_path)

            if duration is None:
                skipped += 1
                continue

            if duration > MAX_DURATION_SEC:
                out_path.unlink(missing_ok=True)
                skipped += 1
                print(f"\n[WARN] Skipped {sentence_id}: exceeded {MAX_DURATION_SEC}s ({duration:.1f}s)")
                continue

            durations.append(duration)

            # Viết JSON Lines (Lưu ĐẦY ĐỦ metadata cho model train)
            entry = {
                "id": sentence_id,
                "audio": str(out_path.resolve()),
                "duration": round(duration, 3),
                "text": text_to_synthesize,        # Text đã mix
                "text_vi": metadata.get("vi"),     # Text gốc
                "text_en": metadata.get("en"),     # Text Anh gốc
                "lid_targets": metadata.get("lid"),# Label 0/1 cho LID head
                "level": metadata.get("level"),
                "en_ratio": metadata.get("en_ratio"),
                "en_spans": metadata.get("en_spans"),
            }

            manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
            manifest.flush()  # Save liền để resume an toàn

    print("\n--- Synthesis Complete ---")
    print(f"Successfully generated: {len(durations)}")
    print(f"Skipped (errors/too long): {skipped}")

    if durations:
        print(f"Average duration: {sum(durations)/len(durations):.2f}s")
        print(f"Total audio hours: {sum(durations)/3600:.2f} hrs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=None, help="Stop after N samples")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing manifest and start fresh")
    parser.add_argument("--api-base", type=str, default="http://127.0.0.1:23333/v1", help="VieNeu Remote API Base URL")
    parser.add_argument("--api-model-name", type=str, default="pnnbao-ump/VieNeu-TTS", help="Model Name on Remote server")
    args = parser.parse_args()

    run(args)
