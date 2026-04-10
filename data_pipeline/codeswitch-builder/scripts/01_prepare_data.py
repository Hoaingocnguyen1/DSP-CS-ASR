#!/usr/bin/env python3
"""
Step 1 — Prepare Mono Text & Audio Data (Production Ready)
==================================================

Trích xuất Text và Audio gốc từ HuggingFace trong 1 lượt tải.

Lệnh test nhanh (overwrite):
  python scripts/01_prepare_data.py --max-samples 100 --overwrite
Lệnh nối thêm data (append):
  python scripts/01_prepare_data.py --max-samples 5000 --append
"""

import argparse
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

try:
    import soundfile as sf
    _sf_ok = True
except ImportError:
    _sf_ok = False
    print("WARNING: soundfile package not installed. Cannot extract audio.")

# ---------------------------------------------------------------------
# Paths & Configs
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_VI_DIR = REPO_ROOT / "data" / "raw_vi"
RAW_EN_DIR = REPO_ROOT / "data" / "raw_en"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

# ---------------------------------------------------------------------
# Sub-routines
# ---------------------------------------------------------------------
_MULTI_SPACE = re.compile(r"\s+")
_NON_PRINTABLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_HAS_LETTER = re.compile(r"[a-zA-ZÀ-ỹ]")


def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = _NON_PRINTABLE.sub("", text)
    return _MULTI_SPACE.sub(" ", text)


def valid_sentence(text: str, min_tokens: int, max_tokens: int) -> bool:
    if not text or not _HAS_LETTER.search(text):
        return False
    return min_tokens <= len(text.split()) <= max_tokens


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------
def process_language(
    lang: str,
    raw_dir: Path,
    hf_name: str,
    hf_split: str,
    hf_streaming: bool,
    output_file: Path,
    stats_file: Path,
    max_samples: int,
    min_tokens: int,
    max_tokens: int,
    config_name: str = None,
    overwrite: bool = False,
    append: bool = False,
):
    print(f"\n=== Processing {lang.upper()} ===")

    if not _sf_ok:
        print(f"[{lang}] soundfile package missing. Please return and install it.")
        return

    # ---------------------------------------------------------
    # 1. Xử lý Overwrite/Append File cũ
    # ---------------------------------------------------------
    ann_file = output_file.parent / f"{lang}_raw.json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    annotation = {}
    old_stats = {"num_files": 0, "filtered_out": 0, "hf_rows_consumed": 0}

    if overwrite:
        print(f"[{lang}] Overwrite mode: Cleaning old files in {raw_dir.name}...")
        if ann_file.exists():
            ann_file.unlink()
        if output_file.exists():
            output_file.unlink()
        if stats_file.exists():
            stats_file.unlink()
        for f in raw_dir.glob("*.wav"):
            f.unlink()
    elif append:
        if ann_file.exists():
            print(f"[{lang}] Append mode: Loading existing annotations...")
            with open(ann_file, "r", encoding="utf-8") as f:
                annotation = json.load(f)
        if stats_file.exists():
            with open(stats_file, "r", encoding="utf-8") as f:
                old_stats = json.load(f)

    # ---------------------------------------------------------
    # 2. Khởi tạo biến chống trùng lặp (Deduplication & Safety)
    # ---------------------------------------------------------
    valid_lines = [v["words"] for v in annotation.values()]
    filtered_out_session = 0

    # Chống trùng lặp File Name (Lấy ID to nhất hiện có)
    # File name có dạng: vi_audio_000001.wav
    max_id = 0
    if annotation:
        for fname in annotation.keys():
            # Trích xuất số ID từ tên file. VD: 'vi_audio_000042.wav' -> 42
            match = re.search(r"_audio_(\d+)\.wav$", fname)
            if match:
                max_id = max(max_id, int(match.group(1)))
    global_idx = max_id + 1  # Bắt đầu đánh số từ ID cao nhất + 1 để tuyệt đối không đè

    # ---------------------------------------------------------
    # 3. Load Dataset
    # ---------------------------------------------------------
    print(f"[{lang}] Loading HuggingFace dataset: {hf_name}")
    load_kwargs = {
        "split": hf_split,
        "streaming": hf_streaming,
        "trust_remote_code": True,
    }
    if config_name:
        load_kwargs["name"] = config_name

    try:
        ds = load_dataset(hf_name, **load_kwargs)
    except Exception as e:
        print(f"[{lang}] ERROR loading dataset: {e}")
        return

    # ---------------------------------------------------------
    # 4. Vòng lặp chính xử lý data
    # ---------------------------------------------------------
    # Đặt target là (số cũ + số mới). Nếu append thêm 5000, target là old + 5000.
    target_samples = len(valid_lines) + max_samples
    hf_rows_consumed = old_stats.get("hf_rows_consumed", 0)

    pbar = tqdm(
        total=target_samples, initial=len(valid_lines), desc=f"[{lang}] Extracting"
    )

    for idx, item in enumerate(ds):
        # Bỏ qua các dòng đã tải ở những lần chạy trước (để khỏi tải lại file audio trùng)
        if idx < old_stats.get("hf_rows_consumed", 0):
            continue

        hf_rows_consumed = idx + 1

        # Trích xuất Text
        raw_text = (
            item.get("transcription") or item.get("text") or item.get("sentence") or ""
        )
        cleaned = clean_text(str(raw_text))

        # Lọc độ dài
        if not valid_sentence(cleaned, min_tokens, max_tokens):
            filtered_out_session += 1
            continue

        # Xử lý Audio (Đổi tên file theo Index để chống ghi đè)
        if "audio" in item and item["audio"]:
            audio_data = item["audio"]
            audio_array = audio_data.get("array")
            sr = audio_data.get("sampling_rate", 16000)

            if audio_array is not None:
                # Đặt tên an toàn tuyệt đối, kế thừa từ ID max
                fname = f"{lang}_audio_{global_idx:06d}.wav"
                out_path = raw_dir / fname
                
                # TÍNH TOÁN DURATION NGAY TẠI ĐÂY MÀ KHÔNG CẦN ĐỌC LẠI FILE
                duration = round(len(audio_array) / sr, 2) if sr else 0.0

                try:
                    sf.write(str(out_path), audio_array, sr, format="WAV")

                    # Update Dictionary an toàn
                    annotation[fname] = {
                        "wav": str(out_path.resolve()),
                        "words": cleaned,
                        "duration": duration,
                    }
                    valid_lines.append(cleaned)
                    global_idx += 1
                    pbar.update(1)

                    # Lưu định kỳ mỗi 500 file (Sống sót khi rớt mạng HF)
                    if global_idx % 500 == 0:
                        with open(ann_file, "w", encoding="utf-8") as f:
                            json.dump(annotation, f, indent=2, ensure_ascii=False)

                except Exception as e:
                    print(f"\n[{lang}] Error writing {fname}: {e}")

        # Dừng nếu đã đủ target
        if len(valid_lines) >= target_samples:
            break

    pbar.close()

    # ---------------------------------------------------------
    # 5. Lưu kết quả cuối cùng
    # ---------------------------------------------------------
    if annotation:
        with open(ann_file, "w", encoding="utf-8") as f:
            json.dump(annotation, f, indent=2, ensure_ascii=False)
        print(
            f"[{lang}] Saved final annotation ({len(annotation)} files) → {ann_file.name}"
        )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(valid_lines) + "\n")

    # Cập nhật cộng dồn thống kê
    new_filtered_total = old_stats.get("filtered_out", 0) + filtered_out_session
    durations = [v.get("duration", 0.0) for v in annotation.values()]
    
    total_duration_sec = round(sum(durations), 2)
    num_audio_files = len(durations)
    avg_duration = round(total_duration_sec / num_audio_files, 2) if num_audio_files else 0
    min_duration = round(min(durations), 2) if durations else 0
    max_duration = round(max(durations), 2) if durations else 0

    stats = {
        "num_files": len(annotation),
        "filtered_out": new_filtered_total,
        "new_files_added_this_session": len(valid_lines) - old_stats.get("num_files", 0),
        "hf_rows_consumed": hf_rows_consumed,
        "total_audio_duration_sec": total_duration_sec,
        "num_audio_files": num_audio_files,
        "avg_audio_duration_sec": avg_duration,
        "min_audio_duration_sec": min_duration,
        "max_audio_duration_sec": max_duration
    }
    
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[{lang}] Current Total Samples: {stats['num_files']}")
    print(f"[{lang}] Filtered out (Cumulative): {stats['filtered_out']}")


def main():
    parser = argparse.ArgumentParser(description="Prepare Mono Text + Audio")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=50_000,
        help="Số file MUỐN LẤY THÊM trong lần chạy này",
    )
    parser.add_argument("--min-tokens", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=40)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--overwrite", action="store_true", help="Xóa sạch kéo lại từ đầu"
    )
    group.add_argument(
        "--append", action="store_true", help="Nối thêm data mới đằng sau data cũ"
    )

    args = parser.parse_args()
    
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    process_language(
        lang="vi",
        raw_dir=RAW_VI_DIR,
        hf_name="doof-ferb/vlsp2020_vinai_100h",
        hf_split="train",
        hf_streaming=True,  # Đổi thành True cho nhẹ RAM
        output_file=PROCESSED_DIR / "vi_text.txt",
        stats_file=PROCESSED_DIR / "vi_stats.json",
        max_samples=args.max_samples,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overwrite=args.overwrite,
        append=args.append,
    )

    process_language(
        lang="en",
        raw_dir=RAW_EN_DIR,
        hf_name="MLCommons/peoples_speech",
        hf_split="train",
        hf_streaming=True,
        config_name="clean",
        output_file=PROCESSED_DIR / "en_text.txt",
        stats_file=PROCESSED_DIR / "en_stats.json",
        max_samples=args.max_samples,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        overwrite=args.overwrite,
        append=args.append,
    )


if __name__ == "__main__":
    main()
