#!/usr/bin/env python3
"""
Recovery script: rebuild train.json from existing WAV files + HuggingFace dataset.
Re-runs fastText LID offline (fast), then applies saved LLM corrections from
review_corrections_train.jsonl without calling the API.

Usage:
    python recover_train.py --output_dir ../../data/vimedcss
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

# Import functions from prepare_vimedcss
sys.path.insert(0, str(Path(__file__).parent))
from prepare_vimedcss import (
    clean_text,
    parse_cs_terms,
    classify_with_confidence,
    postprocess_labels,
    heuristic_lid,
    tag_exceptions,
    load_fasttext_model,
    _fasttext_predict,
)


def load_review_corrections(review_log_path: Path) -> dict:
    """Load LLM review corrections as dict: utt_id → {idx: llm_label}."""
    corrections = {}
    if not review_log_path.exists():
        return corrections
    with open(review_log_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            utt_id = entry["utt_id"]
            diffs = entry.get("diffs", [])
            if utt_id not in corrections:
                corrections[utt_id] = {}
            for d in diffs:
                corrections[utt_id][d["idx"]] = d["llm_review"]
    return corrections


def main():
    parser = argparse.ArgumentParser(description="Recover train.json from existing WAV files")
    parser.add_argument("--output_dir", type=str, default="../../data/vimedcss")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--fasttext_model", type=str, default=None)
    parser.add_argument("--confidence", type=float, default=0.9)
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    split = args.split
    wav_dir = out_dir / "wavs" / split
    json_path = out_dir / f"{split}.json"
    review_log = out_dir / f"review_corrections_{split}.jsonl"

    # Check WAV files exist
    wav_files = sorted(wav_dir.glob("*.wav")) if wav_dir.exists() else []
    print(f"📁 Found {len(wav_files)} WAV files in {wav_dir}")
    if not wav_files:
        print("❌ No WAV files found. Nothing to recover.")
        sys.exit(1)

    # Load fastText model
    ft_model_path = args.fasttext_model or str(out_dir.parent / "fasttext" / "lid.176.bin")
    print(f"🔤 Loading fastText model: {ft_model_path}")
    ft_model = load_fasttext_model(ft_model_path)
    print("   ✅ Loaded.")

    # Load saved LLM review corrections
    llm_corrections = load_review_corrections(review_log)
    print(f"📋 Loaded {len(llm_corrections)} LLM review corrections from {review_log.name}")

    # Stream dataset to get text + cs_terms metadata
    print("⬇️  Streaming dataset to get text metadata...")
    from datasets import load_dataset
    ds = load_dataset("tensorxt/ViMedCSS", streaming=True)
    split_data = ds[split]

    # Build utt_id → metadata mapping from dataset
    utt_meta = {}
    existing_utt_ids = {f.stem for f in wav_files}  # e.g. "vimedcss_train_Med_CS-0-11"

    for i, item in enumerate(tqdm(split_data, desc=f"Scanning {split}")):
        seg_id = item.get("segment_id") or f"{i:05d}"
        utt_id = f"vimedcss_{split}_{seg_id}"

        if utt_id not in existing_utt_ids:
            continue

        raw_text = item.get("segment_text") or item.get("sentence") or item.get("text") or ""
        text = clean_text(raw_text)
        cs_terms_raw = item.get("cs_terms") or item.get("cs_terms_list") or ""
        cs_terms = parse_cs_terms(cs_terms_raw)

        utt_meta[utt_id] = {"text": text, "cs_terms": cs_terms}

        # Stop early once we have all we need
        if len(utt_meta) >= len(existing_utt_ids):
            break

    print(f"✅ Matched {len(utt_meta)} / {len(existing_utt_ids)} WAV files to dataset entries")

    # Build train.json
    sb_dict = {}
    exception_log = out_dir / f"exceptions_{split}_recovered.jsonl"

    for wav_path in tqdm(wav_files, desc="Rebuilding JSON"):
        utt_id = wav_path.stem

        if utt_id not in utt_meta:
            # WAV exists but no text found → skip
            continue

        meta = utt_meta[utt_id]
        text = meta["text"]
        cs_terms = meta["cs_terms"]

        if not text.strip():
            continue

        # Get WAV duration
        info = sf.info(str(wav_path))
        duration = round(info.duration, 3)

        tokens = text.split()

        # Stage 1+2: fastText + confidence
        ft_labels, confidences, uncertain_indices = classify_with_confidence(
            tokens, ft_model, threshold=args.confidence
        )

        # Stage 2.5: heuristic baseline
        heur_labels = heuristic_lid(text)

        # Stage 3: Apply saved LLM corrections (no API call!)
        labels_after_llm = list(ft_labels)
        if utt_id in llm_corrections:
            for idx_str, llm_label in llm_corrections[utt_id].items():
                idx = int(idx_str) if isinstance(idx_str, str) else idx_str
                if idx < len(labels_after_llm):
                    labels_after_llm[idx] = llm_label

        # Stage 4: Rule correction
        corrected, corrections = postprocess_labels(tokens, labels_after_llm, cs_terms)

        # Store all intermediate labels
        sb_dict[utt_id] = {
            "wav": str(wav_path.resolve()),
            "length": duration,
            "words": text,
            "lid_heuristic": " ".join(heur_labels),
            "lid_fasttext": " ".join(ft_labels),
            "lid_fasttext_conf": " ".join(f"{c:.3f}" for c in confidences),
            "lid_fasttext_llm": " ".join(labels_after_llm),
            "lid_tokens": " ".join(corrected),
        }

    # Save JSON
    print(f"\n💾 Saving {len(sb_dict)} entries → {json_path}")
    tmp_path = json_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(sb_dict, f, ensure_ascii=False, indent=2)
    tmp_path.replace(json_path)

    # Stats
    total_tokens = sum(len(v["lid_tokens"].split()) for v in sb_dict.values())
    en_tokens = sum(v["lid_tokens"].split().count("EN") for v in sb_dict.values())
    vi_tokens = total_tokens - en_tokens
    print(f"✅ Done! {len(sb_dict)} utterances, {total_tokens} tokens")
    print(f"   EN: {en_tokens} ({100*en_tokens/total_tokens:.1f}%)")
    print(f"   VI: {vi_tokens} ({100*vi_tokens/total_tokens:.1f}%)")
    print(f"   LLM corrections applied: {len(llm_corrections)}")
    print(f"\n🔄 To continue processing remaining entries, run:")
    print(f"   python prepare_vimedcss.py --output_dir {args.output_dir} --use_fasttext --use_llm --resume")


if __name__ == "__main__":
    main()
