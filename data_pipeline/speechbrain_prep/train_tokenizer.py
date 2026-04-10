#!/usr/bin/env python3
"""
SentencePiece Tokenizer Training Script
=========================================
Trains a SentencePiece BPE tokenizer on the ViMedCSS transcripts extracted
from train.json. The resulting model is shared across all CTC baselines.

Usage (from repo root DSP-CS-ASR/):
    python data_pipeline/speechbrain_prep/train_tokenizer.py \
        --train_json  data/vimedcss/train.json \
        --output_dir  recipes/DSP_CodeSwitch/Tokenizer/save \
        --vocab_size  4000 \
        --model_type  bpe

Output files:
    <output_dir>/tokenizer_<vocab_size>_<model_type>.model
    <output_dir>/tokenizer_<vocab_size>_<model_type>.vocab
"""

import argparse
import json
import os
import random
import sys
import tempfile
from pathlib import Path

import sentencepiece as spm


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def extract_texts_from_json(json_path: str) -> list[str]:
    """Load all 'words' values from a SpeechBrain JSON manifest."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    for utt_id, meta in data.items():
        text = meta.get("words", "").strip()
        if text:
            texts.append(text)

    print(f"  ✔ Loaded {len(texts):,} transcripts from {json_path}")
    return texts


def compute_tokenizer_stats(sp: spm.SentencePieceProcessor,
                            texts: list[str]) -> dict:
    """Compute useful statistics about the trained tokenizer on given texts."""
    all_lengths = []
    unk_count = 0
    total_tokens = 0
    unk_id = sp.piece_to_id("<unk>")

    for text in texts:
        ids = sp.encode_as_ids(text)
        n = len(ids)
        all_lengths.append(n)
        total_tokens += n
        unk_count += sum(1 for i in ids if i == unk_id)

    avg_len = sum(all_lengths) / len(all_lengths) if all_lengths else 0
    max_len = max(all_lengths) if all_lengths else 0
    min_len = min(all_lengths) if all_lengths else 0
    unk_rate = (unk_count / total_tokens * 100) if total_tokens else 0

    return {
        "num_sentences": len(texts),
        "total_tokens": total_tokens,
        "avg_tokens_per_sentence": round(avg_len, 2),
        "min_tokens": min_len,
        "max_tokens": max_len,
        "unk_count": unk_count,
        "unk_rate_pct": round(unk_rate, 4),
    }


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train a SentencePiece tokenizer for ViMedCSS ASR."
    )
    parser.add_argument("--train_json", type=str, required=True,
                        help="Path to train.json (SpeechBrain format)")
    parser.add_argument("--valid_json", type=str, default=None,
                        help="Optional: path to valid.json — used for sanity-check sampling")
    parser.add_argument("--output_dir", type=str,
                        default="recipes/DSP_CodeSwitch/Tokenizer/save",
                        help="Directory to save the tokenizer model")
    parser.add_argument("--vocab_size", type=int, default=4000,
                        help="Vocabulary size (default: 4000)")
    parser.add_argument("--model_type", type=str, default="bpe",
                        choices=["bpe", "unigram", "char", "word"],
                        help="SentencePiece model type")
    parser.add_argument("--character_coverage", type=float, default=1.0,
                        help="Character coverage (1.0 = full coverage, good for Vietnamese)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_sanity_samples", type=int, default=8,
                        help="Number of sentences to show in sanity check")
    args = parser.parse_args()

    # ── Reproducibility ──
    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Collect transcripts ---
    print("=" * 60)
    print("  STEP 1 / 4 — Reading training transcripts")
    print("=" * 60)
    texts = extract_texts_from_json(args.train_json)

    if not texts:
        print("❌ ERROR: No transcripts found. Check your train.json path and 'words' field.")
        sys.exit(1)

    # --- 2. Write to a temp file for SentencePiece trainer ---
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8",
                                     delete=False) as tmp:
        tmp.write("\n".join(texts))
        tmp_path = tmp.name

    model_prefix = str(out_dir / f"tokenizer_{args.vocab_size}_{args.model_type}")

    # --- 3. Train SentencePiece ---
    print()
    print("=" * 60)
    print("  STEP 2 / 4 — Training SentencePiece tokenizer")
    print("=" * 60)
    print(f"  vocab_size         = {args.vocab_size}")
    print(f"  model_type         = {args.model_type}")
    print(f"  character_coverage = {args.character_coverage}")
    print(f"  seed               = {args.seed}")
    print(f"  num_transcripts    = {len(texts):,}")
    print()

    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
        # Reserve special tokens consistent with SpeechBrain defaults
        pad_id=0,          # <blank> for CTC
        bos_id=1,          # <bos>
        eos_id=2,          # <eos>
        unk_id=3,          # <unk>
        # Normalization & encoding options
        normalization_rule_name="nmt_nfkc_cf",   # NFKC + case-folding
        byte_fallback=True,                       # Handle medical abbreviations (e.g. mL, mg)
        add_dummy_prefix=True,
        # Reproducibility
        shuffle_input_sentence=True,
    )

    os.unlink(tmp_path)  # cleanup temp

    # --- 4. Load & Verify ---
    print()
    print("=" * 60)
    print("  STEP 3 / 4 — Tokenization Statistics")
    print("=" * 60)

    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")

    stats = compute_tokenizer_stats(sp, texts)
    print(f"  Vocab size (actual) = {sp.get_piece_size()}")
    print(f"  Training sentences  = {stats['num_sentences']:,}")
    print(f"  Total tokens        = {stats['total_tokens']:,}")
    print(f"  Avg tokens / sent   = {stats['avg_tokens_per_sentence']}")
    print(f"  Min tokens / sent   = {stats['min_tokens']}")
    print(f"  Max tokens / sent   = {stats['max_tokens']}")
    print(f"  <unk> count         = {stats['unk_count']}")
    print(f"  <unk> rate          = {stats['unk_rate_pct']}%")

    if stats['unk_rate_pct'] > 0:
        print("  ⚠️  WARNING: <unk> rate > 0%. Consider increasing vocab_size "
              "or checking byte_fallback.")
    else:
        print("  ✔ No <unk> tokens — byte_fallback is working correctly.")

    # --- 5. Sanity Check ---
    print()
    print("=" * 60)
    print("  STEP 4 / 4 — Sanity Check")
    print("=" * 60)

    # Use validation set if provided, otherwise sample from training
    sanity_source = "train"
    sanity_texts = texts
    if args.valid_json and Path(args.valid_json).exists():
        sanity_texts = extract_texts_from_json(args.valid_json)
        sanity_source = "valid"

    # Always include these domain-specific hardcoded examples
    hardcoded_examples = [
        "bệnh nhân bị đau ngực",
        "kết quả mri cho thấy khối u",
        "dùng paracetamol 500mg",
        "chụp ct scan phổi",
    ]

    # Sample random sentences from data
    n_sample = min(args.num_sanity_samples, len(sanity_texts))
    sampled = random.sample(sanity_texts, n_sample)

    all_examples = hardcoded_examples + sampled
    print(f"  Showing {len(hardcoded_examples)} hardcoded + "
          f"{n_sample} random samples (from {sanity_source} set):\n")

    for i, sent in enumerate(all_examples, 1):
        tokens = sp.encode_as_pieces(sent)
        ids    = sp.encode_as_ids(sent)
        print(f"  [{i:02d}] Input  : {sent}")
        print(f"       Pieces : {tokens}")
        print(f"       IDs    : {ids}")
        print(f"       #tokens: {len(ids)}")
        print()

    # --- Summary ---
    print("=" * 60)
    print("  ✅ DONE — Tokenizer trained successfully!")
    print("=" * 60)
    print(f"  Model : {model_prefix}.model")
    print(f"  Vocab : {model_prefix}.vocab")
    print(f"  Size  : {sp.get_piece_size()} subword units")
    print()


if __name__ == "__main__":
    main()
