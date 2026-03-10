#!/usr/bin/env python3
"""
SentencePiece Tokenizer Training Script
=========================================
Trains a SentencePiece BPE tokenizer on the ViMedCSS transcripts extracted
from train.json. The resulting model is shared across all CTC baselines.

Usage:
    python scripts/train_tokenizer.py \
        --train_json  data/vimedcss/train.json \
        --output_dir  speechbrain/recipes/DSP_CodeSwitch/Tokenizer/save \
        --vocab_size  4000 \
        --model_type  bpe

Output files:
    <output_dir>/tokenizer_<vocab_size>_<model_type>.model
    <output_dir>/tokenizer_<vocab_size>_<model_type>.vocab
"""

import argparse
import json
import os
import tempfile
from pathlib import Path

import sentencepiece as spm


def extract_texts_from_json(json_path: str) -> list[str]:
    """Load all 'words' values from a SpeechBrain JSON manifest."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    for utt_id, meta in data.items():
        text = meta.get("words", "").strip()
        if text:
            texts.append(text)

    print(f"  Loaded {len(texts)} transcripts from {json_path}")
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, required=True,
                        help="Path to train.json (SpeechBrain format)")
    parser.add_argument("--output_dir", type=str,
                        default="speechbrain/recipes/DSP_CodeSwitch/Tokenizer/save",
                        help="Directory to save the tokenizer model")
    parser.add_argument("--vocab_size", type=int, default=4000,
                        help="Vocabulary size (default: 4000)")
    parser.add_argument("--model_type", type=str, default="bpe",
                        choices=["bpe", "unigram", "char", "word"],
                        help="SentencePiece model type")
    parser.add_argument("--character_coverage", type=float, default=1.0,
                        help="Character coverage (1.0 = full coverage, good for Vietnamese)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Collect transcripts ---
    print("Reading training transcripts...")
    texts = extract_texts_from_json(args.train_json)

    if not texts:
        print("ERROR: No transcripts found. Check your train.json path and 'words' field.")
        return

    # --- 2. Write to a temp file for SentencePiece trainer ---
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8",
                                     delete=False) as tmp:
        tmp.write("\n".join(texts))
        tmp_path = tmp.name

    model_prefix = str(out_dir / f"tokenizer_{args.vocab_size}_{args.model_type}")

    # --- 3. Train SentencePiece ---
    print(f"\nTraining SentencePiece tokenizer...")
    print(f"  vocab_size         = {args.vocab_size}")
    print(f"  model_type         = {args.model_type}")
    print(f"  character_coverage = {args.character_coverage}")

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
        # Keep all whitespace: Vietnamese multi-syllabic words need proper segmentation
        normalization_rule_name="nmt_nfkc_cf",   # NFKC + case-folding
        byte_fallback=True,                       # Handle medical abbreviations (e.g. mL, mg)
        add_dummy_prefix=True,
    )

    os.unlink(tmp_path)  # cleanup temp

    # --- 4. Verify ---
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_prefix}.model")

    test_sentences = [
        "bệnh nhân bị đau ngực",
        "kết quả mri cho thấy khối u",
        "dùng paracetamol 500mg",
        "chụp ct scan phổi",
    ]

    print("\n✅ Tokenizer trained successfully!")
    print(f"   Model: {model_prefix}.model")
    print(f"   Vocab: {model_prefix}.vocab")
    print(f"\n--- Tokenization Sanity Check ---")
    for sent in test_sentences:
        tokens = sp.encode_as_pieces(sent)
        ids    = sp.encode_as_ids(sent)
        print(f"  Input : {sent}")
        print(f"  Pieces: {tokens}")
        print(f"  IDs   : {ids}")
        print()


if __name__ == "__main__":
    main()
