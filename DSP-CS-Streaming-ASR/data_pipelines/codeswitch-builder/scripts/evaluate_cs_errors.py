#!/usr/bin/env python3
"""
Code-Switched ASR Error Analysis Tool
======================================
This script parses the predictions and targets from SpeechBrain and computes 
detailed metrics for the thesis (Table 5 & Ablation Study).

It calculates:
1. Overall WER
2. cs-WER (Code-Switched Word Error Rate) - WER specifically on English words.
3. vi-WER - WER specifically on Vietnamese words.

Usage:
  python scripts/evaluate_cs_errors.py --ref references.txt --hyp hypotheses.txt
"""

import argparse
import re
import json
from pathlib import Path
from typing import List, Dict
import jiwer

# ---------------------------------------------------------------------
# Language Classification Rules
# ---------------------------------------------------------------------
VI_DIACRITICS = "àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
VI_REGEX = re.compile(f"[{VI_DIACRITICS}]")

def is_vietnamese(token: str) -> bool:
    return bool(VI_REGEX.search(token))

def is_probable_english(token: str) -> bool:
    if not token.isascii():
        return False
    if len(token) <= 2:
        return False  # avoid misclassifying short VI words
    return token.isalpha()

def classify_word(token: str) -> str:
    """Classify a single word into VI or EN."""
    token = token.lower()
    if is_vietnamese(token):
        return "VI"
    elif is_probable_english(token):
        return "EN"
    else:
        return "VI" # default fallback

def filter_words(sentence: str, target_lang: str) -> str:
    """Returns a sentence containing only words of the target language."""
    words = sentence.split()
    filtered = [w for w in words if classify_word(w) == target_lang]
    return " ".join(filtered)

def normalize_text(text: str) -> str:
    """Lowercase, strip, and collapse whitespace for consistent WER computation."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

# ---------------------------------------------------------------------
# Metrics Computation
# ---------------------------------------------------------------------
def compute_metrics(references: List[str], hypotheses: List[str]) -> Dict:
    # Normalize before computing (case-insensitive WER is standard)
    references = [normalize_text(r) for r in references]
    hypotheses = [normalize_text(h) for h in hypotheses]
    
    # Drop empty lines to avoid jiwer crashes
    valid_pairs = [(r, h) for r, h in zip(references, hypotheses) if r.strip()]
    if not valid_pairs:
        print("Warning: No valid reference/hypothesis pairs to evaluate.")
        return {"overall_wer": 0.0, "vi_wer": 0.0, "cs_wer": 0.0}
    references, hypotheses = zip(*valid_pairs)
    references, hypotheses = list(references), list(hypotheses)
    
    # 1. Overall WER
    overall_wer = jiwer.wer(references, hypotheses)
    
    # 2. English (Code-Switched) WER
    ref_en = [filter_words(r, "EN") for r in references]
    hyp_en = [filter_words(h, "EN") for h in hypotheses]
    
    # Filter out empty references
    valid_en_idx = [i for i, r in enumerate(ref_en) if len(r.strip()) > 0]
    ref_en_valid = [ref_en[i] for i in valid_en_idx]
    hyp_en_valid = [hyp_en[i] for i in valid_en_idx]
    
    cs_wer = jiwer.wer(ref_en_valid, hyp_en_valid) if ref_en_valid else 0.0

    # 3. Vietnamese WER
    ref_vi = [filter_words(r, "VI") for r in references]
    hyp_vi = [filter_words(h, "VI") for h in hypotheses]
    
    valid_vi_idx = [i for i, r in enumerate(ref_vi) if len(r.strip()) > 0]
    ref_vi_valid = [ref_vi[i] for i in valid_vi_idx]
    hyp_vi_valid = [hyp_vi[i] for i in valid_vi_idx]

    vi_wer = jiwer.wer(ref_vi_valid, hyp_vi_valid) if ref_vi_valid else 0.0

    print("=" * 50)
    print("Code-Switching ASR Evaluation Results")
    print("=" * 50)
    print(f"Total Utterances evaluated: {len(references)}")
    print(f"English (CS) Utterances   : {len(ref_en_valid)}")
    print(f"Vietnamese Utterances     : {len(ref_vi_valid)}")
    print("-" * 50)
    print(f"Overall WER        :  {overall_wer * 100:.2f} %")
    print(f"Vietnamese (vi-WER):  {vi_wer * 100:.2f} %")
    print(f"English (cs-WER)   :  {cs_wer * 100:.2f} %  <-- MAIN METRIC")
    print("=" * 50)
    
    return {"overall_wer": round(overall_wer * 100, 2), "vi_wer": round(vi_wer * 100, 2), "cs_wer": round(cs_wer * 100, 2)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", type=str, help="Path to reference text file (one sentence per line)")
    parser.add_argument("--hyp", type=str, help="Path to hypothesis text file (one sentence per line)")
    parser.add_argument("--json_out", type=str, default=None, help="Optional path to save results in JSON format")
    
    # Make arguments optional for dry-run testing
    args, unknown = parser.parse_known_args()

    if not args.ref or not args.hyp:
        print("Running Dry-Run test because --ref and --hyp were not provided...")
        ref = [
            "tôi bị đau đầu sau khi dùng paracetamol",
            "kết quả mri cho thấy khối u",
        ]
        hyp = [
            "tôi bị đau đầu sau khi trúng paracetamol",
            "kết quả m ra ai cho thấy khối u",
        ]
        compute_metrics(ref, hyp)
        return

    # Read files
    with open(args.ref, "r", encoding="utf-8") as f:
        references = [line.strip() for line in f.readlines()]
        
    with open(args.hyp, "r", encoding="utf-8") as f:
        hypotheses = [line.strip() for line in f.readlines()]

    if len(references) != len(hypotheses):
        print(f"Error: Number of references ({len(references)}) does not match number of hypotheses ({len(hypotheses)})")
        return

    results = compute_metrics(references, hypotheses)
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
