#!/usr/bin/env python3
"""
Step 5 v2 — Code-Switch Dataset Statistics (Academic Stable)

Analyse:
  • English token ratio
  • Sentence length distribution
  • Switch-point frequency
  • Switch density
  • Lexical diversity (type-token ratio)
  • Percentiles (P25, P50, P75, P95)
  • Automatic distribution warnings

Input:
  data/processed/cs_text.txt

Output:
  Printed summary + optional cs_stats.json
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CS_TEXT_PATH = PROCESSED_DIR / "cs_text.txt"
STATS_OUT = PROCESSED_DIR / "cs_stats.json"

# ---------------------------------------------------------------------
# Vietnamese character detection
# ---------------------------------------------------------------------
VI_DIACRITICS = (
    "àáạảãâầấậẩẫăằắặẳẵ" "èéẹẻẽêềếệểễ" "ìíịỉĩ" "òóọỏõôồốộổỗơờớợởỡ" "ùúụủũưừứựửữ" "ỳýỵỷỹđ"
)

VI_REGEX = re.compile(f"[{VI_DIACRITICS}]")

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------


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
    labels = []
    tokens = []

    for raw in sentence.split():
        token = clean_token(raw)
        if not token:
            continue

        tokens.append(token)

        if is_vietnamese(token):
            labels.append("VI")
        elif is_probable_english(token):
            labels.append("EN")
        else:
            labels.append("VI")

    return tokens, labels


def count_switches(labels):
    if len(labels) < 2:
        return 0
    return sum(1 for a, b in zip(labels, labels[1:]) if a != b)


def percentile(data, p):
    if not data:
        return 0
    data = sorted(data)
    k = int(len(data) * p / 100)
    return data[min(k, len(data) - 1)]


# ---------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------


def compute_stats(cs_path: Path, save_json=True):

    with open(cs_path, encoding="utf-8") as f:
        sentences = [l.strip() for l in f if l.strip()]

    if not sentences:
        print("No sentences found.")
        return

    en_ratios = []
    lengths = []
    switch_counts = []
    switch_density = []
    switch_dist = Counter()
    vocab = set()
    total_tokens = 0

    for sent in sentences:

        tokens, labels = classify_tokens(sent)
        n_tokens = len(tokens)

        if n_tokens == 0:
            continue

        total_tokens += n_tokens
        vocab.update(tokens)

        ratio = sum(1 for l in labels if l == "EN") / n_tokens
        switches = count_switches(labels)

        en_ratios.append(ratio)
        lengths.append(n_tokens)
        switch_counts.append(switches)
        switch_density.append(switches / n_tokens)
        switch_dist[switches] += 1

    # --------------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------------
    stats = {
        "total_sentences": len(lengths),
        "total_tokens": total_tokens,
        "vocab_size": len(vocab),
        "type_token_ratio": round(len(vocab) / total_tokens, 4),
        "english_ratio": {
            "mean": round(sum(en_ratios) / len(en_ratios), 4),
            "p25": percentile(en_ratios, 25),
            "p50": percentile(en_ratios, 50),
            "p75": percentile(en_ratios, 75),
            "p95": percentile(en_ratios, 95),
        },
        "sentence_length": {
            "mean": round(sum(lengths) / len(lengths), 2),
            "p50": percentile(lengths, 50),
            "p95": percentile(lengths, 95),
        },
        "switch_points": {
            "mean": round(sum(switch_counts) / len(switch_counts), 2),
            "p95": percentile(switch_counts, 95),
            "distribution": dict(sorted(switch_dist.items())),
        },
        "switch_density_mean": round(sum(switch_density) / len(switch_density), 4),
    }

    # --------------------------------------------------------------
    # Print
    # --------------------------------------------------------------
    print("=" * 65)
    print("  Code-Switch Dataset Statistics (v2 Academic)")
    print("=" * 65)

    print(f"Sentences        : {stats['total_sentences']}")
    print(f"Total tokens     : {stats['total_tokens']}")
    print(f"Vocab size       : {stats['vocab_size']}")
    print(f"Type-Token Ratio : {stats['type_token_ratio']}")
    print()

    print("English Ratio:")
    for k, v in stats["english_ratio"].items():
        print(f"  {k:>4}: {v:.4f}")
    print()

    print("Sentence Length:")
    for k, v in stats["sentence_length"].items():
        print(f"  {k:>4}: {v}")
    print()

    print("Switch Points:")
    for k, v in stats["switch_points"].items():
        if k == "distribution":
            continue
        print(f"  {k:>4}: {v}")
    print()

    print("Switch Density Mean:", stats["switch_density_mean"])
    print()

    print("Switch Distribution:")
    for k, v in stats["switch_points"]["distribution"].items():
        pct = 100 * v / len(lengths)
        print(f"  {k} switches: {v:>6} ({pct:5.1f}%)")

    print("=" * 65)

    # --------------------------------------------------------------
    # Auto Warnings
    # --------------------------------------------------------------
    print("\nAuto Checks:")

    mean_en = stats["english_ratio"]["mean"]
    p95_len = stats["sentence_length"]["p95"]
    mean_switch = stats["switch_points"]["mean"]

    if not (0.20 <= mean_en <= 0.40):
        print("⚠ English ratio outside recommended range (0.20–0.40)")

    if p95_len > 25:
        print("⚠ Long-tail sentences too long (P95 > 25 tokens)")

    if mean_switch < 1 or mean_switch > 3:
        print("⚠ Switch frequency not in target range (1–3 per sentence)")

    print("\nDone.")

    # --------------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------------
    if save_json:
        STATS_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(STATS_OUT, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\nStats saved → {STATS_OUT}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-json", action="store_true")
    args = parser.parse_args()

    if not CS_TEXT_PATH.exists():
        print("ERROR: cs_text.txt not found.")
        return

    compute_stats(CS_TEXT_PATH, save_json=not args.no_json)


if __name__ == "__main__":
    main()
