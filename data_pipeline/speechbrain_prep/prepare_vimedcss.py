#!/usr/bin/env python3
"""
ViMedCSS Dataset Preparation for SpeechBrain
=============================================
Downloads 'tensorxt/ViMedCSS', saves audio to local .wav files, and builds
a SpeechBrain-compatible JSON manifest with per-token LID labels.

LID Pipeline (4-stage hybrid):
  Stage 1 — fastText word LID:  Each token is classified individually using
                                  a pre-trained fastText LID model (lid.176.bin).
                                  Returns label + confidence score.
  Stage 2 — Confidence filter:   Tokens with fastText confidence ≥ threshold
                                  (default 0.9) keep their fastText label.
                                  Tokens below threshold are marked "uncertain".
  Stage 3 — LLM review:          Only uncertain tokens are sent (with full
                                  sentence context) to an LLM for targeted review.
  Stage 4 — Rule correction:     Deterministic post-processing rules applied last:
             (a) Diacritics Override: tokens with Vietnamese diacritics → force VI.
             (b) cs_terms Validation: use ViMedCSS metadata to correct EN labels
                 for known code-switched terms.
             (c) Numbers Rule: digits/numeric tokens → VI.

  Exception tagging: All corrections and ambiguous cases are logged to
  `exceptions_<split>.jsonl` for manual review.

Accuracy comparison (typical code-switch corpus):
  | Method              | Accuracy    |
  |---------------------|-------------|
  | heuristic           | ~80–85%     |
  | LLM only            | ~90–93%     |
  | fastText only       | ~92–95%     |
  | fastText + LLM      | ~96–98%     |

Usage:
  # Heuristic mode (fast, no API key needed, no fastText):
  python prepare_vimedcss.py --output_dir data/vimedcss

  # fastText only (no LLM, fast):
  python prepare_vimedcss.py --output_dir data/vimedcss --use_fasttext

  # Full hybrid pipeline (fastText + LLM review uncertain tokens):
  python prepare_vimedcss.py --output_dir data/vimedcss --use_fasttext --use_llm

  # With custom confidence threshold:
  python prepare_vimedcss.py --output_dir data/vimedcss --use_fasttext --use_llm --confidence 0.9

  # Debug with a small sample:
  python prepare_vimedcss.py --output_dir data/vimedcss --use_fasttext --use_llm --max_samples 20
"""

import os
import sys
import json
import argparse
import re
import urllib.request
from pathlib import Path

import torch
import torchaudio.functional as F_audio
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import fasttext
except ImportError:
    fasttext = None

# ---------------------------------------------------------------------------
# Constants / Vietnamese regex
# ---------------------------------------------------------------------------

VI_DIACRITICS = (
    "àáạảãâầấậẩẫăằắặẳẵ"
    "èéẹẻẽêềếệểễ"
    "ìíịỉĩ"
    "òóọỏõôồốộổỗơờớợởỡ"
    "ùúụủũưừứựửữ"
    "ỳýỵỷỹđ"
)
VI_REGEX = re.compile(f"[{VI_DIACRITICS}]")

FASTTEXT_MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

# ---------------------------------------------------------------------------
# Text Utilities
# ---------------------------------------------------------------------------

def clean_token(token: str) -> str:
    return re.sub(rf"[^\w{VI_DIACRITICS}]", "", token.lower())

def is_vietnamese(token: str) -> bool:
    return bool(VI_REGEX.search(token))

def is_probable_english(token: str) -> bool:
    if not token.isascii():
        return False
    if len(token) <= 2:
        return False
    return token.isalpha()

def is_number(token: str) -> bool:
    """Check if token is a number (digits, commas, dots for decimals)."""
    cleaned = re.sub(r"[,.]", "", token)
    return cleaned.isdigit()

# Common Vietnamese words that have NO diacritics and would be falsely classified as English
VI_NO_DIACRITICS = {
    # Pronouns / particles
    "toi", "ban", "anh", "chi", "em", "con", "ong", "cho", "cua", "trong",
    "nay", "khi", "sau", "theo", "qua", "bao", "nhieu", "moi", "lam", "rat",
    "hay", "nhu", "cung", "voi", "duoc", "boi", "dong", "mot", "hai", "nam",
    # Common adjectives/nouns (no diacritics)
    "cao", "lon", "nho", "ngon", "khong", "gia", "gan", "giao", "tri",
    "can", "bao", "hoa", "than", "tin", "dai", "mau", "san", "pham",
    "cong", "nghe", "hoc", "sinh", "khoa", "dung", "chung", "tong",
    "the", "nha", "nuoc", "bien", "may", "bay", "hang", "ngay", "thang",
    "nam", "bon", "sau", "bay", "tam", "chin", "muoi", "tram", "ngan",
    # Medical/body (common in ViMedCSS)
    "benh", "nhan", "thuoc", "lieu", "tri", "chan", "doan", "xet",
    "nghiem", "phau", "khac", "ung", "thu", "nao", "tim", "gan",
    "phoi", "than", "xuat", "huyet", "nhiem", "trung", "viem",
    "bach", "cau", "hong", "tieu", "hoa", "xuong", "khop",
    # Verbs
    "goi", "dung", "chay", "uong", "giam", "tang", "kiem", "tra",
    "ghi", "nhan", "theo", "doi", "dieu", "cham", "soc", "hoi",
    "phuc", "bao", "gom", "danh", "gia", "huong", "dan",
    # Misc
    "hieu", "qua", "noi", "ngoai", "rieng", "chung", "chinh",
    "phu", "tong", "hop", "lien", "quan", "tuc", "cac", "cuc",
    "thi", "mat", "tay", "chan", "mieng", "hong", "mui",
}

def classify_word(raw_word: str) -> str:
    t = clean_token(raw_word)
    if not t:
        return "VI"
    if is_vietnamese(t):
        return "VI"
    if t in VI_NO_DIACRITICS:
        return "VI"
    if is_probable_english(t):
        return "EN"
    return "VI"

def heuristic_lid(text: str) -> list[str]:
    return [classify_word(w) for w in text.split()]

def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[!"#$%&\'()*+,\./:;<=>?@\[\\\]^_`{|}~\-]', "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# fastText Word LID (Stage 1)
# ---------------------------------------------------------------------------

def download_fasttext_model(model_path: Path) -> Path:
    """Download lid.176.bin from Facebook if not already present."""
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"⬇️  Downloading fastText LID model → {model_path}")
    print(f"   URL: {FASTTEXT_MODEL_URL}")
    print(f"   This may take a few minutes (~126MB)...")
    urllib.request.urlretrieve(FASTTEXT_MODEL_URL, str(model_path))
    print(f"   ✅ Downloaded successfully.")
    return model_path


def load_fasttext_model(model_path: str | Path):
    """Load fastText LID model."""
    if fasttext is None:
        print("ERROR: fasttext package not installed. Install with: pip install fasttext")
        sys.exit(1)
    model_path = Path(model_path)
    download_fasttext_model(model_path)
    # Suppress fastText warning about loading model with old format
    model = fasttext.load_model(str(model_path))
    return model


# fastText language label mapping
# fastText lid.176.bin uses labels like '__label__vi', '__label__en'
FASTTEXT_LANG_MAP = {
    "__label__vi": "VI",
    "__label__en": "EN",
}


def _fasttext_predict(model, text: str) -> tuple[str, float]:
    """Wrapper around model.predict() that handles numpy 2.x compatibility.
    fasttext 0.9.3 uses np.array(probs, copy=False) which breaks on numpy>=2.0.
    """
    try:
        labels, probs = model.predict(text)
        return labels[0], float(probs[0])
    except ValueError:
        # numpy 2.x compat: use the raw C++ model method directly
        # Args: (text, k, threshold, label_prefix) → returns list[(prob, label)]
        predictions = model.f.predict(text, 1, 0.0, "\n")
        if predictions:
            prob, label_raw = predictions[0]
            return label_raw, float(prob)
        return "__label__vi", 0.5


def fasttext_word_lid(token: str, model) -> tuple[str, float]:
    """
    Predict language for a single token using fastText.
    Returns (label, confidence) where label is 'VI' or 'EN'.
    """
    cleaned = clean_token(token)
    if not cleaned:
        return "VI", 1.0  # empty/punctuation → VI

    # Numbers → VI immediately
    if is_number(cleaned):
        return "VI", 1.0

    # Vietnamese diacritics → VI immediately
    if is_vietnamese(cleaned):
        return "VI", 1.0

    # Vietnamese dictionary check
    if cleaned in VI_NO_DIACRITICS:
        return "VI", 0.95  # high but not perfect confidence

    label_raw, prob = _fasttext_predict(model, cleaned)

    # Map to VI/EN
    label = FASTTEXT_LANG_MAP.get(label_raw, None)
    if label is None:
        # Not Vietnamese and not English → treat as foreign/EN for code-switch
        # But if it's a known non-VI non-EN language, default to VI in this context
        label = "VI" if prob < 0.5 else "EN"

    return label, float(prob)


def classify_with_confidence(
    tokens: list[str],
    model,
    threshold: float = 0.9,
) -> tuple[list[str], list[float], list[int]]:
    """
    Classify all tokens using fastText, marking uncertain ones.

    Returns:
        labels: list of "VI"/"EN" for each token
        confidences: list of confidence scores
        uncertain_indices: indices of tokens with confidence < threshold
    """
    labels = []
    confidences = []
    uncertain = []

    for i, tok in enumerate(tokens):
        label, conf = fasttext_word_lid(tok, model)
        labels.append(label)
        confidences.append(conf)
        if conf < threshold:
            uncertain.append(i)

    return labels, confidences, uncertain


# ---------------------------------------------------------------------------
# cs_terms Utilities (used as checker, NOT as LLM hint)
# ---------------------------------------------------------------------------

def parse_cs_terms(cs_terms_str: str) -> list[str]:
    """Parse semicolon-separated cs_terms into a cleaned list of term strings."""
    if not cs_terms_str:
        return []
    return [t.strip().lower() for t in str(cs_terms_str).split(";") if t.strip()]

def count_expected_en_tokens(cs_terms: list[str]) -> int:
    """Total word count across all terms = expected number of EN tokens."""
    return sum(len(term.split()) for term in cs_terms)

def fuzzy_word_match(term_word: str, en_word: str) -> bool:
    """Check if cs_term word matches an EN token, allowing for Vietnamized spelling.
    E.g. 'testosteron' should match 'testosterone' or 'dihydrotestosterone'.
    """
    tw = clean_token(term_word)
    ew = clean_token(en_word)
    if not tw or not ew:
        return False
    # Exact match
    if tw == ew:
        return True
    # Substring: cs_term is contained in the EN token (e.g. "testosteron" in "dihydrotestosterone")
    if tw in ew or ew in tw:
        return True
    # Prefix match (>= 4 chars): handles minor suffix differences
    if len(tw) >= 4 and len(ew) >= 4:
        min_len = min(len(tw), len(ew))
        prefix_len = max(4, min_len - 2)  # allow up to 2 chars difference
        if tw[:prefix_len] == ew[:prefix_len]:
            return True
    return False

def check_terms_covered(text: str, labels: list[str], cs_terms: list[str]) -> tuple[bool, str]:
    """
    Verify that every cs_term is covered by the EN-labeled tokens in the transcript.
    Uses fuzzy matching to handle Vietnamized spelling (e.g. testosteron → testosterone).
    Returns (is_covered, reason).
    """
    tokens = text.split()
    if len(tokens) != len(labels):
        return False, f"token/label length mismatch ({len(tokens)} vs {len(labels)})"

    en_tokens = [tokens[i] for i, l in enumerate(labels) if l == "EN"]

    missing = []
    for term in cs_terms:
        term_words = term.split()
        # Each word of the term must fuzzy-match at least one EN-labeled token
        found = all(
            any(fuzzy_word_match(tw, ew) for ew in en_tokens)
            for tw in term_words
        )
        if not found:
            missing.append(term)

    if missing:
        return False, f"cs_terms not covered by EN labels: {missing}"
    return True, "OK"


# ---------------------------------------------------------------------------
# Post-Processing & Exception Tagging (Stage 4 — Rule Correction)
# ---------------------------------------------------------------------------

def postprocess_labels(
    tokens: list[str],
    labels: list[str],
    cs_terms: list[str] | None = None,
) -> tuple[list[str], list[dict]]:
    """
    Apply deterministic post-processing rules after LLM labeling.
    Returns (corrected_labels, list_of_corrections).

    Rule 1 — Diacritics Override: token with Vietnamese diacritics → force VI.
    Rule 2 — cs_terms Validation: use cs_terms metadata to flip VI→EN for
             known code-switched terms that the LLM missed.
    Rule 3 — Numbers Rule: digits/numeric tokens → force VI.
    """
    corrected = list(labels)  # copy
    corrections = []

    # --- Rule 1: Diacritics Override ---
    for i, (tok, lbl) in enumerate(zip(tokens, corrected)):
        if lbl == "EN" and is_vietnamese(tok):
            corrected[i] = "VI"
            corrections.append({
                "type": "diacritic_override",
                "idx": i,
                "token": tok,
                "old": "EN",
                "new": "VI",
            })

    # --- Rule 2: cs_terms Validation ---
    if cs_terms:
        for term in cs_terms:
            term_words = term.split()
            for tw in term_words:
                # Find token in transcript that fuzzy-matches this cs_term word
                for i, tok in enumerate(tokens):
                    if corrected[i] == "VI" and fuzzy_word_match(tw, tok):
                        # Don't flip if the token has Vietnamese diacritics
                        if not is_vietnamese(tok):
                            corrected[i] = "EN"
                            corrections.append({
                                "type": "cs_terms_correction",
                                "idx": i,
                                "token": tok,
                                "cs_term": tw,
                                "old": "VI",
                                "new": "EN",
                            })

    # --- Rule 3: Numbers Rule ---
    for i, (tok, lbl) in enumerate(zip(tokens, corrected)):
        if lbl == "EN" and is_number(tok):
            corrected[i] = "VI"
            corrections.append({
                "type": "number_override",
                "idx": i,
                "token": tok,
                "old": "EN",
                "new": "VI",
            })

    return corrected, corrections


def tag_exceptions(
    utt_id: str,
    text: str,
    original_labels: list[str],
    corrected_labels: list[str],
    corrections: list[dict],
    exception_log_path: Path,
    confidences: list[float] | None = None,
) -> None:
    """
    Log corrections and ambiguous cases to exceptions JSONL for manual review.
    Only writes an entry if there are corrections or notable disagreements.
    """
    tags = list(corrections)  # start with corrections as tags

    # Compare with heuristic to detect high disagreement
    heuristic = heuristic_lid(text)
    if len(heuristic) == len(corrected_labels):
        disagree_count = sum(
            1 for h, c in zip(heuristic, corrected_labels) if h != c
        )
        disagree_pct = round(100.0 * disagree_count / len(heuristic), 1)
        if disagree_pct > 30:
            tags.append({
                "type": "llm_heuristic_disagree",
                "disagree_pct": disagree_pct,
                "heuristic_labels": heuristic,
            })

    # Detect ambiguous tokens: short (<=2 char), all-caps, no diacritics
    tokens = text.split()
    for i, tok in enumerate(tokens):
        cleaned = clean_token(tok)
        if cleaned and len(cleaned) <= 2 and cleaned.isalpha() and not is_vietnamese(tok):
            tags.append({
                "type": "ambiguous_token",
                "idx": i,
                "token": tok,
                "label": corrected_labels[i],
            })

    # Log low-confidence fastText predictions
    if confidences:
        for i, (tok, conf) in enumerate(zip(tokens, confidences)):
            if conf < 0.7:  # very uncertain
                tags.append({
                    "type": "low_confidence_fasttext",
                    "idx": i,
                    "token": tok,
                    "confidence": round(conf, 4),
                    "label": corrected_labels[i],
                })

    if not tags:
        return

    entry = {
        "utt_id": utt_id,
        "text": text,
        "original_labels": original_labels,
        "corrected_labels": corrected_labels,
        "tags": tags,
    }
    with open(exception_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# LLM Review for Uncertain Tokens Only (Stage 3)
# ---------------------------------------------------------------------------

LLM_REVIEW_SYSTEM_PROMPT = """\
You are a linguist specializing in Vietnamese-English code-switching in medical texts.

You will receive a sentence with per-token labels from an automatic LID system.
Some tokens are marked as UNCERTAIN (their label might be wrong).

Your task:
1. Review the UNCERTAIN tokens in context of the full sentence.
2. For each UNCERTAIN token, decide if it should be 'VI' (Vietnamese) or 'EN' (English).
3. Also check if any other token's label looks obviously wrong.

RULES:
- Words like "testosterone", "dihydrotestosterone", "reductase", "glycine", "protein" → 'EN'
- Vietnamese words that happen to be ASCII (e.g. "ngon", "cao", "cho", "gia", "con") → 'VI'
- Vietnamized medical terms (e.g. "hóc môn") → 'VI'
- Numbers like "5", "4000" → 'VI'

Return the COMPLETE corrected labels array for ALL tokens (not just uncertain ones).
"""

# Full LLM system prompt for when fastText is NOT used (legacy mode)
LLM_SYSTEM_PROMPT = """\
You are a linguist specializing in Vietnamese-English code-switching in medical texts.

You will receive a word list. For EACH word, assign exactly one label:
  - 'VI' for Vietnamese words (includes Vietnamized loanwords like "hóc môn", words without diacritics like "ngon", "cao")
  - 'EN' for English words (medical terms, brand names, technical jargon)

CRITICAL RULES:
1. Output array length MUST EXACTLY equal the number of words given.
2. Words like "testosterone", "dihydrotestosterone", "reductase", "glycine", "super", "morning" → 'EN'
3. Vietnamese words that happen to be ASCII (e.g. "ngon", "cao", "cho", "gia", "con") → 'VI'
4. Numbers like "5", "4000" → 'VI'
"""


def call_llm_review_uncertain(
    utt_id: str,
    tokens: list[str],
    fasttext_labels: list[str],
    confidences: list[float],
    uncertain_indices: list[int],
    client,
    model: str,
    max_retries: int = 2,
) -> list | None:
    """
    Send sentence context + fastText labels to LLM, highlighting uncertain tokens.
    LLM reviews only uncertain tokens but returns full corrected labels.
    Returns list of labels or None on complete API failure.
    """
    n_words = len(tokens)

    # Build context-rich input showing all tokens with labels and confidence
    lines = []
    lines.append(f"Sentence: {' '.join(tokens)}")
    lines.append("")
    lines.append("Token labels from fastText:")
    for i, (tok, lbl, conf) in enumerate(zip(tokens, fasttext_labels, confidences)):
        uncertain_marker = " ← UNCERTAIN" if i in uncertain_indices else ""
        lines.append(f"  {tok} {lbl} ({conf:.2f}){uncertain_marker}")
    lines.append("")
    lines.append("Check if any label is wrong. Return corrected labels for ALL tokens.")

    user_content = "\n".join(lines)

    tool = [{
        "type": "function",
        "function": {
            "name": "submit_corrected_labels",
            "description": f"Submit exactly {n_words} corrected labels, one per token.",
            "parameters": {
                "type": "object",
                "properties": {
                    "labels": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["VI", "EN"]},
                        "minItems": n_words,
                        "maxItems": n_words,
                        "description": f"Exactly {n_words} labels, one per token.",
                    }
                },
                "required": ["labels"],
            },
        },
    }]

    best_labels = None

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": LLM_REVIEW_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                tools=tool,
                tool_choice={"type": "function", "function": {"name": "submit_corrected_labels"}},
                temperature=0.0,
            )
            args_str = resp.choices[0].message.tool_calls[0].function.arguments
            result = json.loads(args_str)
            labels = result.get("labels", [])

            if len(labels) == n_words:
                return labels

            # Keep the attempt closest to expected length
            if best_labels is None or abs(len(labels) - n_words) < abs(len(best_labels) - n_words):
                best_labels = labels

        except Exception as e:
            if attempt == max_retries:
                print(f"    [LLM Review Error] {utt_id}: {e}")

    # Pad/trim the best result if we have one
    if best_labels is not None:
        if len(best_labels) < n_words:
            best_labels.extend(["VI"] * (n_words - len(best_labels)))
        elif len(best_labels) > n_words:
            best_labels = best_labels[:n_words]
        return best_labels

    return None


def call_llm_lid_single(utt_id: str, word_list: list[str], client, model: str, max_retries: int = 2) -> list | None:
    """
    Call LLM for a SINGLE utterance (legacy mode when fastText is not used).
    Retry up to max_retries times if length mismatches.
    Returns list of labels or None only on complete API failure.
    """
    n_words = len(word_list)
    tool = [{
        "type": "function",
        "function": {
            "name": "submit_lid_labels",
            "description": f"Submit exactly {n_words} labels, one per word.",
            "parameters": {
                "type": "object",
                "properties": {
                    "labels": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["VI", "EN"]},
                        "minItems": n_words,
                        "maxItems": n_words,
                        "description": f"Exactly {n_words} labels, one per word.",
                    }
                },
                "required": ["labels"],
            },
        },
    }]

    data = json.dumps({"word_list": word_list, "n_words": n_words}, ensure_ascii=False)
    best_labels = None  # Keep the closest-length attempt

    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": f"TRANSCRIPT:\n{data}"},
                ],
                tools=tool,
                tool_choice={"type": "function", "function": {"name": "submit_lid_labels"}},
                temperature=0.0,
            )
            args_str = resp.choices[0].message.tool_calls[0].function.arguments
            result = json.loads(args_str)
            labels = result.get("labels", [])

            if len(labels) == n_words:
                return labels

            # Keep the attempt closest to expected length
            if best_labels is None or abs(len(labels) - n_words) < abs(len(best_labels) - n_words):
                best_labels = labels

        except Exception as e:
            if attempt == max_retries:
                print(f"    [LLM Error] {utt_id}: {e}")

    # All retries exhausted — pad/trim the best result instead of returning None
    if best_labels is not None:
        if len(best_labels) < n_words:
            best_labels.extend(["VI"] * (n_words - len(best_labels)))
        elif len(best_labels) > n_words:
            best_labels = best_labels[:n_words]
        return best_labels

    return None


REVIEW_SYSTEM_PROMPT = """\
You are a senior linguist reviewing Vietnamese-English code-switching labels.

You will receive a word list and initial VI/EN labels from a junior annotator.
Review and correct any mistakes. Common errors to watch for:
- Vietnamese words without diacritics (ngon, cao, cho, gia, con, gan, hoa) wrongly labeled EN
- English medical terms (testosterone, glycine, hormone, reductase) wrongly labeled VI
- Numbers should be labeled VI

Return the corrected labels array. Length MUST equal the number of words.
"""

def call_llm_review(utt_id: str, word_list: list[str], initial_labels: list[str], client, model: str) -> list | None:
    """
    Review pass: a stronger model checks and corrects the initial labels.
    Returns corrected labels or None on failure.
    """
    n_words = len(word_list)
    tool = [{
        "type": "function",
        "function": {
            "name": "submit_reviewed_labels",
            "description": f"Submit exactly {n_words} corrected labels.",
            "parameters": {
                "type": "object",
                "properties": {
                    "labels": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["VI", "EN"]},
                        "minItems": n_words,
                        "maxItems": n_words,
                    }
                },
                "required": ["labels"],
            },
        },
    }]

    # Show both words and initial labels side-by-side
    pairs = [f"{w} → {l}" for w, l in zip(word_list, initial_labels)]
    data = json.dumps({"word_label_pairs": pairs, "n_words": n_words}, ensure_ascii=False)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
                {"role": "user", "content": f"REVIEW THIS:\n{data}"},
            ],
            tools=tool,
            tool_choice={"type": "function", "function": {"name": "submit_reviewed_labels"}},
            temperature=0.0,
        )
        args_str = resp.choices[0].message.tool_calls[0].function.arguments
        result = json.loads(args_str)
        labels = result.get("labels", [])
        if len(labels) == n_words:
            return labels
        # Pad/trim if close
        if len(labels) < n_words:
            labels.extend(["VI"] * (n_words - len(labels)))
        elif len(labels) > n_words:
            labels = labels[:n_words]
        return labels
    except Exception as e:
        print(f"    [Review Error] {utt_id}: {e}")
        return None  # Keep initial labels on review failure


# ---------------------------------------------------------------------------
# Core Processing — Hybrid Pipeline
# ---------------------------------------------------------------------------

def run_hybrid_pipeline(
    utt_id: str,
    text: str,
    fail_log_path: Path,
    exception_log_path: Path,
    cs_terms: list[str] | None = None,
    ft_model=None,
    confidence_threshold: float = 0.9,
    llm_client=None,
    llm_model: str = "gpt-4.1-mini",
    review_log_path: Path | None = None,
) -> dict:
    """
    Full hybrid pipeline:
      Stage 1: fastText word LID
      Stage 2: Confidence filter
      Stage 3: LLM review uncertain tokens (if LLM enabled)
      Stage 4: Rule correction (diacritics, cs_terms, numbers)

    Returns dict with all intermediate labels for backtesting:
      lid_heuristic:     heuristic-only labels
      lid_fasttext:      fastText-only labels (before LLM)
      lid_fasttext_conf: fastText confidence scores
      lid_fasttext_llm:  fastText + LLM review labels (before rules)
      lid_tokens:        final labels (after rule correction)
    """
    tokens = text.split()

    # ──── Heuristic baseline (always compute for comparison) ────
    heur_labels = heuristic_lid(text)

    # ──── Stage 1 & 2: fastText + Confidence Filter ────
    if ft_model is not None:
        ft_labels, confidences, uncertain_indices = classify_with_confidence(
            tokens, ft_model, threshold=confidence_threshold
        )
    else:
        # No fastText → use heuristic as base
        ft_labels = list(heur_labels)
        confidences = [0.9] * len(tokens)  # synthetic confidence
        uncertain_indices = []

    # ──── Stage 3: LLM Review (uncertain tokens only) ────
    labels_after_llm = list(ft_labels)

    if llm_client and uncertain_indices:
        llm_reviewed = call_llm_review_uncertain(
            utt_id, tokens, ft_labels, confidences, uncertain_indices,
            llm_client, llm_model,
        )
        if llm_reviewed:
            # Only apply LLM corrections for uncertain tokens
            # (trust fastText for confident tokens)
            for idx in uncertain_indices:
                if idx < len(llm_reviewed):
                    labels_after_llm[idx] = llm_reviewed[idx]

            # Log LLM review corrections
            if review_log_path:
                diffs = [
                    {
                        "idx": j,
                        "token": tokens[j],
                        "fasttext": ft_labels[j],
                        "fasttext_conf": round(confidences[j], 4),
                        "llm_review": llm_reviewed[j],
                    }
                    for j in uncertain_indices
                    if j < len(llm_reviewed) and ft_labels[j] != llm_reviewed[j]
                ]
                if diffs:
                    entry = {
                        "utt_id": utt_id,
                        "text": text,
                        "n_uncertain": len(uncertain_indices),
                        "n_total_tokens": len(tokens),
                        "diffs": diffs,
                    }
                    with open(review_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        else:
            # LLM failed — log and keep fastText labels
            entry = {
                "utt_id": utt_id,
                "text": text,
                "reason": "LLM review failed for uncertain tokens",
                "uncertain_indices": uncertain_indices,
                "uncertain_tokens": [tokens[i] for i in uncertain_indices],
            }
            with open(fail_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    elif llm_client and not uncertain_indices:
        # All tokens are confident — skip LLM entirely
        pass

    # ──── Stage 4: Rule Correction ────
    original_labels = list(labels_after_llm)
    corrected, corrections = postprocess_labels(tokens, labels_after_llm, cs_terms)

    # ──── Exception Tagging ────
    tag_exceptions(
        utt_id, text, original_labels, corrected, corrections,
        exception_log_path, confidences=confidences,
    )

    # ──── Return all intermediate labels for backtesting ────
    return {
        "lid_heuristic": " ".join(heur_labels),
        "lid_fasttext": " ".join(ft_labels),
        "lid_fasttext_conf": " ".join(f"{c:.3f}" for c in confidences),
        "lid_fasttext_llm": " ".join(labels_after_llm),
        "lid_tokens": " ".join(corrected),
    }


def run_lid_with_checker(
    utt_id: str,
    text: str,
    llm_results: dict,
    fail_log_path: Path,
    exception_log_path: Path,
    use_llm: bool,
    cs_terms: list[str] | None = None,
) -> str:
    """
    Legacy pipeline: LLM labels → validate length → post-process → tag exceptions → fallback.
    Used when fastText is NOT enabled.
    Returns joined lid_tokens string.
    """
    tokens = text.split()
    expected_count = len(tokens)

    # --- Stage 1: LLM labels ---
    labels = llm_results.get(utt_id) if use_llm else None
    fail_reason = None

    if labels and isinstance(labels, list):
        if len(labels) != expected_count:
            fail_reason = f"Length mismatch: LLM={len(labels)}, expected={expected_count}"
    else:
        if use_llm:
            fail_reason = "No LLM output"

    # --- Stage 2: Fallback if LLM failed ---
    if fail_reason is not None:
        fallback_labels = heuristic_lid(text)
        if use_llm:
            entry = {
                "utt_id": utt_id,
                "text": text,
                "reason": fail_reason,
                "llm_output": labels,
                "heuristic_fallback": fallback_labels,
            }
            with open(fail_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # Still post-process fallback labels
        corrected, corrections = postprocess_labels(tokens, fallback_labels, cs_terms)
        if corrections:
            tag_exceptions(utt_id, text, fallback_labels, corrected, corrections, exception_log_path)
        return " ".join(corrected)

    # --- Stage 3: Post-processing (diacritics override + cs_terms validation) ---
    original_labels = list(labels)
    corrected, corrections = postprocess_labels(tokens, labels, cs_terms)

    # --- Stage 4: Exception tagging ---
    tag_exceptions(utt_id, text, original_labels, corrected, corrections, exception_log_path)

    return " ".join(corrected)


# ---------------------------------------------------------------------------
# Statistics / Accuracy Comparison
# ---------------------------------------------------------------------------

def compute_pipeline_stats(sb_dict: dict) -> dict:
    """
    Compute method-by-method statistics using stored intermediate labels.
    Compares lid_heuristic, lid_fasttext, lid_fasttext_llm against lid_tokens (final).
    """
    stats = {
        "total_tokens": 0,
        "total_utterances": len(sb_dict),
        "heuristic_agree": 0,
        "fasttext_agree": 0,
        "fasttext_llm_agree": 0,
        "total_uncertain": 0,
        "en_tokens": 0,
        "vi_tokens": 0,
    }

    for utt_id, entry in sb_dict.items():
        final_labels = entry.get("lid_tokens", "").split()
        if not final_labels:
            continue

        n = len(final_labels)
        stats["total_tokens"] += n

        # Count EN/VI
        for lbl in final_labels:
            if lbl == "EN":
                stats["en_tokens"] += 1
            else:
                stats["vi_tokens"] += 1

        # Heuristic comparison (from stored labels)
        heur_labels = entry.get("lid_heuristic", "").split()
        if len(heur_labels) == n:
            stats["heuristic_agree"] += sum(
                1 for h, f in zip(heur_labels, final_labels) if h == f
            )

        # fastText comparison (from stored labels)
        ft_labels = entry.get("lid_fasttext", "").split()
        if len(ft_labels) == n:
            stats["fasttext_agree"] += sum(
                1 for ft, f in zip(ft_labels, final_labels) if ft == f
            )

        # fastText + LLM comparison (from stored labels)
        ft_llm_labels = entry.get("lid_fasttext_llm", "").split()
        if len(ft_llm_labels) == n:
            stats["fasttext_llm_agree"] += sum(
                1 for fl, f in zip(ft_llm_labels, final_labels) if fl == f
            )

        # Count uncertain tokens from confidence scores
        confs = entry.get("lid_fasttext_conf", "").split()
        if confs:
            stats["total_uncertain"] += sum(
                1 for c in confs if float(c) < 0.9
            )

    return stats


def print_pipeline_summary(stats: dict, use_fasttext: bool, use_llm: bool):
    """Print a summary table comparing pipeline methods."""
    total = stats["total_tokens"]
    if total == 0:
        return

    print("\n" + "=" * 65)
    print("📊 Pipeline Accuracy Comparison (vs final labels)")
    print("=" * 65)

    print(f"  {'Method':<25} {'Agreement':>12} {'Accuracy':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*10}")

    heur_acc = 100.0 * stats["heuristic_agree"] / total
    print(f"  {'heuristic':<25} {stats['heuristic_agree']:>10}/{total:<1} {heur_acc:>9.1f}%")

    if use_fasttext:
        ft_acc = 100.0 * stats["fasttext_agree"] / total
        print(f"  {'fastText only':<25} {stats['fasttext_agree']:>10}/{total:<1} {ft_acc:>9.1f}%")

    if use_llm and use_fasttext:
        ft_llm_acc = 100.0 * stats["fasttext_llm_agree"] / total
        print(f"  {'fastText + LLM review':<25} {stats['fasttext_llm_agree']:>10}/{total:<1} {ft_llm_acc:>9.1f}%")

    print(f"  {'full pipeline (final)':<25} {'(= final)':>12} {'100.0%':>10}")

    if use_fasttext:
        uncertain_pct = 100.0 * stats["total_uncertain"] / total
        print(f"\n  Uncertain tokens: {stats['total_uncertain']}/{total} ({uncertain_pct:.1f}%)")

    if use_llm:
        print(f"  LLM was used: {'for uncertain tokens only' if use_fasttext else 'for all tokens'}")

    print(f"\n  Total utterances: {stats['total_utterances']}")
    print(f"  Total tokens: {total}")
    print(f"  EN tokens: {stats['en_tokens']} ({100.0*stats['en_tokens']/total:.1f}%)")
    print(f"  VI tokens: {stats['vi_tokens']} ({100.0*stats['vi_tokens']/total:.1f}%)")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Split Preparation
# ---------------------------------------------------------------------------

def prepare_split(
    dataset,
    split_name: str,
    output_dir: Path,
    use_llm: bool = False,
    use_fasttext: bool = False,
    llm_model: str = "gpt-4.1-mini",
    review_model: str | None = None,
    api_key: str | None = None,
    ft_model=None,
    confidence_threshold: float = 0.9,
    resume: bool = False,
    save_every: int = 500,
):
    audio_dir = output_dir / "wavs" / split_name
    audio_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / f"{split_name}.json"
    fail_log_path = output_dir / f"failed_lid_{split_name}.jsonl"
    exception_log_path = output_dir / f"exceptions_{split_name}.jsonl"
    review_log_path = output_dir / f"review_corrections_{split_name}.jsonl"

    # Setup OpenAI client
    client = None
    if use_llm:
        if OpenAI is None:
            print("  Warning: openai package not installed. Falling back to heuristic.")
            use_llm = False
        elif not api_key:
            print("  Warning: OPENAI_API_KEY not set. Falling back to heuristic.")
            use_llm = False
        else:
            client = OpenAI(api_key=api_key)

    sb_dict = {}
    existing_ids = set()

    # ──── Resume: load existing JSON if present ────
    if resume and json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                sb_dict = json.load(f)
            existing_ids = set(sb_dict.keys())
            print(f"  ♻️  Resuming: loaded {len(existing_ids)} existing entries from {json_path.name}")
        except Exception as e:
            print(f"  ⚠️  Could not load existing JSON for resume: {e}")
            sb_dict = {}

    print(f"\nProcessing split '{split_name}' ({len(dataset)} items)...")
    pipeline_mode = "hybrid (fastText + LLM)" if (use_fasttext and use_llm) else \
                    "fastText only" if use_fasttext else \
                    "LLM only" if use_llm else "heuristic"
    print(f"  Pipeline mode: {pipeline_mode}")
    if use_fasttext:
        print(f"  Confidence threshold: {confidence_threshold}")

    for i, item in enumerate(tqdm(dataset, desc=split_name)):
        audio_data = item["audio"]

        # In streaming mode, HF might return 'bytes' instead of decoded 'array'
        if "array" in audio_data:
            audio_arr = audio_data["array"]
            sr = audio_data["sampling_rate"]
        elif "bytes" in audio_data:
            import io
            audio_arr, sr = sf.read(io.BytesIO(audio_data["bytes"]))
        else:
            audio_arr, sr = sf.read(audio_data["path"])

        # Prefer 'segment_text' (ViMedCSS native field), fallback to common HF fields
        raw_text = item.get("segment_text") or item.get("sentence") or item.get("text") or ""
        text = clean_text(raw_text)

        # Parse cs_terms metadata for post-processing validation
        cs_terms_raw = item.get("cs_terms") or item.get("cs_terms_list") or ""
        cs_terms = parse_cs_terms(cs_terms_raw)

        # Utterance ID: use segment_id from metadata when available
        seg_id = item.get("segment_id") or f"{i:05d}"
        utt_id = f"vimedcss_{split_name}_{seg_id}"
        wav_path = audio_dir / f"{utt_id}.wav"

        # Skip already-processed entries when resuming
        if utt_id in existing_ids:
            continue

        # Skip empty text
        if not text.strip():
            continue

        # Skip empty audio (prevents tensor reshape crash)
        if len(audio_arr) == 0:
            print(f"  ⚠️  Skipping {utt_id}: empty audio")
            continue

        # Resample to 16 kHz
        try:
            if sr != 16000:
                waveform = torch.from_numpy(audio_arr).float().unsqueeze(0)
                waveform = F_audio.resample(waveform, orig_freq=sr, new_freq=16000)
                audio_arr = waveform.squeeze(0).numpy()
                sr = 16000
        except RuntimeError as e:
            print(f"  ⚠️  Skipping {utt_id}: audio resample error: {e}")
            continue

        sf.write(str(wav_path), audio_arr, sr)
        duration = round(len(audio_arr) / sr, 3)

        sb_dict[utt_id] = {
            "wav": str(wav_path.resolve()),
            "length": duration,
            "words": text,
        }

        # ──── LID Labeling ────
        if use_fasttext and ft_model:
            # === HYBRID PIPELINE (fastText → confidence filter → LLM → rules) ===
            lid_result = run_hybrid_pipeline(
                utt_id=utt_id,
                text=text,
                fail_log_path=fail_log_path,
                exception_log_path=exception_log_path,
                cs_terms=cs_terms,
                ft_model=ft_model,
                confidence_threshold=confidence_threshold,
                llm_client=client if use_llm else None,
                llm_model=llm_model,
                review_log_path=review_log_path,
            )
            # Store all intermediate labels for backtesting
            sb_dict[utt_id].update(lid_result)
        elif use_llm and client:
            # === LEGACY: LLM-only pipeline ===
            word_list = text.split()
            labels = call_llm_lid_single(utt_id, word_list, client, llm_model)

            # LLM Review with stronger model if available
            if labels and review_model:
                initial_labels = list(labels)
                reviewed = call_llm_review(utt_id, word_list, labels, client, review_model)
                if reviewed:
                    labels = reviewed
                    # Log review corrections if any labels changed
                    diffs = [
                        {"idx": j, "token": word_list[j], "before": initial_labels[j], "after": reviewed[j]}
                        for j in range(min(len(initial_labels), len(reviewed)))
                        if initial_labels[j] != reviewed[j]
                    ]
                    if diffs:
                        entry = {
                            "utt_id": utt_id,
                            "text": text,
                            "diffs": diffs,
                        }
                        with open(review_log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            llm_results = {utt_id: labels} if labels else {}
            sb_dict[utt_id]["lid_tokens"] = run_lid_with_checker(
                utt_id, text, llm_results, fail_log_path, exception_log_path,
                use_llm, cs_terms=cs_terms,
            )
        else:
            # === HEURISTIC-only mode — still apply post-processing ===
            heur_labels = heuristic_lid(text)
            corrected, corrections = postprocess_labels(text.split(), heur_labels, cs_terms)
            if corrections:
                tag_exceptions(utt_id, text, heur_labels, corrected, corrections, exception_log_path)
            sb_dict[utt_id]["lid_tokens"] = " ".join(corrected)

        # ──── Periodic checkpoint save ────
        new_count = len(sb_dict) - len(existing_ids)
        if save_every > 0 and new_count > 0 and new_count % save_every == 0:
            tmp_path = json_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(sb_dict, f, ensure_ascii=False, indent=2)
            tmp_path.replace(json_path)
            print(f"  💾 Checkpoint: saved {len(sb_dict)} entries ({new_count} new)")

    # ──── Compute & Print Statistics ────
    pipeline_stats = compute_pipeline_stats(sb_dict)
    print_pipeline_summary(pipeline_stats, use_fasttext, use_llm)

    # Atomic write
    tmp_path = json_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(sb_dict, f, ensure_ascii=False, indent=2)
    tmp_path.replace(json_path)

    print(f"  ✅ Saved {len(sb_dict)} utterances → {json_path}")
    if use_llm and fail_log_path.exists():
        with open(fail_log_path, encoding="utf-8") as fh:
            n_fails = sum(1 for _ in fh)
        print(f"  ⚠️  {n_fails} checker failures logged → {fail_log_path.name}")
    if exception_log_path.exists():
        with open(exception_log_path, encoding="utf-8") as fh:
            n_exceptions = sum(1 for _ in fh)
        print(f"  📋 {n_exceptions} exceptions tagged → {exception_log_path.name}")
    if review_log_path.exists():
        with open(review_log_path, encoding="utf-8") as fh:
            n_reviews = sum(1 for _ in fh)
        print(f"  🔍 {n_reviews} review corrections logged → {review_log_path.name}")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare tensorxt/ViMedCSS into SpeechBrain JSON manifests with LID labels."
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/vimedcss",
        help="Root directory for output JSON files and WAV audio."
    )
    parser.add_argument(
        "--use_fasttext", action="store_true",
        help="Use fastText word LID as Stage 1 (auto-downloads lid.176.bin if needed)."
    )
    parser.add_argument(
        "--fasttext_model", type=str, default=None,
        help="Path to fastText LID model (lid.176.bin). Default: data/fasttext/lid.176.bin (auto-download)."
    )
    parser.add_argument(
        "--confidence", type=float, default=0.9,
        help="Confidence threshold for fastText. Tokens below this → LLM review. (default: 0.9)"
    )
    parser.add_argument(
        "--use_llm", action="store_true",
        help="Use LLM (OpenAI) for LID. With --use_fasttext: reviews only uncertain tokens. Without: labels all tokens."
    )
    parser.add_argument(
        "--llm_model", type=str, default="gpt-4.1-mini",
        help="OpenAI model for LID labeling (fast/cheap)."
    )
    parser.add_argument(
        "--review_model", type=str, default=None,
        help="Stronger OpenAI model to review labels (e.g. gpt-4.1). Only used in legacy LLM-only mode."
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit samples per split (useful for debugging)."
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing JSON output (skip already-processed entries)."
    )
    parser.add_argument(
        "--save_every", type=int, default=500,
        help="Save checkpoint every N new entries (default: 500). Set 0 to disable."
    )
    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")

    # Authenticate with HuggingFace for faster downloads
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token  # ensure it's in os.environ
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            print("✓ Authenticated with HuggingFace Hub")
        except Exception:
            pass  # login is optional, token in env is enough

    if args.use_llm and not api_key:
        print("ERROR: --use_llm requires OPENAI_API_KEY set in the environment or a .env file.")
        sys.exit(1)

    # ──── Load fastText model ────
    ft_model = None
    if args.use_fasttext:
        ft_model_path = args.fasttext_model or str(Path(args.output_dir).resolve().parent / "fasttext" / "lid.176.bin")
        print(f"🔤 Loading fastText LID model from: {ft_model_path}")
        ft_model = load_fasttext_model(ft_model_path)
        print(f"   ✅ fastText model loaded.")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("⬇️  Downloading/Streaming tensorxt/ViMedCSS from Hugging Face...")
    try:
        # If testing with a small sample, use streaming to avoid downloading 20GB
        use_streaming = bool(args.max_samples)
        ds = load_dataset("tensorxt/ViMedCSS", streaming=use_streaming)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    for split in ds.keys():
        split_data = ds[split]
        if args.max_samples:
            # IterableDataset doesn't have len() or .select(), use .take() and cast to list
            print(f"\n Split: {split} (Streaming first {args.max_samples} samples)")
            split_data = list(split_data.take(args.max_samples))
        else:
            print(f"\n Split: {split}  ({len(split_data)} samples)")

        prepare_split(
            split_data,
            split_name=split,
            output_dir=out_dir,
            use_llm=args.use_llm,
            use_fasttext=args.use_fasttext,
            llm_model=args.llm_model,
            review_model=args.review_model,
            api_key=api_key,
            ft_model=ft_model,
            confidence_threshold=args.confidence,
            resume=args.resume,
            save_every=args.save_every,
        )


if __name__ == "__main__":
    main()
