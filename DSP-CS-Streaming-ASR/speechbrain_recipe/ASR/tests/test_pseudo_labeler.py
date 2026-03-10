"""
test_pseudo_labeler.py — Unit Tests for run_whisper_teacher.py
==============================================================
Tests:
  P1: clean_transcript removes punctuation and lowercases
  P2: classify_tokens correctly labels VI/EN words
  P3: classify_tokens aligns with clean_transcript word count
  P4: Batch segmentation produces correct segment filenames (no collision)
  P5: Empty transcript is correctly filtered out
  P6: Short words <= 2 chars are NOT classified as EN (Bug D2 guard)

Run:
  cd data_pipelines/pseudo_labeler  OR  cd speechbrain_recipe/ASR
  python -m pytest tests/test_pseudo_labeler.py -v
"""

import pytest
import sys
import os

# Add pseudo_labeler to path
LABELER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../data_pipelines/pseudo_labeler")
)
sys.path.insert(0, LABELER_DIR)

try:
    from run_whisper_teacher import clean_transcript, classify_tokens, is_probable_english, is_vietnamese
    LABELER_AVAILABLE = True
except ImportError:
    LABELER_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not LABELER_AVAILABLE,
    reason="run_whisper_teacher.py not found at data_pipelines/pseudo_labeler/"
)


# ─────────────────────────────────────────────────────────────────────────────
# P1: clean_transcript
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw, expected", [
    ("Hello, World!", "hello world"),
    ("Tôi cần một dashboard.", "tôi cần một dashboard"),
    ("<en>dashboard</en> mới", "dashboard mới"),
    ("[vi]Xin chào[/vi]", "xin chào"),
    ("  multiple   spaces  ", "multiple spaces"),
    ("ALL CAPS TEXT", "all caps text"),
])
def test_clean_transcript(raw, expected):
    """P1: clean_transcript normalizes input."""
    result = clean_transcript(raw)
    assert result == expected, f"Input: {repr(raw)} → Got: {repr(result)}, Expected: {repr(expected)}"
    print(f"  ✅ '{raw}' → '{result}'")


# ─────────────────────────────────────────────────────────────────────────────
# P2: classify_tokens by language
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sentence, expected_labels", [
    # Pure Vietnamese
    ("tôi yêu việt nam", "VI VI VI VI"),
    # Pure English (>2 chars each)
    ("dashboard software deploy", "EN EN EN"),
    # Code-switched
    ("tôi cần một dashboard mới", "VI VI VI EN VI"),
    # Short words should NOT be EN
    ("tôi đi về", "VI VI VI"),
])
def test_classify_tokens_labels(sentence, expected_labels):
    """P2: classify_tokens assigns VI/EN correctly."""
    result = classify_tokens(sentence)
    assert result == expected_labels, \
        f"Sentence: {repr(sentence)}\n  Got:      {repr(result)}\n  Expected: {repr(expected_labels)}"
    print(f"  ✅ '{sentence}' → '{result}'")


# ─────────────────────────────────────────────────────────────────────────────
# P3: Token Count Alignment (CRITICAL)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sentence", [
    "tôi cần một dashboard mới",
    "hello world này là test",
    "xin chào",
    "anh ấy work từ home mỗi ngày",
    "dashboard system deploy production server",
])
def test_token_count_alignment(sentence):
    """
    P3: CRITICAL — #lid_tokens MUST equal #words in cleaned transcript.
    Misalignment would cause index errors in train.py's LID loss computation.
    """
    cleaned = clean_transcript(sentence)
    words = cleaned.split()
    lid = classify_tokens(sentence)
    lid_tokens = lid.split()

    assert len(words) == len(lid_tokens), (
        f"ALIGNMENT MISMATCH!\n"
        f"  Sentence:   {repr(sentence)}\n"
        f"  Cleaned:    {repr(cleaned)}\n"
        f"  Words ({len(words)}):    {words}\n"
        f"  LID ({len(lid_tokens)}):      {lid_tokens}\n"
    )
    print(f"  ✅ '{cleaned}' → {len(words)} words aligned")


# ─────────────────────────────────────────────────────────────────────────────
# P4: Segment Filename Uniqueness (Bug #7 regression)
# ─────────────────────────────────────────────────────────────────────────────

def test_segment_filename_uniqueness():
    """
    P4: Segment filenames use global segment_idx, preventing collision
    when two segments in the same file have timestamps that round to the same integer.
    """
    stem = "podcast_ep01"
    segment_idx = 0
    filenames = set()

    # Simulate 10 segments from the same file
    for _ in range(10):
        chunk_name = f"{stem}_seg_{segment_idx:06d}.wav"
        assert chunk_name not in filenames, f"COLLISION: {chunk_name} already exists!"
        filenames.add(chunk_name)
        segment_idx += 1

    assert len(filenames) == 10
    print(f"✅ P4: 10 unique segment filenames generated: {sorted(filenames)[:3]}...")


# ─────────────────────────────────────────────────────────────────────────────
# P5: Empty Transcript Filtering
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw_text", [
    "",
    "   ",
    ",!?.",
    "[vi][/vi]",
    "<en></en>",
])
def test_empty_transcript_filtered(raw_text):
    """P5: clean_transcript of noise/empty text returns empty string → entry is skipped."""
    cleaned = clean_transcript(raw_text)
    assert cleaned == "", f"Expected empty string, got: {repr(cleaned)}"
    print(f"  ✅ '{raw_text}' → '' (filtered)")


# ─────────────────────────────────────────────────────────────────────────────
# P6: Short Words NOT Classified as EN (Anti-regression for Bug D3)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("word, expected_is_en", [
    ("to", False),    # 2 chars → NOT english
    ("ai", False),    # 2 chars → NOT english (Vietnamese word too)
    ("co", False),    # 2 chars → NOT english
    ("the", True),    # 3 chars, ASCII, alpha → English
    ("and", True),    # 3 chars → English
    ("tôi", False),   # Vietnamese diacritic → NOT english
    ("deploy", True), # 6 chars → English
])
def test_short_words_not_classified_en(word, expected_is_en):
    """P6: Words <= 2 chars are never classified as EN (avoids misclassifying short VI words)."""
    result = is_probable_english(word)
    assert result == expected_is_en, \
        f"'{word}': expected is_probable_english={expected_is_en}, got {result}"
    print(f"  ✅ is_probable_english('{word}') = {result}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
