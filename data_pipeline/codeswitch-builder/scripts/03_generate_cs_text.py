#!/usr/bin/env python3
"""
Step 3 — Code-Switch Text Generation (OpenAI Tool Calling)
================================================

Reads bilingual pairs from Step 2 and generates Vietnamese-English
code-switched sentences at 3 density levels using OpenAI Tool Calling
to enforce strict structured output.

Also generates per-token LID (Language ID) labels for training the
Causal GRU LID Head in the thesis ASR model.

  --level light   : 5-18%  English
  --level medium  : 15-35% English
  --level heavy   : 30-55% English
  --level all     : Run all 3
  --level mix     : Blend from 3 output files with --mix-ratio

Output per entry:
  { "vi", "en", "cs", "lid": [0,0,1,1,...], "level", "en_ratio", "en_spans", "valid" }

LID label: 0 = Vietnamese token, 1 = English token (per whitespace-split token)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from dotenv import load_dotenv
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: 'openai' package not installed. Run: pip install openai python-dotenv")
    sys.exit(1)

load_dotenv()
RAW_API_KEYS = os.getenv("OPENAI_API_KEY", "")
API_KEYS = [k.strip() for k in RAW_API_KEYS.split(",") if k.strip()]
CURRENT_KEY_IDX = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"


# =====================================================================
# Validation & LID Helpers
# =====================================================================

def clean_token(t: str) -> str:
    return re.sub(r"[^\w]", "", t.lower())

def is_ascii_alpha(t: str) -> bool:
    ct = clean_token(t)
    return ct.isascii() and ct.isalpha() and len(ct) > 0

def english_words_from_ref(sentence: str) -> set:
    """Trả về tập các token ASCII từ câu tham chiếu EN.
    Chỉ gọi trên câu tiếng Anh để lấy reference words,
    không gọi trên câu CS (tránh đếm nhầm từ Việt không dấu).
    """
    return {clean_token(t) for t in sentence.split() if is_ascii_alpha(t)}


def generate_lid_labels(cs_sentence: str, en_ref_words: set) -> List[int]:
    """Per-token LID: 1=English, 0=Vietnamese. Uses EN reference to avoid false positives."""
    return [1 if clean_token(t) in en_ref_words else 0 for t in cs_sentence.split()]

def english_ratio(cs_sentence: str, en_ref_words: set) -> float:
    tokens = cs_sentence.split()
    if not tokens:
        return 0.0
    return sum(1 for t in tokens if clean_token(t) in en_ref_words) / len(tokens)

def count_en_spans(cs_sentence: str, en_ref_words: set) -> int:
    tokens = cs_sentence.split()
    spans, in_en = 0, False
    for t in tokens:
        if clean_token(t) in en_ref_words:
            if not in_en:
                spans += 1
                in_en = True
        else:
            in_en = False
    return spans

def is_valid_cs(cs: str, vi: str, en: str, min_ratio: float, max_ratio: float, max_spans: int) -> Tuple[bool, str]:
    tokens = cs.split()
    en_ref_words = english_words_from_ref(en)
    vi_words = {clean_token(t) for t in vi.split()}

    if len(tokens) < 4:
        return False, f"Too short ({len(tokens)} tokens)"
    if len(tokens) > 45:
        return False, f"Too long ({len(tokens)} tokens)"
    
    ratio = english_ratio(cs, en_ref_words)
    if ratio < min_ratio:
        return False, f"English ratio too low ({ratio:.1%})"
    if ratio > max_ratio:
        return False, f"English ratio too high ({ratio:.1%})"
    
    spans = count_en_spans(cs, en_ref_words)
    if spans < 1:
        return False, "No English spans found"
    if spans > max_spans:
        return False, f"Too many English spans ({spans})"

    # Hallucination check logic:
    # Lấy tất cả các từ ascii alphabetic do LLM sinh ra
    cs_ascii_words = {clean_token(t) for t in tokens if is_ascii_alpha(t)}
    # Bất kỳ từ ascii nào KHÔNG có trong câu gốc (vi) VÀ KHÔNG có trong câu đích (en) đều là bịa (hallucination)
    hallucinated = cs_ascii_words - en_ref_words - vi_words
    if hallucinated:
        return False, f"Hallucinated English: {hallucinated}"

    return True, "OK"



# =====================================================================
# Prompts
# =====================================================================

PROMPT_LIGHT = """Bạn là người Việt Nam nói chuyện tự nhiên, thỉnh thoảng dùng một vài từ tiếng Anh.

Tôi sẽ cho bạn các cặp câu. Nhiệm vụ: Viết lại thành câu MIX NHẸ.

QUY TẮC (LIGHT — 5-15% English):
1. Giữ nguyên tiếng Việt có dấu đầy đủ.
2. Chỉ chêm TÊN RIÊNG và THUẬT NGỮ ĐƠN bằng tiếng Anh (1-2 từ/câu).
   - Tên người, địa danh: "Vladimir Putin", "Singapore", "New York", "NATO"
   - Thuật ngữ ngắn: "green card", "deadline", "server"
3. KHÔNG chêm nguyên cụm câu tiếng Anh dài.
4. Nghĩa GIỐNG HỆT câu gốc. Chỉ dùng từ Anh có trong "en".

Ví dụ:
- "sẽ nhận được căn hộ ở thành phố" → "sẽ nhận được căn hộ ở New York City."
- "tổng thống ký lệnh" → "tổng thống Vladimir Putin ký lệnh"
"""

PROMPT_MEDIUM = """Bạn là người Việt Nam làm việc chuyên nghiệp, hay xen tiếng Anh vào câu.

Tôi sẽ cho bạn các cặp câu. Nhiệm vụ: Viết lại thành câu MIX VỪA PHẢI.

QUY TẮC (MEDIUM — 15-30% English):
1. Giữ nguyên tiếng Việt có dấu (KHÔNG xóa dấu).
2. Thay 1-2 CỤM tiếng Anh liền nhau (2-4 từ/cụm) vào câu.
   - Cụm danh từ: "permanent residency", "the capital city", "press statement"
   - Cụm động từ: "to protest against", "to correct this statement"
   - Tên đầy đủ: "President Vladimir Putin", "New York City"
3. Mục tiêu: 15-30% tổng từ là tiếng Anh. Chỉ dùng từ Anh từ "en".

Ví dụ:
- "Tướng Campuchia yêu cầu thủ tướng Singapore phải correct this false statement theo lời ông."
- "Khoảng mười ngàn người Nga biểu tình để protest against the government of President Putin."
"""

PROMPT_HEAVY = """Bạn là người Việt Nam làm việc quốc tế, thường mix tiếng Anh nhiều vào câu.

Tôi sẽ cho bạn các cặp câu. Nhiệm vụ: Viết lại thành câu MIX NẶNG.

QUY TẮC (HEAVY — 30-50% English):
1. Giữ khung tiếng Việt có dấu (chủ ngữ, từ kết nối).
2. Thay các cụm NP/VP lớn bằng tiếng Anh nguyên văn từ "en".
3. Mục tiêu: 30-50% tổng từ là tiếng Anh, 2-3 đoạn tiếng Anh.
4. Không đọc hoàn toàn tiếng Anh — vẫn cần phần tiếng Việt rõ ràng.
5. Nghĩa GIỐNG HỆT câu gốc. Chỉ dùng từ Anh từ "en".

Ví dụ:
- "Theo bà, đây là a widely acknowledged fact that has been recognized."
- "Anh sẽ receive a new clothing collection, an undisclosed amount of money, và một căn hộ ở New York City."
"""

PROMPT_BY_LEVEL = {"light": PROMPT_LIGHT, "medium": PROMPT_MEDIUM, "heavy": PROMPT_HEAVY}

# =====================================================================
# Level Configs
# =====================================================================

LEVEL_CONFIG = {
    "light":  {"min_ratio": 0.05, "max_ratio": 0.18, "max_spans": 2,
                "description": "Chỉ chêm tên riêng & thuật ngữ đơn (5-18% English)"},
    "medium": {"min_ratio": 0.15, "max_ratio": 0.35, "max_spans": 3,
                "description": "Chêm cụm danh từ/động từ kỹ thuật (15-35% English)"},
    "heavy":  {"min_ratio": 0.30, "max_ratio": 0.55, "max_spans": 4,
                "description": "Chêm đoạn câu lớn tiếng Anh (30-55% English)"},
}


# =====================================================================
# OpenAI Tool Calling — Dynamic Schema per batch
# =====================================================================

def build_cs_tool(batch_ids: List[str]) -> list:
    """
    Build tool schema with each sentence ID as an explicit required property.
    OpenAI tool calling does NOT fill 'additionalProperties' dynamic keys —
    it only fills explicitly declared properties with required[].
    """
    properties = {
        bid: {
            "type": "string",
            "description": f"Code-switched Vietnamese-English sentence for id='{bid}'",
        }
        for bid in batch_ids
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "submit_cs_sentences",
                "description": "Submit the code-switched sentences for each provided sentence ID.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": batch_ids,
                },
            },
        }
    ]


def call_openai_batch(
    batch_ids: List[str],
    data: dict,
    level: str,
    fallback_list: List[str],
) -> List[str]:
    """Gọi GPT-4o-mini qua Tool Calling để đảm bảo output đúng schema."""
    global CURRENT_KEY_IDX

    system_prompt = PROMPT_BY_LEVEL[level]
    user_msg = "DATA:\n" + json.dumps(data, ensure_ascii=False, indent=2)
    tools = build_cs_tool(batch_ids)
    max_retries = 3

    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=API_KEYS[CURRENT_KEY_IDX])
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "submit_cs_sentences"}},
                temperature=0.7,
                max_tokens=2048,
            )

            # Tool calling: arguments là JSON string guaranteed by OpenAI
            tool_args_str = response.choices[0].message.tool_calls[0].function.arguments
            tool_args = json.loads(tool_args_str)

            final_list = []
            for idx, bid in enumerate(batch_ids):
                val = tool_args.get(bid)
                if isinstance(val, str) and val.strip():
                    final_list.append(val.strip())
                else:
                    print(f"  Warning: LLM missed '{bid}', using fallback.")
                    final_list.append(fallback_list[idx])
            return final_list

        except Exception as e:
            err_msg = str(e).lower()
            print(f"  OpenAI Error (attempt {attempt+1}, key={CURRENT_KEY_IDX}): {e}")
            if "rate limit" in err_msg or "429" in err_msg or "insufficient_quota" in err_msg:
                CURRENT_KEY_IDX = (CURRENT_KEY_IDX + 1) % len(API_KEYS)
                print(f"  -> Rotated to key idx={CURRENT_KEY_IDX}")
                time.sleep(1)
            if attempt == max_retries - 1:
                return fallback_list
            time.sleep(2)


# =====================================================================
# Core Process
# =====================================================================

def process_level(args, level: str):
    cfg = LEVEL_CONFIG[level]
    input_json = DATA_DIR / f"vi_en_pairs_{args.step2_mode}.json"
    output_json = DATA_DIR / f"cs_text_{level}.json"

    print(f"\n{'='*55}")
    print(f"  Level: {level.upper()} — {cfg['description']}")
    print(f"  Input : {input_json.name}  Output: {output_json.name}")
    print(f"{'='*55}")

    if not input_json.exists():
        print(f"ERROR: {input_json} not found. Run Step 2 first.")
        return

    if args.overwrite and output_json.exists():
        output_json.unlink()
        print(f"[INFO] Removed old {output_json.name}")
    elif args.append:
        print(f"[INFO] Append mode.")

    results = {}
    if output_json.exists():
        with open(output_json, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Checkpoint: {len(results)} existing entries.")

    with open(input_json, "r", encoding="utf-8") as f:
        pairs: Dict[str, dict] = json.load(f)

    all_ids = sorted(pairs.keys())[:args.max_samples]
    pending_ids = [pid for pid in all_ids if pid not in results]

    if not pending_ids:
        print("Already complete!")
        return

    print(f"Pending: {len(pending_ids)} sentences")
    stats = {"valid": 0, "invalid": 0, "fallback": 0}
    output_json.parent.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(total=len(pending_ids), desc=f"CS [{level}]", unit="sent")

    for i in range(0, len(pending_ids), args.batch_size):
        batch_ids = pending_ids[i: i + args.batch_size]
        payload, fallback_vi = {}, []

        for bid in batch_ids:
            vi_text = pairs[bid]["vi"]
            en_text = pairs[bid].get("en", pairs[bid].get("en_draft", vi_text))
            payload[bid] = {"vi": vi_text, "en": en_text}
            fallback_vi.append(vi_text)

        cs_sentences = call_openai_batch(batch_ids, payload, level, fallback_list=fallback_vi)

        for bid, cs_sentence in zip(batch_ids, cs_sentences):
            vi_text = payload[bid]["vi"]
            en_text = payload[bid]["en"]
            en_ref_words = english_words_from_ref(en_text)

            valid, reason = is_valid_cs(cs_sentence, vi_text, en_text, cfg["min_ratio"], cfg["max_ratio"], cfg["max_spans"])
            lid_labels = generate_lid_labels(cs_sentence, en_ref_words)
            ratio = english_ratio(cs_sentence, en_ref_words)
            spans = count_en_spans(cs_sentence, en_ref_words)

            is_fallback = (cs_sentence == vi_text)
            if is_fallback:
                stats["fallback"] += 1
            elif valid:
                stats["valid"] += 1
            else:
                stats["invalid"] += 1

            results[bid] = {
                "vi": vi_text, "en": en_text, "cs": cs_sentence,
                "lid": lid_labels, "level": level,
                "en_ratio": round(ratio, 3), "en_spans": spans,
                "valid": valid and not is_fallback,
                **({"reason": reason} if not valid or is_fallback else {}),
                **({"reason": "API fallback"} if is_fallback else {}),
            }

        temp = output_json.with_suffix(".tmp")
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        temp.replace(output_json)

        pbar.update(len(batch_ids))
        time.sleep(0.3)

    pbar.close()
    total = sum(stats.values())
    print(f"  Valid: {stats['valid']} | Invalid: {stats['invalid']} | Fallback: {stats['fallback']}")
    print(f"  Valid Rate: {stats['valid']/total:.1%}" if total else "  No data.")
    print(f"  Saved -> {output_json.resolve()}")


def mix_levels(args):
    import random
    output_json = DATA_DIR / "cs_text_mixed.json"
    ratio_l, ratio_m, ratio_h = args.mix_ratio
    total_w = ratio_l + ratio_m + ratio_h
    level_names = ["light", "medium", "heavy"]

    print(f"\n{'='*55}")
    print(f"  Level: MIX — Light:{ratio_l} / Medium:{ratio_m} / Heavy:{ratio_h}")
    print(f"  Output: {output_json.name}")
    print(f"{'='*55}")

    min_tokens = getattr(args, "min_tokens", 8)  # Câu ngắn < 8 token không ổn định CS ratio

    level_data = {}
    for level in level_names:
        f = DATA_DIR / f"cs_text_{level}.json"
        if not f.exists():
            print(f"ERROR: {f.name} not found. Run --level {level} first.")
            return
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        level_data[level] = [
            v for v in data.values()
            if v.get("reason") != "API fallback"           # Loại bỏ fallback (cs == vi)
            and len(v.get("cs", "").split()) >= min_tokens  # Loại câu quá ngắn
            and not str(v.get("reason", "")).startswith("Hallucinated") # Loại câu bị LLM bịa từ
        ]
        valid_count = sum(1 for v in level_data[level] if v.get("valid", False))
        print(f"  {level.capitalize()}: {len(level_data[level])} entries ({valid_count} valid, ≥{min_tokens} tokens, no fallback)")

    total_target = args.max_samples or 50000
    random.seed(42)
    mixed = []
    for level, weight in zip(level_names, [ratio_l, ratio_m, ratio_h]):
        count = round(total_target * weight / total_w)
        available = level_data[level]
        sampled = random.sample(available, min(count, len(available)))
        mixed.extend(sampled)
        print(f"  Sampled {len(sampled)} from {level}")

    random.shuffle(mixed)
    result_dict = {f"mix_{i:06d}": item for i, item in enumerate(mixed)}

    if args.overwrite and output_json.exists():
        output_json.unlink()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    level_counts = {l: sum(1 for v in result_dict.values() if v.get("level") == l) for l in level_names}
    print(f"\n  Mix complete: {len(result_dict)} total")
    for l, c in level_counts.items():
        print(f"    {l}: {c} ({c/len(result_dict):.1%})")
    print(f"  Saved -> {output_json.resolve()}")


def process_step_3(args):
    if args.level == "mix":
        mix_levels(args)
        return
    if not API_KEYS:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)
    print(f"Loaded {len(API_KEYS)} OpenAI API key(s). Using Tool Calling.")
    levels_to_run = ["light", "medium", "heavy"] if args.level == "all" else [args.level]
    for level in levels_to_run:
        process_level(args, level)
    print("\n=== Step 3 Complete ===")


def main():
    parser = argparse.ArgumentParser(description="Step 3: Code-Switch Text Generation (Tool Calling + LID)")
    parser.add_argument("--step2-mode", choices=["gpt_only", "nllb_gpt"], default="gpt_only")
    parser.add_argument("--level", choices=["light", "medium", "heavy", "all", "mix"], default="medium")
    parser.add_argument("--mix-ratio", type=int, nargs=3, default=[60, 30, 10],
                        metavar=("LIGHT", "MEDIUM", "HEAVY"),
                        help="Sampling ratio for mix mode (default: 60 30 10)")
    parser.add_argument("--min-tokens", type=int, default=8,
                        help="[mix only] Minimum CS tokens to include in mix pool (default: 8)")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Sentences per API call (keep <=10 for tool calling reliability)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--overwrite", action="store_true")
    group.add_argument("--append", action="store_true")
    args = parser.parse_args()
    process_step_3(args)


if __name__ == "__main__":
    main()
