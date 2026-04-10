#!/usr/bin/env python3
"""
Step 2 — Draft Translation & LLM Refinement (Production Ready)
==================================================

Supports 2 translation modes for Ablation Study:
  --mode nllb_gpt : NLLB-200 (local draft) + GPT-4o-mini (refinement) [2-stage Pipeline]
  --mode gpt_only : GPT-4o-mini direct translation from Vi -> En    [1-stage, higher quality]

Output:
  data/processed/vi_en_pairs_{mode}.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from tqdm import tqdm

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: 'openai' package not installed. Run: pip install openai python-dotenv")
    sys.exit(1)

# Load environment variables (API keys)
load_dotenv()
RAW_API_KEYS = os.getenv("OPENAI_API_KEY", "")
API_KEYS = [k.strip() for k in RAW_API_KEYS.split(",") if k.strip()]
HF_TOKEN = os.getenv("HF_TOKEN", "")
CURRENT_KEY_IDX = 0

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
VI_TEXT_FILE = REPO_ROOT / "data" / "processed" / "vi_text.txt"

MODEL_NAME = "facebook/nllb-200-distilled-600M"

# ---------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------
REFINE_PROMPT = """You are a highly-skilled Bilingual Translation Editor.
I will provide you with a JSON object where each key is an ID and the value contains:
- "vi": ORIGINAL Vietnamese sentence.
- "en_draft": DRAFT English translation (machine translated).

REWRITE each English draft to be natural, grammatically correct English, STRICTLY preserving 100% of the Vietnamese meaning.

CRITICAL RULES:
1. DO NOT add or remove information (especially dates, times, numbers, names).
   - If "vi" says "ngày mười ba tháng năm", output MUST contain "May 13th".
2. Fix phonetic name spellings in Vietnamese:
   - "niu oóc" -> "New York", "na tô" -> "NATO", "cam pu chia" -> "Cambodia",
   - "sinh ga po" -> "Singapore", "pu tin" -> "Putin", "vờ la đi mia" -> "Vladimir".
3. For fragmented/conversational text, translate the intended meaning naturally.

Return only a valid raw JSON object (no markdown, no ```). Keys = same IDs I gave.
Example:
{
  "vi_000000": "Refined English sentence.",
  "vi_000001": "Another refined sentence."
}"""

DIRECT_TRANSLATE_PROMPT = """You are an expert Vietnamese-to-English translator.
I will provide you with a JSON object where each key is an ID and the value is:
- "vi": A Vietnamese sentence to translate directly to English.

Translate each Vietnamese sentence into NATURAL, IDIOMATIC English.

CRITICAL RULES:
1. Translate the COMPLETE meaning. Do not omit dates, numbers, names, or context.
2. Correct well-known phonetic misspellings of proper names:
   - "niu oóc" -> "New York", "na tô" -> "NATO", "cam pu chia" -> "Cambodia",
   - "sinh ga po" -> "Singapore", "pu tin" -> "Putin", "vờ la đi mia" -> "Vladimir".
3. For fragmented/spoken Vietnamese (lots of "ờ", "mà"), capture the intended meaning naturally.

Return ONLY a valid raw JSON object (no markdown, no ```). Keys = same IDs I gave.
Example:
{
  "vi_000000": "Translated English sentence.",
  "vi_000001": "Another translated sentence."
}"""


# ---------------------------------------------------------------------
# NLLB-200 Draft Translation
# ---------------------------------------------------------------------
def load_nllb_model(device: str):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    print(f"Loading NLLB-200 Translation Model on {device.upper()}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang="vie_Latn", token=HF_TOKEN or None)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, use_safetensors=True, token=HF_TOKEN or None).to(device)
    return tokenizer, model


def batch_translate_nllb(texts: List[str], tokenizer, model, device: str, batch_size: int = 32) -> List[str]:
    """Dịch thô (Vi -> En) sử dụng NLLB-200."""
    import torch
    translated_texts = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        eng_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")

        with torch.no_grad():
            generated_tokens = model.generate(**inputs, forced_bos_token_id=eng_token_id, max_length=128)

        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        translated_texts.extend(decoded)

    return translated_texts


# ---------------------------------------------------------------------
# OpenAI API Calls (Both modes use this)
# ---------------------------------------------------------------------
def call_openai(batch_ids: List[str], prompt_template: str, data: dict, fallback_list: List[str]) -> List[str]:
    """Gọi OpenAI API với JSON Input/Output chuẩn. Dùng cho cả 2 Mode."""
    global CURRENT_KEY_IDX

    prompt = "DATA:\n" + json.dumps(data, ensure_ascii=False, indent=2)
    max_retries = 3

    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=API_KEYS[CURRENT_KEY_IDX])
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt_template},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=2048,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content.strip()
            result_dict = json.loads(content)

            # Map result back to ordered list using batch_ids
            final_list = []
            for idx, bid in enumerate(batch_ids):
                if bid in result_dict and isinstance(result_dict[bid], str):
                    final_list.append(result_dict[bid])
                else:
                    print(f"  Warning: LLM missed ID '{bid}', using fallback.")
                    final_list.append(fallback_list[idx])

            return final_list

        except Exception as e:
            err_msg = str(e).lower()
            print(f"  OpenAI Error (attempt {attempt+1}, Key idx={CURRENT_KEY_IDX}): {e}")

            if "rate limit" in err_msg or "429" in err_msg or "insufficient_quota" in err_msg:
                CURRENT_KEY_IDX = (CURRENT_KEY_IDX + 1) % len(API_KEYS)
                print(f"  -> Rotated to next API Key (idx={CURRENT_KEY_IDX})")
                time.sleep(1)
            
            if attempt == max_retries - 1:
                return fallback_list
            time.sleep(2)


# ---------------------------------------------------------------------
# Mode A: NLLB-200 Draft -> GPT-4o-mini Refinement
# ---------------------------------------------------------------------
def translate_batch_nllb_gpt(batch_ids, batch_vi, tokenizer, model, device, batch_nllb):
    # Step A1: Draft translation (local)
    draft_en = batch_translate_nllb(batch_vi, tokenizer, model, device, batch_size=batch_nllb)

    # Step A2: Build refine payload with draft
    data = {bid: {"vi": vi, "en_draft": draft} for bid, vi, draft in zip(batch_ids, batch_vi, draft_en)}
    refined_en = call_openai(batch_ids, REFINE_PROMPT, data, fallback_list=draft_en)

    return [{"vi": vi, "en_draft": draft, "en": refined}
            for vi, draft, refined in zip(batch_vi, draft_en, refined_en)]


# ---------------------------------------------------------------------
# Mode B: GPT-4o-mini Direct Translation
# ---------------------------------------------------------------------
def translate_batch_gpt_only(batch_ids, batch_vi):
    # Build direct translate payload (no draft)
    data = {bid: {"vi": vi} for bid, vi in zip(batch_ids, batch_vi)}
    translated_en = call_openai(batch_ids, DIRECT_TRANSLATE_PROMPT, data, fallback_list=batch_vi)

    return [{"vi": vi, "en": translated}
            for vi, translated in zip(batch_vi, translated_en)]


# ---------------------------------------------------------------------
# Core WorkFlow
# ---------------------------------------------------------------------
def load_vi_text(max_samples: int) -> List[str]:
    if not VI_TEXT_FILE.exists():
        print(f"Error: {VI_TEXT_FILE} not found. Run 01_prepare_data.py first.")
        sys.exit(1)
    with open(VI_TEXT_FILE, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()][:max_samples]


def process_step_2(args):
    mode = args.mode
    output_json = REPO_ROOT / "data" / "processed" / f"vi_en_pairs_{mode}.json"
    
    print(f"=== Step 2: Translate [MODE: {mode.upper()}] ===")

    # Handle overwrite / append
    if args.overwrite:
        if output_json.exists():
            output_json.unlink()
            print(f"[INFO] Overwrite: Removed {output_json.name}")
    elif args.append:
        print(f"[INFO] Append: Will extend {output_json.name}")

    if not API_KEYS:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    print(f"Loaded {len(API_KEYS)} OpenAI API key(s) for auto-rotation.")

    # Load checkpoint
    results = {}
    if output_json.exists():
        with open(output_json, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"Checkpoint loaded: {len(results)} existing sentences.")

    # Load Vietnamese text
    vi_lines = load_vi_text(args.max_samples)
    print(f"Loaded {len(vi_lines)} source sentences.")

    # Filter pending
    pending_tasks = [(f"vi_{idx:06d}", text) for idx, text in enumerate(vi_lines) if f"vi_{idx:06d}" not in results]

    if not pending_tasks:
        print("Checkpoint complete! Nothing to process.")
        return

    print(f"Pending: {len(pending_tasks)} sentences to translate.")

    # Load NLLB if needed
    tokenizer, model, device = None, None, None
    if mode == "nllb_gpt":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer, model = load_nllb_model(device)

    # Process loop
    output_json.parent.mkdir(parents=True, exist_ok=True)
    pbar = tqdm(total=len(pending_tasks), desc=f"Translating [{mode}]", unit="sent")

    for i in range(0, len(pending_tasks), args.batch_llm):
        batch = pending_tasks[i: i + args.batch_llm]
        batch_ids = [t[0] for t in batch]
        batch_vi = [t[1] for t in batch]

        if mode == "nllb_gpt":
            batch_results = translate_batch_nllb_gpt(batch_ids, batch_vi, tokenizer, model, device, args.batch_nllb)
        else:  # gpt_only
            batch_results = translate_batch_gpt_only(batch_ids, batch_vi)

        for bid, res in zip(batch_ids, batch_results):
            results[bid] = res

        # Write checkpoint atomically
        temp = output_json.with_suffix(".tmp")
        with open(temp, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        temp.replace(output_json)

        pbar.update(len(batch))
        time.sleep(0.3)

    pbar.close()
    print(f"\nCompleted! Results saved to: {output_json.resolve()}")


def main():
    parser = argparse.ArgumentParser(description="Step 2: Vi->En Translation with Ablation Mode")
    parser.add_argument("--mode", choices=["nllb_gpt", "gpt_only"], required=True,
                        help="Translation mode: 'nllb_gpt' (2-stage) or 'gpt_only' (1-stage)")
    parser.add_argument("--max-samples", type=int, default=50000,
                        help="Max sentences to process from Step 1")
    parser.add_argument("--batch-nllb", type=int, default=32,
                        help="[nllb_gpt only] Batch size for local NLLB model (GPU VRAM dependent)")
    parser.add_argument("--batch-llm", type=int, default=20,
                        help="Batch size per OpenAI API call")

    # Safety: require explicit mode to avoid accidental overwrite
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--overwrite", action="store_true", help="Clear output and start fresh")
    group.add_argument("--append", action="store_true", help="Resume from existing checkpoint")

    args = parser.parse_args()
    process_step_2(args)


if __name__ == "__main__":
    main()
