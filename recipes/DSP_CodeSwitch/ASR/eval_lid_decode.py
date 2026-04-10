"""
LID-Aware CTC Decoding Evaluation
===================================
Load v3 checkpoint and evaluate test set with LID-conditioned vocab bias.
When LID predicts Vietnamese → boost Vietnamese BPE token logits
When LID predicts English → boost English BPE token logits

Usage:
    python3 eval_lid_decode.py train_xlsr_v3.yaml --bias_strength 0.5
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.metric_stats import ErrorRateStats
from speechbrain.decoders.ctc import ctc_greedy_decode

import logging
logger = logging.getLogger(__name__)


def classify_bpe_vocab(sp_model):
    """Classify each BPE token as Vietnamese-only, English-only, or shared.
    
    Returns:
        vi_mask: tensor [vocab_size], 1.0 for VI tokens, 0.0 otherwise
        en_mask: tensor [vocab_size], 1.0 for EN tokens, 0.0 otherwise
    """
    vi_diacritics = set('àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ')
    vi_diacritics |= set(c.upper() for c in vi_diacritics)
    
    latin_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    vocab_size = sp_model.get_piece_size()
    vi_mask = torch.zeros(vocab_size)
    en_mask = torch.zeros(vocab_size)
    
    for i in range(vocab_size):
        piece = sp_model.id_to_piece(i)
        piece_clean = piece.replace('▁', '')
        if not piece_clean:
            continue
        
        has_vi_char = any(c in vi_diacritics for c in piece_clean)
        is_pure_latin = all(c in latin_chars for c in piece_clean) and len(piece_clean) > 1
        
        if has_vi_char:
            vi_mask[i] = 1.0
        elif is_pure_latin:
            en_mask[i] = 1.0
    
    return vi_mask, en_mask


def lid_aware_ctc_decode(log_probs, lid_logits, vi_mask, en_mask, bias_strength=0.5):
    """Apply LID-conditioned bias to CTC logits before greedy decoding.
    
    Args:
        log_probs: [B, T, V] - CTC log probabilities
        lid_logits: [B, T, 3] - LID logits (SIL=0, VI=1, EN=2)
        vi_mask: [V] - 1.0 for Vietnamese tokens
        en_mask: [V] - 1.0 for English tokens
        bias_strength: float - how much to boost (0.0 = no bias, 1.0 = strong)
    
    Returns:
        decoded_ids: list of lists of token IDs
    """
    # Get LID probabilities per frame
    lid_probs = F.softmax(lid_logits, dim=-1)  # [B, T, 3]
    
    # VI probability (index 1), EN probability (index 2)
    vi_prob = lid_probs[:, :, 1].unsqueeze(-1)  # [B, T, 1]
    en_prob = lid_probs[:, :, 2].unsqueeze(-1)  # [B, T, 1]
    
    # Create bias: when LID thinks VI → boost VI tokens, suppress EN tokens
    # when LID thinks EN → boost EN tokens, suppress VI tokens
    vi_mask_expanded = vi_mask.unsqueeze(0).unsqueeze(0).to(log_probs.device)  # [1, 1, V]
    en_mask_expanded = en_mask.unsqueeze(0).unsqueeze(0).to(log_probs.device)  # [1, 1, V]
    
    # Bias = VI_prob * VI_mask + EN_prob * EN_mask (positive bias for correct language)
    # Also subtract bias for WRONG language
    bias = bias_strength * (
        vi_prob * vi_mask_expanded - vi_prob * en_mask_expanded +
        en_prob * en_mask_expanded - en_prob * vi_mask_expanded
    )
    
    # Apply bias to log probs
    biased_log_probs = log_probs + bias
    
    # Greedy decode
    # Get most probable token at each frame
    predictions = biased_log_probs.argmax(dim=-1)  # [B, T]
    
    # CTC collapse: remove consecutive duplicates and blanks (blank_index=0)
    decoded = []
    for b in range(predictions.shape[0]):
        seq = predictions[b].tolist()
        collapsed = []
        prev = -1
        for token_id in seq:
            if token_id != prev:
                if token_id != 0:  # skip blank
                    collapsed.append(token_id)
                prev = token_id
        decoded.append(collapsed)
    
    return decoded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("yaml_file", help="YAML config file")
    parser.add_argument("--bias_strength", type=float, default=0.5,
                        help="LID bias strength (0.0=none, 1.0=strong)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # Load config
    with open(args.yaml_file) as f:
        hparams = load_hyperpyyaml(f)
    
    # Setup
    device = torch.device(args.device)
    tokenizer = hparams["tokenizer"]
    sp = tokenizer.sp
    
    # Classify vocabulary
    vi_mask, en_mask = classify_bpe_vocab(sp)
    logger.info(f"Vocab: {vi_mask.sum().int()} VI tokens, {en_mask.sum().int()} EN tokens, "
                f"{(vi_mask + en_mask == 0).sum().int() - 1} shared tokens")
    
    # Load model
    dsp_model = hparams["dsp_model"].to(device)
    ctc_lin = hparams["ctc_lin"].to(device)
    log_softmax = hparams["log_softmax"]
    
    # Load checkpoint
    checkpointer = hparams["checkpointer"]
    checkpointer.recover_if_possible()
    
    dsp_model.eval()
    ctc_lin.eval()
    
    # Load test data
    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_annotation"],
        output_keys=["id", "sig", "words"],
    )
    test_data.add_dynamic_item(
        sb.dataio.dataio.read_audio, takes="wav", provides="sig"
    )
    
    # Metrics
    wer_standard = ErrorRateStats()
    wer_lid = ErrorRateStats()
    cswer_standard = ErrorRateStats()
    cswer_lid = ErrorRateStats()
    
    # Evaluate
    total = len(test_data)
    logger.info(f"Evaluating {total} test samples with bias={args.bias_strength}")
    
    with torch.no_grad():
        for i, sample_id in enumerate(test_data.data_ids):
            sample = test_data.data[sample_id]
            wav_path = sample["wav"]
            ref_words = sample["words"]
            
            # Load audio
            sig = sb.dataio.dataio.read_audio(wav_path).unsqueeze(0).to(device)
            wav_lens = torch.tensor([1.0]).to(device)
            
            # Forward
            dsp_model.warmup = False
            features, lid_logits, _ = dsp_model(sig, wav_lens)
            logits = ctc_lin(features)
            log_probs = log_softmax(logits)
            
            # Standard CTC greedy decode (no bias)
            std_preds = log_probs.argmax(dim=-1)  # [1, T]
            std_seq = std_preds[0].tolist()
            std_collapsed = []
            prev = -1
            for tid in std_seq:
                if tid != prev:
                    if tid != 0:
                        std_collapsed.append(tid)
                    prev = tid
            std_text = sp.decode(std_collapsed)
            
            # LID-Aware decode
            lid_decoded = lid_aware_ctc_decode(
                log_probs, lid_logits, vi_mask, en_mask,
                bias_strength=args.bias_strength
            )
            lid_text = sp.decode(lid_decoded[0])
            
            # Compute WER
            ref_tokens = ref_words.split()
            std_tokens = std_text.split()
            lid_tokens = lid_text.split()
            
            wer_standard.append(
                ids=[sample_id],
                predict=[std_tokens],
                target=[ref_tokens],
            )
            wer_lid.append(
                ids=[sample_id],
                predict=[lid_tokens],
                target=[ref_tokens],
            )
            
            if (i + 1) % 200 == 0:
                std_wer = wer_standard.summarize("error_rate")
                lid_wer = wer_lid.summarize("error_rate")
                print(f"[{i+1}/{total}] Standard WER: {std_wer:.2f}% | LID-Aware WER: {lid_wer:.2f}%")
    
    # Final results
    std_wer_final = wer_standard.summarize("error_rate")
    lid_wer_final = wer_lid.summarize("error_rate")
    
    print("\n" + "=" * 60)
    print(f"RESULTS (bias_strength={args.bias_strength})")
    print(f"=" * 60)
    print(f"Standard CTC Decode:   WER = {std_wer_final:.2f}%")
    print(f"LID-Aware CTC Decode:  WER = {lid_wer_final:.2f}%")
    print(f"Improvement:           {std_wer_final - lid_wer_final:.2f}%")
    print(f"=" * 60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
