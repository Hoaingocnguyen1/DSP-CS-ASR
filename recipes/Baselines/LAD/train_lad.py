#!/usr/bin/env python3
"""
LAD Baseline Training Recipe (Zhao et al., ICASSP 2025)
========================================================
Paper : "Adapting Whisper for Code-Switching through Encoding
         Refining and Language-Aware Decoding"
Authors: Jiahui Zhao, Haihua Xu, Eng Siong Chng
Ref   : IEEE ICASSP 2025 (10.1109/ICASSP49660.2025.10889634)

Architecture (Per-layer Dual-path — KHÔNG chạy decoder 2 lần):
  Audio → [Whisper Encoder + LoRA] → encoder_out [B, 1500, 768]
               ↓
         Encoding Refiner (GRU 2-layer bidirectional)
               ↓ (CTC auxiliary loss)
         encoder_out_refined [B, 1500, 768]
               ↓
         [Whisper Decoder — MODIFIED per layer]
           Mỗi decoder layer:
             h + p_vi → SharedSelfAttn → h_vi_sa
             h + p_en → SharedSelfAttn → h_en_sa
             h_vi_sa  → SharedCrossAttn → h_vi_ca → vi_adapter(64) → h_vi
             h_en_sa  → SharedCrossAttn → h_en_ca → en_adapter(64) → h_en
             h_vi     → SharedMLP → h_vi_out
             h_en     → SharedMLP → h_en_out
               ↓
         Fusion (learned sigmoid gate):
           gate = σ(Linear([h_vi; h_en]))    → [w_vi, w_en]
           h_fused = w_vi · h_vi + w_en · h_en
               ↓
         logits → L_att (Seq2Seq CE)
               ↓
         Total Loss: L = α · L_CTC + (1 - α) · L_att     (α = 0.3)

Key differences from DSP-CS-ASR V5:
  ┌────────────────────┬──────────────────┬──────────────────────┐
  │ Component          │ LAD (this file)  │ DSP V5 (eLAL)        │
  ├────────────────────┼──────────────────┼──────────────────────┤
  │ Encoder refiner    │ GRU + CTC        │ No encoder refiner   │
  │ Decoder adapters   │ Per-layer dual   │ No decoder adapter   │
  │ Decoder paths      │ VI + EN fusion   │ Single path          │
  │ Fusion             │ Learned sigmoid  │ N/A                  │
  │ LID supervision    │ CTC (implicit)   │ eLAL (explicit)      │
  │ Encoder adapter    │ LoRA             │ LoRA + Soft-Route LAA│
  │ Language prompt    │ Learnable embed  │ Whisper prefix only   │
  └────────────────────┴──────────────────┴──────────────────────┘

Hyperparameters (from paper):
  - Adapter bottleneck: 64
  - Encoding Refiner: GRU 2-layer, hidden=384 (768/2), bidirectional
  - CTC weight α: 0.3 (paper uses 0.7 for CTC+Attention, we start lower)
  - LR: 3e-4 (backbone), 1.5e-4 (decoder)
  - Epochs: 50 (early stop patience=5)

Usage (on server):
  # 1. cd vào thư mục LAD
  cd /workspace/DSP-CS-ASR/recipes/Baselines/LAD

  # 2. Chạy train
  PYTHONPATH=/workspace/DSP-CS-ASR python3 train_lad.py train_lad.yaml \\
      --data_folder /workspace/DSP-CS-ASR/data/vimedcss \\
      --output_folder /workspace/DSP-CS-ASR/results/Baseline_LAD/2025 \\
      --save_folder /workspace/DSP-CS-ASR/results/Baseline_LAD/2025/save \\
      --train_log /workspace/DSP-CS-ASR/results/Baseline_LAD/2025/train_log.txt

  # 3. Chạy trong tmux (nền)
  tmux new -s lad
  # paste lệnh trên, rồi Ctrl+B D để detach

  # 4. Xem log
  tail -f /workspace/DSP-CS-ASR/results/Baseline_LAD/2025/train_log.txt
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class LAD_ASR(sb.Brain):
    """LAD baseline: Whisper + LoRA + Encoding Refiner + LAD decoder."""

    def init_optimizers(self):
        backbone_params = []
        decoder_params = []

        dsp = self.modules.dsp_model
        for name, param in dsp.named_parameters():
            if not param.requires_grad:
                continue
            if "decoder" in name and "refiner" not in name and "ctc" not in name and "fusion" not in name and "prompt_emb" not in name and "lad_wrappers" not in name:
                decoder_params.append(param)
            else:
                backbone_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": self.hparams.lr_backbone},
            {"params": decoder_params, "lr": self.hparams.lr_decoder},
        ], weight_decay=self.hparams.weight_decay)

        self._base_lrs = [self.hparams.lr_backbone, self.hparams.lr_decoder]
        self.optimizers_dict = {"opt_class": self.optimizer}
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def fit_batch(self, batch):
        """Override fit_batch to support gradient accumulation natively."""
        if not hasattr(self, "accumulation_steps"):
            self.accumulation_steps = getattr(self.hparams, "accumulation_steps", 4)
            self.accumulation_counter = 0

        should_step = ((self.accumulation_counter + 1) % self.accumulation_steps) == 0
        
        # Forward and Loss
        outputs = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(outputs, batch, sb.Stage.TRAIN)
        
        # Scale for accumulation
        loss = loss / self.accumulation_steps
        self.scaler.scale(loss).backward()
        
        if should_step:
            self.optimizers_step()
            
        self.accumulation_counter += 1
        # Return true unscaled loss for stats
        return (loss * self.accumulation_steps).detach().cpu()

    def optimizers_step(self):
        if self.optimizers_dict is not None:
            valid_optimizers = self.freeze_optimizers(self.optimizers_dict)
        elif self.opt_class is not None:
            valid_optimizers = {"optimizer": self.optimizer}
        else:
            return

        for opt in valid_optimizers.values():
            self.scaler.unscale_(opt)
        for opt in valid_optimizers.values():
            for group in opt.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], self.max_grad_norm)
        if not self.scaler.is_enabled() and self.skip_nonfinite_grads:
            self.check_gradients()
        for opt in valid_optimizers.values():
            self.scaler.step(opt)
        self.scaler.update()
        for opt in valid_optimizers.values():
            opt.zero_grad(set_to_none=True)
        self.optimizer_step += 1

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        if stage == sb.Stage.TEST:
            self.total_audio_duration += (wavs.shape[1] * wav_lens).sum().item() / 16000.0

        tokens, token_lens = batch.tokens
        dsp = self.modules.dsp_model
        bos_tokens = self._get_decoder_input_tokens(tokens, token_lens)

        logits, ctc_logits, cross_attn, lid_logits = dsp(wavs, bos_tokens)

        predictions = {
            "logits": logits,
            "ctc_logits": ctc_logits,
            "lid_logits": lid_logits,
        }

        if not hasattr(self, "_logged_mode"):
            n_train = sum(p.numel() for p in dsp.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in dsp.parameters())
            logger.info(f"[LAD Baseline] Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")
            self._logged_mode = True

        if stage != sb.Stage.TRAIN:
            encoder_out = dsp.get_encoder_out(wavs)
            search = (
                self.hparams.valid_search
                if stage == sb.Stage.VALID
                else self.hparams.test_search
            )
            predicted_tokens, _, _, _ = search(encoder_out.detach(), wav_lens)
            predictions["pred_tokens"] = predicted_tokens

        return predictions

    def _get_decoder_input_tokens(self, target_tokens, target_lens):
        dsp = self.modules.dsp_model
        whisper = dsp.whisper
        B = target_tokens.shape[0]
        device = target_tokens.device

        prefix = whisper.tokenizer.prefix_tokens
        prefix_tensor = torch.tensor(prefix, device=device).unsqueeze(0).expand(B, -1)

        target_lengths = torch.round(target_lens * target_tokens.shape[1]).long()
        target_mask = (
            torch.arange(target_tokens.shape[1], device=device)
            .unsqueeze(0)
            .expand(B, -1)
        ) < target_lengths.unsqueeze(1)
        padded_targets = target_tokens.masked_fill(
            ~target_mask, whisper.tokenizer.pad_token_id
        )

        return torch.cat([prefix_tensor, padded_targets], dim=1)

    def _expand_word_lid_to_token_lid(self, batch, prefix_len, device):
        """Map word-level LID to BPE-token-level LID."""
        whisper = self.modules.dsp_model.whisper
        B = len(batch.words)
        
        max_tgt_len = prefix_len + batch.tokens[0].shape[1]
        token_lids = torch.zeros(B, max_tgt_len, dtype=torch.long, device=device)
        
        lid_targets_padded = batch.lid_ids[0] if isinstance(batch.lid_ids, tuple) else batch.lid_ids
        for b in range(B):
            words = batch.words[b].split()
            lids = lid_targets_padded[b]
            
            curr_idx = prefix_len
            for word, w_lid in zip(words, lids):
                if curr_idx > prefix_len:
                    word_input = " " + word
                else:
                    word_input = word
                    
                tokens = whisper.tokenizer.encode(word_input, add_special_tokens=False)
                t_len = len(tokens)
                
                if curr_idx + t_len <= max_tgt_len:
                    token_lids[b, curr_idx : curr_idx + t_len] = w_lid
                    curr_idx += t_len
                else:
                    rem = max_tgt_len - curr_idx
                    if rem > 0:
                        token_lids[b, curr_idx:] = w_lid
                    break
                    
        return token_lids

    def compute_objectives(self, predictions, batch, stage):
        # --- 1. Seq2Seq Cross-Entropy Loss ---
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        logits = predictions["logits"]

        prefix_len = len(self.modules.dsp_model.whisper.tokenizer.prefix_tokens)
        pred_logits = logits[:, prefix_len - 1:, :]

        target_lengths = torch.round(
            tokens_eos_lens * tokens_eos.shape[1]
        ).long()
        target_mask = (
            torch.arange(tokens_eos.shape[1], device=tokens_eos.device)
            .unsqueeze(0)
            .expand(tokens_eos.shape[0], -1)
        ) < target_lengths.unsqueeze(1)
        seq_targets = tokens_eos.masked_fill(~target_mask, -100)

        vocab_size = pred_logits.shape[-1]
        seq_loss = F.cross_entropy(
            pred_logits.reshape(-1, vocab_size),
            seq_targets.reshape(-1),
            ignore_index=-100,
            label_smoothing=getattr(self.hparams, "label_smoothing", 0.0),
        )
        seq_loss = torch.nan_to_num(seq_loss, nan=0.0, posinf=100.0, neginf=0.0)

        # --- 2. CTC Loss (Encoding Refiner auxiliary) ---
        ctc_logits = predictions["ctc_logits"]  # [B, T_enc, vocab]
        tokens, token_lens = batch.tokens

        ctc_log_probs = F.log_softmax(ctc_logits, dim=-1)
        input_lengths = torch.round(batch.sig[1] * ctc_logits.shape[1]).long()
        target_lengths = torch.round(token_lens * tokens.shape[1]).long()

        ctc_loss = F.ctc_loss(
            ctc_log_probs.transpose(0, 1),  # [T, B, vocab]
            tokens,
            input_lengths,
            target_lengths,
            blank=0,
            zero_infinity=True,
        )
        ctc_loss = torch.nan_to_num(ctc_loss, nan=0.0, posinf=0.0, neginf=0.0)

        # --- 3. Token-Level LID Loss (LAD paper specifically uses this) ---
        lid_logits = predictions["lid_logits"]  # [B, seq, num_languages]
        token_lids = self._expand_word_lid_to_token_lid(batch, prefix_len, lid_logits.device)
        
        # We need to align lid_logits to predictions. The LID logic spans over prefix + tokens
        # Target token_lids: [B, seq]
        B, seq_len, num_langs = lid_logits.shape
        # Ensure they match in length, trim appropriately
        min_len = min(seq_len, token_lids.shape[1])
        lid_logits_trimmed = lid_logits[:, :min_len, :]
        token_lids_trimmed = token_lids[:, :min_len]

        # Ignore padding (pad token is typically 0 for LID here, but wait, usually we should use mask)
        # Actually 0 = VI, 1 = EN. NEU might be 2. Let's compute loss over non-pad target tokens.
        # target_mask was calculated above, but only covers the target generation part.
        mask_min_len = min(min_len - prefix_len, target_mask.shape[1])
        full_mask = torch.zeros((B, min_len), dtype=torch.bool, device=lid_logits.device)
        if mask_min_len > 0:
             full_mask[:, prefix_len:prefix_len+mask_min_len] = target_mask[:, :mask_min_len]

        lid_loss = F.cross_entropy(
            lid_logits_trimmed[full_mask].reshape(-1, num_langs),
            token_lids_trimmed[full_mask].reshape(-1),
            ignore_index=-100
        )
        lid_loss = torch.nan_to_num(lid_loss, nan=0.0, posinf=0.0, neginf=0.0)

        # --- Combined: α·L_CTC + β·L_seq + γ·L_LID ---
        alpha = getattr(self.hparams, "ctc_weight", 0.3)
        beta = 1.0 - alpha
        gamma = getattr(self.hparams, "lid_weight", 0.5)  # typically 0.5 for LID auxiliary
        loss = alpha * ctc_loss + beta * seq_loss + gamma * lid_loss

        if not hasattr(self, "_logged_loss"):
            logger.info(
                f"[LAD] Losses: ctc={ctc_loss.item():.4f}, seq={seq_loss.item():.4f}, "
                f"lid={lid_loss.item():.4f}, alpha={alpha}, gamma={gamma}, total={loss.item():.4f}"
            )
            self._logged_loss = True

        # WER during evaluation
        if stage != sb.Stage.TRAIN and "pred_tokens" in predictions:
            whisper = self.modules.dsp_model.whisper
            predicted_words = []
            for pred in predictions["pred_tokens"]:
                pred_list = pred.tolist() if hasattr(pred, "tolist") else pred
                text = whisper.tokenizer.decode(pred_list, skip_special_tokens=True)
                predicted_words.append(text.strip().split())

            target_words = [words.split() for words in batch.words]
            self.wer_metric.append(batch.id, predicted_words, target_words)

            def filter_cs(words):
                return [w for w in words if w.isascii() and w.isalpha() and len(w) > 2]
            pred_cs = [filter_cs(pw) for pw in predicted_words]
            tgt_cs = [filter_cs(tw) for tw in target_words]
            self.cs_wer_metric.append(batch.id, pred_cs, tgt_cs)

            def filter_native(words):
                return [w for w in words if not (w.isascii() and w.isalpha() and len(w) > 2)]
            pred_native = [filter_native(pw) for pw in predicted_words]
            tgt_native = [filter_native(tw) for tw in target_words]
            self.n_wer_metric.append(batch.id, pred_native, tgt_native)

            self.cer_metric.append(batch.id, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.wer_metric = self.hparams.error_rate_computer()
            self.cs_wer_metric = self.hparams.error_rate_computer()
            self.n_wer_metric = self.hparams.error_rate_computer()
            self.cer_metric = sb.utils.metric_stats.ErrorRateStats(split_tokens=True)

        if stage == sb.Stage.TEST:
            import time
            self.test_start_time = time.time()
            self.total_audio_duration = 0.0

    def on_fit_batch_end(self, batch, outputs, loss, should_update):
        if should_update and hasattr(self.hparams, "lr_annealing"):
            scheduler = self.hparams.lr_annealing
            scheduler(self.optimizer)
            scale = scheduler.current_lr
            for i, group in enumerate(self.optimizer.param_groups):
                group["lr"] = self._base_lrs[i] * scale

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["CS-WER"] = self.cs_wer_metric.summarize("error_rate")
            stage_stats["N-WER"] = self.n_wer_metric.summarize("error_rate")

        if stage == sb.Stage.VALID:
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            current_wer = stage_stats.get("WER", stage_loss)
            if not hasattr(self, "_best_wer"):
                self._best_wer = current_wer
                self._patience_counter = 0
            elif current_wer < self._best_wer:
                self._best_wer = current_wer
                self._patience_counter = 0
            else:
                self._patience_counter += 1
                patience = getattr(self.hparams, "early_stop_patience", 5)
                if self._patience_counter >= patience:
                    logger.info(f"Early stopping. Best WER: {self._best_wer:.2f}%")
                    self.hparams.epoch_counter.current = self.hparams.number_of_epochs

            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats.get("WER", stage_loss)},
                min_keys=["WER"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.output_folder + "/wer_test_details.txt", "w", encoding="utf-8") as w:
                self.wer_metric.write_stats(w)


def dataio_prepare(hparams):
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        if len(sig.shape) > 1 and sig.shape[1] > 1:
            sig = torch.mean(sig, dim=1)
        return sig

    @sb.utils.data_pipeline.takes("words")
    @sb.utils.data_pipeline.provides("words", "tokens", "tokens_eos")
    def text_pipeline(words):
        yield words
        whisper = hparams["dsp_model"].whisper
        token_ids = whisper.tokenizer.encode(words, add_special_tokens=False)
        yield torch.LongTensor(token_ids)
        yield torch.LongTensor(token_ids + [whisper.eos])

    lid_map = {"VI": 1, "EN": 2, "SIL": 0}

    @sb.utils.data_pipeline.takes("lid_tokens")
    @sb.utils.data_pipeline.provides("lid_ids")
    def lid_pipeline(lid_tokens):
        tokens = lid_tokens.split(" ")
        ids = [lid_map.get(t, 0) for t in tokens]
        yield torch.LongTensor(ids)

    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"],
    }

    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, text_pipeline, lid_pipeline],
            output_keys=["id", "sig", "words", "tokens", "tokens_eos", "lid_ids"],
        )
    return datasets


if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    datasets = dataio_prepare(hparams)

    asr_brain = LAD_ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.fit(
        asr_brain.hparams.epoch_counter,
        datasets["train"],
        datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    asr_brain.evaluate(
        datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["valid_dataloader_opts"],
    )
