#!/usr/bin/env python3
"""
AG Baseline Training Recipe (Aditya et al., ICASSP 2024)
=========================================================
Paper : "Attention-Guided Adaptation for Code-Switching Speech Recognition"
Authors: Aditya Yadavalli, Gowtham Premananth, Lu Yeheng,
         Yuchen Hu, Eng Siong Chng
Ref   : arXiv:2312.08856v2

Architecture:
  Audio → [Whisper Encoder + LoRA (φ_e)] → encoder_out [B, 1500, 768]
               ↓
         [Whisper Decoder + Serial Adapter (φ_d)] → output
               ↓                    ↓
          logits [B, T, V]    self-attention maps [12 layers × H heads]
                                    ↓
                             Head Selection:
                             Chọn top K heads có LID token attention pattern
                             (approx: last 4 layers, all heads)
                                    ↓
                             AG Loss: L_AG = Σ||A_selected - G(y)||²
                             G(y) = ground-truth code-switching attention map
                                    ↓
                             L = L_CE + γ · L_AG     (γ = 0.01)

  Bilingual prompt: <|sot|><|vi|><|en|><|transcribe|><|notimestamps|>
  (Paper gốc dùng <|zh|><|en|>, adapt sang <|vi|><|en|> cho ViMedCSS)

2-Stage Training (theo paper):
  ┌──────────┬────────────────────────────┬────────────────────┐
  │ Stage    │ Trainable                  │ Loss               │
  ├──────────┼────────────────────────────┼────────────────────┤
  │ Stage 1  │ Encoder LoRA only          │ L_CE               │
  │ (8 ep)   │ Dec adapters FROZEN        │                    │
  ├──────────┼────────────────────────────┼────────────────────┤
  │ Stage 2  │ Encoder LoRA + Dec adapter │ L_CE + 0.01·L_AG   │
  │ (rest)   │ All UNFROZEN               │                    │
  └──────────┴────────────────────────────┴────────────────────┘

Key differences from DSP-CS-ASR V5:
  ┌────────────────────┬──────────────────┬──────────────────────┐
  │ Component          │ AG (this file)   │ DSP V5 (eLAL)        │
  ├────────────────────┼──────────────────┼──────────────────────┤
  │ LID supervision    │ AG Loss (MSE)    │ eLAL (Focal CE)      │
  │ Attention target   │ Self-attention   │ Cross-attention       │
  │ Prompt             │ <|vi|><|en|>     │ <|vi|> only           │
  │ Decoder adapter    │ Serial (Houlsby) │ No decoder adapter    │
  │ Training           │ 2-stage          │ 1-stage joint         │
  │ Encoder adapter    │ LoRA             │ LoRA + Soft-Route LAA │
  └────────────────────┴──────────────────┴──────────────────────┘

Usage (on server):
  # 1. cd vào thư mục AG
  cd /workspace/DSP-CS-ASR/recipes/Baselines/AG

  # 2. Chạy train
  PYTHONPATH=/workspace/DSP-CS-ASR python3 train_ag.py train_ag.yaml \\
      --data_folder /workspace/DSP-CS-ASR/data/vimedcss \\
      --output_folder /workspace/DSP-CS-ASR/results/Baseline_AG/2025 \\
      --save_folder /workspace/DSP-CS-ASR/results/Baseline_AG/2025/save \\
      --train_log /workspace/DSP-CS-ASR/results/Baseline_AG/2025/train_log.txt

  # 3. Chạy trong tmux (nền)
  tmux new -s ag
  # paste lệnh trên, rồi Ctrl+B D để detach

  # 4. Xem log
  tail -f /workspace/DSP-CS-ASR/results/Baseline_AG/2025/train_log.txt
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class AG_ASR(sb.Brain):
    """AG baseline: Whisper + Adapters + 2-stage AG training."""

    def init_optimizers(self):
        """Separate param groups for encoder vs decoder adapters."""
        encoder_params = []
        decoder_params = []

        dsp = self.modules.dsp_model
        for name, param in dsp.named_parameters():
            if not param.requires_grad:
                continue
            if "dec_adapter" in name:
                decoder_params.append(param)
            elif "decoder" in name:
                decoder_params.append(param)
            else:
                encoder_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": self.hparams.lr_backbone},
            {"params": decoder_params, "lr": self.hparams.lr_decoder},
        ], weight_decay=self.hparams.weight_decay)

        self._base_lrs = [self.hparams.lr_backbone, self.hparams.lr_decoder]
        self.optimizers_dict = {"opt_class": self.optimizer}
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

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

    def _is_stage2(self, epoch):
        """Check if we're in Stage 2 (AG Loss active)."""
        stage1_epochs = getattr(self.hparams, "stage1_epochs", 8)
        return epoch is not None and epoch > stage1_epochs

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        if stage == sb.Stage.TEST:
            self.total_audio_duration += (wavs.shape[1] * wav_lens).sum().item() / 16000.0

        tokens, token_lens = batch.tokens
        dsp = self.modules.dsp_model
        bos_tokens = self._get_decoder_input_tokens(tokens, token_lens)

        logits, self_attns = dsp(wavs, bos_tokens)

        predictions = {
            "logits": logits,
            "self_attns": self_attns,
        }

        if not hasattr(self, "_logged_mode"):
            n_train = sum(p.numel() for p in dsp.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in dsp.parameters())
            logger.info(f"[AG Baseline] Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")
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
        """Build bi-lingual decoder input: [sot, vi, en, transcribe, notimestamps, ...tokens...]"""
        dsp = self.modules.dsp_model
        whisper = dsp.whisper
        B = target_tokens.shape[0]
        device = target_tokens.device

        # Use bilingual prefix from AG_Whisper
        prefix = dsp.bilingual_prefix
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

    def _build_ag_ground_truth(self, batch, seq_len, prefix_len, device):
        """Build ground-truth code-switching attention map G(y).
        
        For each word token, it should attend to its corresponding LID token
        in the bilingual prefix. VI words → attend to <|vi|> (pos 1), 
        EN words → attend to <|en|> (pos 2).
        
        Returns G: [B, seq_len, seq_len] where G[b, i, j] indicates 
        the guidance for token i attending to token j.
        """
        c = getattr(self.hparams, "ag_soft_label", 0.6)
        
        B = len(batch.words)
        G = torch.zeros(B, seq_len, seq_len, device=device)
        
        lid_positions = self.modules.dsp_model.lid_positions  # [1, 2]
        vi_pos, en_pos = lid_positions
        
        lid_targets_padded = batch.lid_ids[0] if isinstance(batch.lid_ids, tuple) else batch.lid_ids
        whisper = self.modules.dsp_model.whisper
        
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

                for t in range(t_len):
                    pos = curr_idx + t
                    if pos >= seq_len:
                        break
                    # Set guidance: attend to correct LID token
                    if w_lid == 1:  # VI
                        G[b, pos, vi_pos] = c
                    elif w_lid == 2:  # EN
                        G[b, pos, en_pos] = c
                        
                curr_idx += t_len
                if curr_idx >= seq_len:
                    break
                    
        return G

    def _compute_ag_loss(self, self_attns, G, prefix_len):
        """Compute Attention-Guided Loss.
        
        AG Loss = Σ ||A_head[:, :, lid_cols] - G[:, :, lid_cols]||²
        
        We use the top 60% of heads that show LID attention pattern.
        For simplicity, we use ALL heads from the last 4 decoder layers
        (approximation of the paper's head selection).
        """
        lid_positions = self.modules.dsp_model.lid_positions
        
        ag_loss = torch.tensor(0.0, device=G.device)
        n_heads = 0
        
        # Use last 4 layers' self-attention (approximation of head selection)
        for layer_idx in range(max(0, len(self_attns) - 4), len(self_attns)):
            attn = self_attns[layer_idx]  # [B, H, tgt, tgt]
            num_heads = attn.shape[1]
            
            for h in range(num_heads):
                attn_head = attn[:, h, :, :]  # [B, tgt, tgt]
                
                # Only compute loss on LID token columns
                for lid_col in lid_positions:
                    if lid_col < attn_head.shape[-1]:
                        pred_col = attn_head[:, prefix_len:, lid_col]  # [B, tgt-prefix]
                        gt_col = G[:, prefix_len:, lid_col]  # [B, tgt-prefix]
                        ag_loss = ag_loss + F.mse_loss(pred_col, gt_col)
                        n_heads += 1
        
        if n_heads > 0:
            ag_loss = ag_loss / n_heads
            
        return ag_loss

    def compute_objectives(self, predictions, batch, stage):
        # --- 1. Seq2Seq Cross-Entropy Loss ---
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        logits = predictions["logits"]

        prefix_len = len(self.modules.dsp_model.bilingual_prefix)
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

        # --- 2. AG Loss (Stage 2 only) ---
        current_epoch = getattr(self.hparams.epoch_counter, "current", 1)
        
        if self._is_stage2(current_epoch) and stage == sb.Stage.TRAIN:
            self_attns = predictions["self_attns"]
            seq_len = logits.shape[1]
            
            G = self._build_ag_ground_truth(batch, seq_len, prefix_len, logits.device)
            ag_loss = self._compute_ag_loss(self_attns, G, prefix_len)
            
            gamma = getattr(self.hparams, "ag_gamma", 0.01)
            loss = seq_loss + gamma * ag_loss
            
            if not hasattr(self, "_logged_ag"):
                logger.info(f"[AG Stage 2] ag_loss={ag_loss.item():.6f}, gamma={gamma}")
                self._logged_ag = True
        else:
            loss = seq_loss

        if not hasattr(self, "_logged_loss"):
            stage_name = "Stage 2 (CE+AG)" if self._is_stage2(current_epoch) else "Stage 1 (CE only)"
            logger.info(f"[AG {stage_name}] seq_loss={seq_loss.item():.4f}, total={loss.item():.4f}")
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
            self.cs_wer_metric.append(
                batch.id,
                [filter_cs(pw) for pw in predicted_words],
                [filter_cs(tw) for tw in target_words],
            )

            def filter_native(words):
                return [w for w in words if not (w.isascii() and w.isalpha() and len(w) > 2)]
            self.n_wer_metric.append(
                batch.id,
                [filter_native(pw) for pw in predicted_words],
                [filter_native(tw) for tw in target_words],
            )

            self.cer_metric.append(batch.id, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        # Manage 2-stage training: freeze/unfreeze decoder adapters
        dsp = self.modules.dsp_model
        if stage == sb.Stage.TRAIN:
            if not self._is_stage2(epoch):
                # Stage 1: freeze decoder adapters
                for param in dsp.dec_adapters.parameters():
                    param.requires_grad = False
                if not hasattr(self, "_logged_stage1"):
                    logger.info("[AG] Stage 1: Decoder adapters FROZEN, training encoder only")
                    self._logged_stage1 = True
            else:
                # Stage 2: unfreeze decoder adapters
                for param in dsp.dec_adapters.parameters():
                    param.requires_grad = True
                if not hasattr(self, "_logged_stage2"):
                    logger.info("[AG] Stage 2: Decoder adapters UNFROZEN, AG Loss active")
                    self._logged_stage2 = True

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
            current_epoch = getattr(self.hparams.epoch_counter, "current", 1)
            stage_stats["training_stage"] = 2 if self._is_stage2(current_epoch) else 1

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

    asr_brain = AG_ASR(
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
