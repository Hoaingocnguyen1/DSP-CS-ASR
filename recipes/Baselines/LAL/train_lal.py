#!/usr/bin/env python3
"""
LAL Baseline Training Recipe (Liu et al., 2024)
=================================================
Paper : "Aligning Speech to Languages to Enhance Code-switching
         Speech Recognition"
Authors: Hexin Liu, Leibny Paola Garcia, Xiangyu Zhang, Andy W.H. Khong,
         Eng Siong Chng
Ref   : arXiv:2403.05887v3

Architecture:
  Audio → [Whisper Encoder + LoRA] → encoder_out [B, 1500, 768]
               ↓
         nn.Linear(768, 3) → lid_logits (frame-level LID prediction)
               ↓
         Cross-Attention Alignment → pseudo language labels per frame
               ↓
         L_LAL = CrossEntropy(lid_logits, pseudo_labels)
               ↓
         [Whisper Decoder (frozen attn, fine-tune)] → logits
               ↓
         Total Loss: L = (1 - β) · L_seq + β · L_LAL     (β = 0.05 fixed)

Key differences from DSP-CS-ASR V5 (eLAL):
  ┌────────────────────┬──────────────────┬──────────────────────┐
  │ Component          │ LAL (this file)  │ DSP V5 (eLAL)        │
  ├────────────────────┼──────────────────┼──────────────────────┤
  │ LID module         │ nn.Linear        │ CausalPromptGen(GRU) │
  │ LID loss           │ CrossEntropy     │ Focal Loss + Smooth  │
  │ Confidence mask    │ ✗ No             │ ✓ threshold > 0.05   │
  │ Language adapters  │ ✗ No             │ ✓ Soft-Routed LAA    │
  │ β scheduling       │ Fixed β=0.05     │ Linear decay 0.05→01 │
  └────────────────────┴──────────────────┴──────────────────────┘

Usage (on server):
  # 1. cd vào thư mục LAL
  cd /workspace/DSP-CS-ASR/recipes/Baselines/LAL

  # 2. Chạy train
  PYTHONPATH=/workspace/DSP-CS-ASR python3 train_lal.py train_lal.yaml \\
      --data_folder /workspace/DSP-CS-ASR/data/vimedcss \\
      --output_folder /workspace/DSP-CS-ASR/results/Baseline_LAL/2025 \\
      --save_folder /workspace/DSP-CS-ASR/results/Baseline_LAL/2025/save \\
      --train_log /workspace/DSP-CS-ASR/results/Baseline_LAL/2025/train_log.txt

  # 3. Chạy trong tmux (nền)
  tmux new -s lal
  # paste lệnh trên, rồi Ctrl+B D để detach

  # 4. Xem log
  tail -f /workspace/DSP-CS-ASR/results/Baseline_LAL/2025/train_log.txt
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class LAL_ASR(sb.Brain):
    """LAL baseline: Whisper + LoRA + Linear LID + CrossEntropy."""
    lid_label_names = ("SIL", "VI", "EN")

    @staticmethod
    def _compute_lid_targets(batch, num_classes):
        """Interpolate word-level LID targets to encoder frame resolution."""
        lid_targets_padded, lid_lens = batch.lid_ids
        lid_targets = (
            lid_targets_padded.data
            if hasattr(lid_targets_padded, "data")
            else lid_targets_padded
        )
        target_lengths = torch.round(lid_lens * lid_targets.shape[1]).long()
        target_mask = (
            torch.arange(lid_targets.shape[1], device=lid_targets.device)
            .unsqueeze(0)
            .expand(lid_targets.shape[0], -1)
        ) < target_lengths.unsqueeze(1)
        lid_targets = lid_targets.masked_fill(~target_mask, 0)
        return lid_targets

    @staticmethod
    def _compute_frame_mask(lengths, max_len, device):
        return (
            torch.arange(max_len, device=device)
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
        ) < lengths.unsqueeze(1)

    def init_optimizers(self):
        """Separate LRs for backbone vs decoder."""
        backbone_params = []
        decoder_params = []

        dsp = self.modules.dsp_model
        for name, param in dsp.named_parameters():
            if not param.requires_grad:
                continue
            if "decoder" in name and "lid" not in name:
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

    def optimizers_step(self):
        """Clip and step all optimizer param groups."""
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
                torch.nn.utils.clip_grad_norm_(
                    group["params"], self.max_grad_norm
                )

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

        logits, lid_logits, cross_attn = dsp(wavs, bos_tokens)

        predictions = {
            "logits": logits,
            "lid_logits": lid_logits,
            "cross_attn": cross_attn,
        }

        if not hasattr(self, "_logged_mode"):
            n_train = sum(p.numel() for p in dsp.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in dsp.parameters())
            logger.info(f"[LAL Baseline] Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")
            self._logged_mode = True

        if stage != sb.Stage.TRAIN:
            encoder_out, _ = dsp.get_encoder_out(wavs)
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

        decoder_input = torch.cat([prefix_tensor, padded_targets], dim=1)
        return decoder_input

    def _expand_word_lid_to_token_lid(self, batch, prefix_len, device):
        """Map word-level LID to BPE-token-level LID (same as V5)."""
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
        # --- 1. LAL: Language Alignment Loss (Paper 4 exact) ---
        lid_logits = predictions["lid_logits"]  # [B, T_enc, C]
        cross_attn = predictions.get("cross_attn", None)

        B, T_enc, C = lid_logits.shape
        lid_targets = self._compute_lid_targets(batch, C)

        frame_lengths = torch.round(batch.sig[1] * T_enc).long()
        frame_mask = self._compute_frame_mask(frame_lengths, T_enc, lid_logits.device)

        if cross_attn is not None and stage == sb.Stage.TRAIN:
            whisper = self.modules.dsp_model.whisper
            prefix_len = len(whisper.tokenizer.prefix_tokens)

            # A. Word to BPE mapping (same as V5)
            token_lid = self._expand_word_lid_to_token_lid(batch, prefix_len, lid_logits.device)

            # B. Frame-to-Token Alignment via Cross-Attention
            attn_avg = cross_attn.mean(dim=1)  # [B, tgt_len, src_len]
            _, frame_to_token = attn_avg.max(dim=1)  # [B, src_len]

            # C. NO confidence masking (Paper 4 exact)
            # D. Pseudo labels
            pseudo_labels = token_lid.gather(1, frame_to_token.clamp(0, token_lid.size(1) - 1))

            pseudo_flat = pseudo_labels.masked_select(frame_mask).clamp(0, C - 1)
            logits_flat = lid_logits[frame_mask]

            # E. Weighted LAL (Paper 4 exact for class imbalance)
            # w_c ∝ 1 / Count(lang_c)
            if pseudo_flat.numel() > 0:
                counts = torch.bincount(pseudo_flat, minlength=C).float().clamp_min(1.0)
                inv_freq = counts.sum() / (C * counts)
                weight_tensor = inv_freq / inv_freq.sum()
                weight_tensor = weight_tensor.clamp(min=0.01) # Tránh weight bằng 0
            else:
                weight_tensor = torch.ones(C, device=logits_flat.device) / C

            lid_loss = F.cross_entropy(logits_flat, pseudo_flat, weight=weight_tensor)
            lid_targets_interp = pseudo_labels
        else:
            # Valid/Test fallback
            lid_targets_float = lid_targets.float().unsqueeze(1)
            lid_targets_interp = F.interpolate(
                lid_targets_float, size=T_enc, mode="nearest"
            ).squeeze(1).long()

            lid_targets_flat = lid_targets_interp.masked_select(frame_mask).clamp(0, C - 1)
            logits_flat = lid_logits[frame_mask]
            lid_loss = F.cross_entropy(logits_flat, lid_targets_flat)

        lid_loss = torch.nan_to_num(lid_loss, nan=0.0, posinf=0.0, neginf=0.0)

        # --- 2. Seq2Seq Cross-Entropy Loss ---
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

        # --- Fixed β combination (Paper 4 exact) ---
        beta = getattr(self.hparams, "lal_beta", 0.05)
        loss = (1.0 - beta) * seq_loss + beta * lid_loss

        if not hasattr(self, "_logged_loss"):
            logger.info(
                f"[LAL] Losses: lid={lid_loss.item():.4f}, seq2seq={seq_loss.item():.4f}, "
                f"beta={beta}, total={loss.item():.4f}"
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

            lid_pred = lid_logits.argmax(dim=-1)
            self._update_lid_metrics(lid_pred, lid_targets_interp, batch.sig[1])

        return loss

    def _update_lid_metrics(self, lid_pred, lid_targets, wav_lens):
        frame_lengths = torch.round(wav_lens * lid_pred.shape[1]).long()
        frame_mask = (
            torch.arange(lid_pred.shape[1], device=lid_pred.device)
            .unsqueeze(0)
            .expand(lid_pred.shape[0], -1)
        ) < frame_lengths.unsqueeze(1)

        pred_flat = lid_pred.masked_select(frame_mask)
        target_flat = lid_targets.masked_select(frame_mask)

        self.lid_correct += (pred_flat == target_flat).sum().item()
        self.lid_total += target_flat.numel()

        indices = target_flat * self.lid_num_classes + pred_flat
        self.lid_confusion += torch.bincount(
            indices,
            minlength=self.lid_num_classes * self.lid_num_classes,
        ).reshape(self.lid_num_classes, self.lid_num_classes).cpu()

    def _summarize_lid_metrics(self):
        if self.lid_total == 0:
            return {"LID-ACC": 0.0, "LID-mF1": 0.0}
        confusion = self.lid_confusion.float()
        tp = confusion.diag()
        precision = tp / confusion.sum(dim=0).clamp_min(1.0)
        recall = tp / confusion.sum(dim=1).clamp_min(1.0)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
        return {
            "LID-ACC": 100.0 * self.lid_correct / self.lid_total,
            "LID-mF1": 100.0 * f1.mean().item(),
        }

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.lid_num_classes = self.modules.dsp_model.lid_head.out_features
            self.lid_correct = 0
            self.lid_total = 0
            self.lid_confusion = torch.zeros(
                self.lid_num_classes, self.lid_num_classes, dtype=torch.long
            )
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
            stage_stats["lal_beta"] = getattr(self.hparams, "lal_beta", 0.05)
            stage_stats.update(self._summarize_lid_metrics())

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
                    logger.info(
                        f"Early stopping: WER did not improve for {patience} epochs. "
                        f"Best WER: {self._best_wer:.2f}%"
                    )
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

            import time
            import torch as th
            test_end_time = time.time()
            total_decode_time = test_end_time - self.test_start_time if hasattr(self, 'test_start_time') else 0
            rtf = total_decode_time / self.total_audio_duration if getattr(self, "total_audio_duration", 0) > 0 else 0

            vram_gb = 0
            if th.cuda.is_available():
                vram_gb = th.cuda.max_memory_allocated() / (1024**3)

            total_params = sum(p.numel() for p in self.modules.parameters()) / 1e6
            trainable_params = sum(p.numel() for p in self.modules.parameters() if p.requires_grad) / 1e6

            print("=" * 60)
            print("ĐÁNH GIÁ CHUYÊN SÂU (COMPREHENSIVE METRICS - LAL BASELINE)")
            print("=" * 60)
            print("1. TỐC ĐỘ VÀ TÀI NGUYÊN (SPEED & RESOURCES):")
            print(f" - Tổng thời gian giải mã : {total_decode_time:.2f} giây")
            print(f" - Tổng thời lượng Audio  : {getattr(self, 'total_audio_duration', 0):.2f} giây")
            print(f" - Tốc độ giải mã (RTF)   : {rtf:.4f} (Càng nhỏ càng tốt)")
            print(f" - Đỉnh VRAM tiêu thụ     : {vram_gb:.2f} GB")
            print(f" - Kích thước Level Model : {total_params:.1f} M params")
            percent_train = (trainable_params / total_params * 100) if total_params > 0 else 0
            print(f" - Tham số huấn luyện     : {trainable_params:.1f} M params ({percent_train:.2f}%)")
            print("-" * 60)

            wer_summ = self.wer_metric.summarize()
            N = wer_summ['num_scored_tokens']
            if N > 0:
                print("2. PHÂN TÍCH LỖI WER TỔNG THỂ (Substitutions/Deletions/Insertions):")
                print(f" - Substitutions (S) : {wer_summ['substitutions']/N*100:.2f}% (Nhận diện sai từ)")
                print(f" - Deletions (D)     : {wer_summ['deletions']/N*100:.2f}% (Bỏ sót từ)")
                print(f" - Insertions (I)    : {wer_summ['insertions']/N*100:.2f}% (Nhận diện thừa từ)")
            print("=" * 60)

            with open(self.hparams.output_folder + "/wer_test_details.txt", "w", encoding="utf-8") as w:
                self.wer_metric.write_stats(w)
            with open(self.hparams.output_folder + "/cer_test_details.txt", "w", encoding="utf-8") as w:
                self.cer_metric.write_stats(w)
            with open(self.hparams.output_folder + "/cswer_test_details.txt", "w", encoding="utf-8") as w:
                self.cs_wer_metric.write_stats(w)
            with open(self.hparams.output_folder + "/nwer_test_details.txt", "w", encoding="utf-8") as w:
                self.n_wer_metric.write_stats(w)

            print(f"Đã lưu chi tiết vào thư mục: {self.hparams.output_folder}")


def dataio_prepare(hparams):
    """Prepare data pipelines (identical to DSP-CS-ASR V5)."""
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

    asr_brain = LAL_ASR(
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
