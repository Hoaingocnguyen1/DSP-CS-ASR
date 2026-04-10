#!/usr/bin/env python3
"""
DSP-CS Whisper Training Recipe
================================
Trains the DSP_Whisper model (Whisper Small + DSP-CS) for Vietnamese-English
code-switching ASR using SpeechBrain's Brain class.

Loss: (1 - lid_w) * seq2seq_CE + lid_w * LID_focal
Decoder: Whisper autoregressive decoder
Evaluation: WER via Whisper beam search
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class DSP_Whisper_ASR(sb.Brain):
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

    def _get_current_lid_weight(self, epoch):
        """Epoch-wise linear schedule for the auxiliary LID loss."""
        start_w = getattr(self.hparams, "lid_loss_weight_start", None)
        end_w = getattr(self.hparams, "lid_loss_weight_end", None)
        if start_w is None or end_w is None:
            return self.hparams.lid_loss_weight

        start_epoch = getattr(self.hparams, "lid_loss_weight_decay_start_epoch", 1)
        end_epoch = getattr(
            self.hparams,
            "lid_loss_weight_decay_end_epoch",
            self.hparams.number_of_epochs,
        )

        if epoch <= start_epoch:
            return start_w
        if epoch >= end_epoch:
            return end_w

        progress = (epoch - start_epoch) / max(end_epoch - start_epoch, 1)
        return start_w + progress * (end_w - start_w)

    def _write_lid_confusion(self, stage, epoch):
        """Persist confusion matrix for later LID/code-switch analysis."""
        out_dir = Path(self.hparams.output_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"lid_confusion_{stage.name.lower()}_epoch{epoch}.txt"

        confusion = self.lid_confusion.cpu()
        labels = self.lid_label_names[: self.lid_num_classes]
        lines = [
            f"Stage: {stage.name}",
            f"Epoch: {epoch}",
            f"LID-ACC: {100.0 * self.lid_correct / max(self.lid_total, 1):.4f}",
            f"LID-mF1: {self._summarize_lid_metrics()['LID-mF1']:.4f}",
            "",
            "Labels: " + " ".join(labels),
            "Confusion Matrix (rows=target, cols=pred)",
            "\t" + "\t".join(labels),
        ]

        for i, label in enumerate(labels):
            row = "\t".join(str(int(x)) for x in confusion[i].tolist())
            lines.append(f"{label}\t{row}")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def init_optimizers(self):
        """Separate LRs for DSP backbone vs decoder."""
        # Group 1: LoRA + CausalPromptGenerator + Gate (backbone side)
        backbone_params = []
        # Group 2: Whisper decoder params
        decoder_params = []

        dsp = self.modules.dsp_model
        for name, param in dsp.named_parameters():
            if not param.requires_grad:
                continue
            if "decoder" in name and "prompt" not in name and "gate" not in name:
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
        """Clip and step all optimizer param groups, not only the first one."""
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

        # Get decoder input tokens (shift right for teacher forcing)
        tokens, token_lens = batch.tokens

        # Build decoder input: [bos, lang, task, notimestamps, ...tokens...]
        dsp = self.modules.dsp_model
        whisper = dsp.whisper

        bos_tokens = self._get_decoder_input_tokens(tokens, token_lens)

        # Forward through DSP_Whisper
        logits, lid_logits = dsp(wavs, bos_tokens)
        lid_logprobs = F.log_softmax(lid_logits.clamp(min=-10, max=10), dim=-1)

        predictions = {
            "logits": logits,
            "lid_logprobs": lid_logprobs,
        }

        # Log training mode once
        if not hasattr(self, "_logged_mode"):
            mode = "WARMUP (LID only)" if self.hparams.warmup_only else "JOINT (Seq2Seq + LID)"
            logger.info(f"Training mode: {mode}")
            n_train = sum(p.numel() for p in dsp.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in dsp.parameters())
            logger.info(f"Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.2f}%)")
            self._logged_mode = True

        if not self.hparams.warmup_only and stage != sb.Stage.TRAIN:
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
        """Build Whisper decoder input: prefix tokens + target tokens."""
        dsp = self.modules.dsp_model
        whisper = dsp.whisper
        B = target_tokens.shape[0]
        device = target_tokens.device

        # Whisper prefix: [bos, lang, task, notimestamps]
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

        # Concat prefix + target tokens (teacher forcing input)
        decoder_input = torch.cat([prefix_tensor, padded_targets], dim=1)
        return decoder_input

    def compute_objectives(self, predictions, batch, stage):
        # --- 1. LID Loss (Focal Loss) ---
        lid_logprobs = predictions["lid_logprobs"]  # [B, 1500, 3]
        B, T_enc, C = lid_logprobs.shape
        lid_targets = self._compute_lid_targets(batch, C)

        # Stretch word-level LID targets to encoder frame count (1500)
        lid_targets_float = lid_targets.float().unsqueeze(1)
        lid_targets_interp = F.interpolate(
            lid_targets_float, size=T_enc, mode="nearest"
        ).squeeze(1).long()
        frame_lengths = torch.round(batch.sig[1] * T_enc).long()
        frame_mask = self._compute_frame_mask(frame_lengths, T_enc, lid_logprobs.device)

        # Focal Loss (gamma=2.0)
        lid_targets_flat = lid_targets_interp.masked_select(frame_mask).clamp(0, C - 1)
        logprobs_flat = lid_logprobs[frame_mask]
        target_logprobs = logprobs_flat.gather(1, lid_targets_flat.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - target_logprobs.exp()) ** 2.0
        lid_loss = -(focal_weight * target_logprobs).mean()
        lid_loss = torch.nan_to_num(lid_loss, nan=0.0, posinf=0.0, neginf=0.0)

        if self.hparams.warmup_only:
            return lid_loss

        # --- 2. Seq2Seq Cross-Entropy Loss ---
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        logits = predictions["logits"]  # [B, prefix_len + seq_len, vocab]

        # Predict transcript tokens plus EOS.
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

        # Flatten for cross-entropy
        vocab_size = pred_logits.shape[-1]
        seq_loss = F.cross_entropy(
            pred_logits.reshape(-1, vocab_size),
            seq_targets.reshape(-1),
            ignore_index=-100,
            label_smoothing=getattr(self.hparams, "label_smoothing", 0.0),
        )
        seq_loss = torch.nan_to_num(seq_loss, nan=0.0, posinf=100.0, neginf=0.0)

        # --- Loss combination ---
        current_epoch = getattr(self.hparams.epoch_counter, "current", 1)
        lid_w = self._get_current_lid_weight(current_epoch)
        loss = lid_w * lid_loss + (1.0 - lid_w) * seq_loss

        # Log once
        if not hasattr(self, "_logged_loss"):
            logger.info(
                f"Losses: lid={lid_loss.item():.4f}, seq2seq={seq_loss.item():.4f}, "
                f"lid_w={lid_w:.4f}, total={loss.item():.4f}"
            )
            self._logged_loss = True

        # WER during evaluation
        if stage != sb.Stage.TRAIN and "pred_tokens" in predictions:
            whisper = self.modules.dsp_model.whisper
            predicted_words = []
            for pred in predictions["pred_tokens"]:
                # Decode, skipping special tokens
                text = whisper.tokenizer.decode(pred.tolist(), skip_special_tokens=True)
                predicted_words.append(text.strip().split())

            target_words = [words.split() for words in batch.words]
            self.wer_metric.append(batch.id, predicted_words, target_words)

            # Code-Switch WER
            def filter_cs(words):
                return [w for w in words if w.isascii() and w.isalpha() and len(w) > 2]
            pred_cs = [filter_cs(pw) for pw in predicted_words]
            tgt_cs = [filter_cs(tw) for tw in target_words]
            self.cs_wer_metric.append(batch.id, pred_cs, tgt_cs)

            lid_pred = lid_logprobs.argmax(dim=-1)
            self._update_lid_metrics(lid_pred, lid_targets, batch.sig[1])

        return loss

    def _update_lid_metrics(self, lid_pred, lid_targets, wav_lens):
        """Accumulate frame-level LID accuracy and confusion matrix."""
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
        if hasattr(self.modules.dsp_model, "warmup"):
            self.modules.dsp_model.warmup = self.hparams.warmup_only

        if stage != sb.Stage.TRAIN:
            self.lid_num_classes = self.modules.dsp_model.prompt_generator.lid_head.out_features
            self.lid_correct = 0
            self.lid_total = 0
            self.lid_confusion = torch.zeros(
                self.lid_num_classes, self.lid_num_classes, dtype=torch.long
            )
            self.wer_metric = self.hparams.error_rate_computer()
            self.cs_wer_metric = self.hparams.error_rate_computer()

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
        elif not self.hparams.warmup_only:
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["cs-WER"] = self.cs_wer_metric.summarize("error_rate")
            stage_stats["lid_w"] = self._get_current_lid_weight(epoch)
            stage_stats.update(self._summarize_lid_metrics())

        if stage == sb.Stage.VALID:
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            # Early stopping check
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
            if not self.hparams.warmup_only:
                self._write_lid_confusion(stage, epoch)

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if not self.hparams.warmup_only:
                self._write_lid_confusion(stage, self.hparams.epoch_counter.current)


def dataio_prepare(hparams):
    """Prepare data pipelines for Whisper training."""

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
        # Use Whisper tokenizer
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

    asr_brain = DSP_Whisper_ASR(
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
