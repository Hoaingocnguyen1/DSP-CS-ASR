#!/usr/bin/env python3
"""
DSP-CS Whisper Training Recipe
================================
Trains the DSP_Whisper model (Whisper Small + DSP-CS) for Vietnamese-English
code-switching ASR using SpeechBrain's Brain class.

Loss: (1 - lid_w) * seq2seq_CE + lid_w * LID_focal
Decoder: Whisper autoregressive decoder
Evaluation: WER via greedy decoding
"""

import sys
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class DSP_Whisper_ASR(sb.Brain):
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

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig

        # Get decoder input tokens (shift right for teacher forcing)
        tokens, token_lens = batch.tokens

        # Build decoder input: [bos, lang, task, notimestamps, ...tokens...]
        dsp = self.modules.dsp_model
        whisper = dsp.whisper

        bos_tokens = self._get_decoder_input_tokens(tokens)

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
            # Greedy decode for evaluation
            predicted_tokens = self._greedy_decode(wavs)
            predictions["pred_tokens"] = predicted_tokens

        return predictions

    def _get_decoder_input_tokens(self, target_tokens):
        """Build Whisper decoder input: prefix tokens + target tokens."""
        dsp = self.modules.dsp_model
        whisper = dsp.whisper
        B = target_tokens.shape[0]
        device = target_tokens.device

        # Whisper prefix: [bos, lang, task, notimestamps]
        prefix = whisper.tokenizer.prefix_tokens
        prefix_tensor = torch.tensor(prefix, device=device).unsqueeze(0).expand(B, -1)

        # Concat prefix + target tokens (teacher forcing input)
        decoder_input = torch.cat([prefix_tensor, target_tokens], dim=1)
        return decoder_input

    @torch.no_grad()
    def _greedy_decode(self, wavs, max_len=224):
        """Simple greedy decode for evaluation WER."""
        dsp = self.modules.dsp_model
        whisper = dsp.whisper
        B = wavs.shape[0]
        device = wavs.device

        # Get encoder output with DSP injection
        encoder_out, _ = dsp.get_encoder_out(wavs)

        # Start with prefix tokens
        prefix = whisper.tokenizer.prefix_tokens
        generated = torch.tensor(prefix, device=device).unsqueeze(0).expand(B, -1)

        eos_id = whisper.eos
        past_kv = None

        for step in range(max_len):
            logits, _, past_kv = dsp.decode_step(
                encoder_out, generated, past_key_values=past_kv,
            )
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences produced EOS
            if (next_token.squeeze(-1) == eos_id).all():
                break

        return generated

    def compute_objectives(self, predictions, batch, stage):
        # --- 1. LID Loss (Focal Loss) ---
        lid_targets_padded, lid_lens = batch.lid_ids
        lid_targets = lid_targets_padded.data if hasattr(lid_targets_padded, "data") else lid_targets_padded

        lid_logprobs = predictions["lid_logprobs"]  # [B, 1500, 3]
        B, T_enc, C = lid_logprobs.shape

        # Stretch word-level LID targets to encoder frame count (1500)
        lid_targets_float = lid_targets.float().unsqueeze(1)
        lid_targets_interp = F.interpolate(
            lid_targets_float, size=T_enc, mode="nearest"
        ).squeeze(1).long()

        # Focal Loss (gamma=2.0)
        lid_targets_flat = lid_targets_interp.reshape(-1).clamp(0, C - 1)
        logprobs_flat = lid_logprobs.reshape(-1, C)
        target_logprobs = logprobs_flat.gather(1, lid_targets_flat.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - target_logprobs.exp()) ** 2.0
        lid_loss = -(focal_weight * target_logprobs).mean()
        lid_loss = torch.nan_to_num(lid_loss, nan=0.0, posinf=0.0, neginf=0.0)

        if self.hparams.warmup_only:
            return lid_loss

        # --- 2. Seq2Seq Cross-Entropy Loss ---
        tokens, token_lens = batch.tokens
        logits = predictions["logits"]  # [B, prefix_len + seq_len, vocab]

        # Target: same as input but shifted (predict next token)
        # logits[:, prefix_len-1:-1, :] should predict tokens
        prefix_len = len(self.modules.dsp_model.whisper.tokenizer.prefix_tokens)
        # Prediction starts after the last prefix token
        pred_logits = logits[:, prefix_len - 1:-1, :]  # [B, seq_len, vocab]

        # Flatten for cross-entropy
        vocab_size = pred_logits.shape[-1]
        seq_loss = F.cross_entropy(
            pred_logits.reshape(-1, vocab_size),
            tokens.reshape(-1),
            ignore_index=-100,
        )
        seq_loss = torch.nan_to_num(seq_loss, nan=0.0, posinf=100.0, neginf=0.0)

        # --- Loss combination ---
        lid_w = self.hparams.lid_loss_weight
        loss = lid_w * lid_loss + (1.0 - lid_w) * seq_loss

        # Log once
        if not hasattr(self, "_logged_loss"):
            logger.info(f"Losses: lid={lid_loss.item():.4f}, seq2seq={seq_loss.item():.4f}, total={loss.item():.4f}")
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

        return loss

    def on_stage_start(self, stage, epoch):
        if hasattr(self.modules.dsp_model, "warmup"):
            self.modules.dsp_model.warmup = self.hparams.warmup_only

        if stage != sb.Stage.TRAIN:
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

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


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
    @sb.utils.data_pipeline.provides("words", "tokens")
    def text_pipeline(words):
        yield words
        # Use Whisper tokenizer
        whisper = hparams["dsp_model"].whisper
        token_ids = whisper.tokenizer.encode(words, add_special_tokens=False)
        yield torch.LongTensor(token_ids)

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
            output_keys=["id", "sig", "words", "tokens", "lid_ids"],
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
