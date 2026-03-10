#!/usr/bin/env python3
import sys
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)

class CodeSwitchASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """
        Runs all the computation of the Seq2Seq ASR and LID prediction.
        """
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        # Audio augmentations (training only)
        # Fix: hparams accessed as dict-like attributes in sb.Brain, not with .get()
        # Use getattr with default False for safe access.
        if stage == sb.Stage.TRAIN \
                and getattr(self.hparams, "enable_env_corrupt", False) \
                and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
        
        # 1. Forward pass through DSP_W2VBERT (Hybrid LoRA + Acoustic Prompt Injection)
        # DSP_W2VBERT.forward() returns 3 values:
        #   adapted_features: [B, T, 1024] Final features (LoRA + Prompt injected)
        #   lid_logits:       [B, T, 3]    Per-frame language predictions
        #   _:                [1, B, 256]  GRU hidden state (discarded in training)
        encoded_signal, lid_logits, _ = self.modules.dsp_w2vbert(wavs, wav_lens)
        lid_logprobs = self.hparams.log_softmax(lid_logits)

        predictions = {"lid_logprobs": lid_logprobs, "encoded_signal": encoded_signal}

        # BUG FIX (Perf): In Warmup Stage 1, skip the decoder entirely to save VRAM.
        # Only run the expensive attention decoder during joint training (Stage 2).
        if not self.hparams.warmup_only:
            tokens_bos, _ = batch.tokens_bos
            embedded_tokens = self.modules.embedding(tokens_bos)
            
            decoder_outputs, _ = self.modules.decoder(
                embedded_tokens, encoded_signal, wav_lens
            )
            
            logits = self.modules.seq_lin(decoder_outputs)
            seq_logprobs = self.hparams.log_softmax(logits)
            predictions["seq_logprobs"] = seq_logprobs

        # Decoding in valid/test — only run beam-search when NOT in warmup mode.
        # In warmup_only, the decoder never ran so `encoded_signal` is the only output.
        if stage != sb.Stage.TRAIN and not self.hparams.warmup_only:
            if stage == sb.Stage.VALID:
                hyps, _, _, _ = self.hparams.valid_search(encoded_signal, wav_lens)
            else:
                hyps, _, _, _ = self.hparams.test_search(encoded_signal, wav_lens)
            predictions["tokens"] = hyps

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        # 1. LID Loss (Frame-synchronous NLLLoss)
        # BUG FIX #2: SpeechBrain PaddedBatch returns PaddedData objects.
        # Call .data to extract the underlying padded Tensor before operating on it.
        lid_targets_padded, lid_lens = batch.lid_ids
        lid_targets = lid_targets_padded.data if hasattr(lid_targets_padded, "data") else lid_targets_padded
        
        # Use F.interpolate to stretch the word-level LID targets to match 
        # the backbone's output frame count (e.g., 320x sub-sampling for w2v-bert).
        B, T_backbone, num_classes = predictions["lid_logprobs"].shape
        
        lid_targets_float = lid_targets.float().unsqueeze(1)  # [B, 1, N_words]
        lid_targets_interp = F.interpolate(
            lid_targets_float,
            size=T_backbone,
            mode="nearest"
        ).squeeze(1).long()  # [B, T_backbone]
        
        # Mask loss with wav_lens to exclude padding frames
        lid_loss = sb.nnet.losses.nll_loss(
            log_probabilities=predictions["lid_logprobs"],
            targets=lid_targets_interp,
            length=batch.sig[1]
        )

        if self.hparams.warmup_only:
            return lid_loss
            
        # 2. ASR Seq2Seq Loss (Stage 2 joint training)
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        asr_loss = sb.nnet.losses.nll_loss(
            log_probabilities=predictions["seq_logprobs"],
            targets=tokens_eos,
            length=tokens_eos_lens,
            label_smoothing=self.hparams.label_smoothing,
        )

        # 3. CTC Loss on adapted features (trains ctc_lin, used for ONNX export)
        # adapted_features → ctc_lin → ctc_log_probs [B, T, vocab]
        # Medium Fix #4: ctc_lin (1024→vocab) must be trained to produce meaningful
        # logits in the exported ONNX graph. Without this loss, ctc_lin is random.
        ctc_logprobs = self.hparams.log_softmax(
            self.modules.ctc_lin(predictions["encoded_signal"])
        )
        ctc_loss = sb.nnet.losses.ctc_loss(
            log_probs=ctc_logprobs,
            targets=tokens_eos,
            # REVIEW FIX: input_lens must be fractional (0-1) for SpeechBrain ctc_loss.
            # batch.sig[1] is already fractional relative length — no scaling needed here.
            # SpeechBrain's ctc_loss internally does: (input_lens * T).round().int()
            input_lens=batch.sig[1],
            target_lens=tokens_eos_lens,
            blank_index=self.hparams.blank_index,
        )

        # Joint Training Loss: L = L_asr + w_lid * L_lid + w_ctc * L_ctc
        loss = asr_loss + self.hparams.lid_loss_weight * lid_loss + 0.3 * ctc_loss

        if stage != sb.Stage.TRAIN:
            # Skip WER metrics in warmup_only (decoder/tokens not available)
            if not self.hparams.warmup_only and "tokens" in predictions:
                # Overall WER
                predicted_words = [
                    self.hparams.tokenizer.decode_ids(prediction).split(" ")
                    for prediction in predictions["tokens"]
                ]
                target_words = [words.split(" ") for words in batch.words]
                self.wer_metric.append(batch.id, predicted_words, target_words)
                
                # cs-WER: Filter to English code-switched words only.
                # Guard: len > 2 prevents short Vietnamese words without diacritics
                # (e.g., "co", "ban", "toi") from being misclassified as English.
                def filter_cs_words(words_list):
                    return [w for w in words_list if w.isascii() and w.isalpha() and len(w) > 2]
                    
                pred_cs = [filter_cs_words(pw) for pw in predicted_words]
                target_cs = [filter_cs_words(tw) for tw in target_words]
                self.cs_wer_metric.append(batch.id, pred_cs, target_cs)

        return loss

    def on_stage_start(self, stage, epoch):
        """Initialize metrics only when decoder is active (Stage 2).
        In warmup_only (Stage 1), no decoder runs so WER metrics are meaningless.
        """
        if stage != sb.Stage.TRAIN and not self.hparams.warmup_only:
            self.wer_metric = self.hparams.error_rate_computer()
            self.cs_wer_metric = self.hparams.error_rate_computer()

    def on_fit_batch_end(self, batch, outputs, loss, should_update):
        """Medium Fix #4: WarmAndExpDecayLRSchedule is step-based (per optimizer update).
        This hook calls the scheduler every time the optimizer steps.
        """
        if should_update:
            self.hparams.lr_annealing(self.optimizer)

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        elif not self.hparams.warmup_only:
            # Only report WER metrics when decoder is active (Stage 2)
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["cs-WER"] = self.cs_wer_metric.summarize("error_rate")

        if stage == sb.Stage.VALID:
            # Step-based scheduler: LR already updated per batch via on_fit_batch_end.
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            # REVIEW FIX: In warmup_only mode, no WER is computed (decoder never ran).
            # Save checkpoints keyed by 'loss' in Stage 1 and 'WER' in Stage 2.
            if not self.hparams.warmup_only:
                self.checkpointer.save_and_keep_only(
                    meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
                )
            else:
                self.checkpointer.save_and_keep_only(
                    meta={"loss": stage_loss}, min_keys=["loss"],
                )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Test Stage": "Final Evaluation"},
                test_stats=stage_stats,
            )


def dataio_prepare(hparams):
    # 1. Audio Pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # 2. Clean Words Pipeline (for ASR)
    @sb.utils.data_pipeline.takes("words")
    @sb.utils.data_pipeline.provides("words", "tokens_list", "tokens_bos", "tokens_eos", "tokens")
    def text_pipeline(words):
        yield words
        tokens_list = hparams["tokenizer"].encode_as_ids(words)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    # 3. LID Pipeline (for Prompt Generator Warmup)
    # Ex: lid_tokens = "VI VI EN EN VI" -> convert to IDs 0, 1, 2
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
            output_keys=["id", "sig", "words", "tokens_bos", "tokens_eos", "tokens", "lid_ids"],
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

    asr_brain = CodeSwitchASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
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

    # Final evaluation on test set
    # REVIEW FIX: In warmup_only mode, no WER is computed, so use min_key="loss".
    # In Stage 2 joint training, use min_key="WER" to load the best checkpoint.
    eval_min_key = "loss" if hparams["warmup_only"] else "WER"
    asr_brain.evaluate(
        datasets["test"],
        min_key=eval_min_key,
        test_loader_kwargs=hparams["valid_dataloader_opts"],
    )
