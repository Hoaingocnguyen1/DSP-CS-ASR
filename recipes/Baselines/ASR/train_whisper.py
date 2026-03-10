#!/usr/bin/env python3
"""
Baseline Whisper ASR Training Recipe (Seq2Seq)
==============================================
This script is used to fine-tune HuggingFace's Whisper model (e.g. whisper-small)
on the ViMedCSS dataset using its native Seq2Seq objective.

Because Whisper is an Encoder-Decoder model, the training objective and data formatting
differs from the CTC-only `train_ssl_ctc.py` script.
"""

import sys
import torch
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class WhisperASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward pass for Whisper Seq2Seq (using SpeechBrain HuggingFaceWhisper API)."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        # Audio augmentations (training only)
        if stage == sb.Stage.TRAIN \
                and getattr(self.hparams, "enable_env_corrupt", False) \
                and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
        
        # ─── STEP 1: Run the Whisper encoder ────────────────────────────────────
        # SpeechBrain HuggingFaceWhisper.forward() returns encoder hidden states.
        # Signature: forward(wav, wav_lens) -> encoder_output [B, T, D]
        encoder_out, encoder_out_lens = self.modules.whisper.forward_encoder(wavs, wav_lens)
        
        predictions = {}
        
        if stage == sb.Stage.TRAIN:
            # ─── STEP 2a (Train): Teacher-forced decoding ────────────────────────
            # The decoder takes the encoder output + target BOS tokens.
            tokens_bos, _ = batch.tokens_bos
            logits = self.modules.whisper.forward_decoder(tokens_bos, encoder_out)
            p_seq = self.hparams.log_softmax(logits)
            predictions["p_seq"] = p_seq
        else:
            # ─── STEP 2b (Eval): Autoregressive generation ───────────────────────
            # Greedy decoding from encoder output (not from raw wav).
            # Also run teacher forcing to get a valid NLL loss for checkpointing.
            tokens_bos, _ = batch.tokens_bos
            logits = self.modules.whisper.forward_decoder(tokens_bos, encoder_out)
            p_seq = self.hparams.log_softmax(logits)
            predictions["p_seq"] = p_seq
            # Greedy decode: argmax at each step
            predicted_tokens = logits.argmax(dim=-1)   # [B, T]
            predictions["tokens"] = predicted_tokens
        
        return predictions

    def compute_objectives(self, predictions, batch, stage):
        """Computes the Seq2Seq NLL loss for both train and eval."""
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        # NLL loss is computed from teacher-forced logits (available in both stages)
        loss = sb.nnet.losses.nll_loss(
            log_probabilities=predictions["p_seq"],
            targets=tokens_eos,
            length=tokens_eos_lens,
            label_smoothing=self.hparams.label_smoothing,
        )

        # Compute Word Error Rate (WER) during evaluation
        if stage != sb.Stage.TRAIN and "tokens" in predictions:
            # Greedy-decoded token IDs → text
            predicted_words = [
                self.hparams.tokenizer.decode(
                    [t for t in prediction.tolist() if t not in (self.hparams.bos_index, self.hparams.eos_index)],
                    skip_special_tokens=True
                ).strip().split(" ")
                for prediction in predictions["tokens"]
            ]
            target_words = [words.strip().split(" ") for words in batch.words]

            self.wer_metric.append(batch.id, predicted_words, target_words)

            # Code-Switch WER (English terms only, >2 chars)
            def filter_cs_words(words_list):
                return [w for w in words_list if w.isascii() and w.isalpha() and len(w) > 2]

            pred_cs = [filter_cs_words(pw) for pw in predicted_words]
            target_cs = [filter_cs_words(tw) for tw in target_words]
            self.cs_wer_metric.append(batch.id, pred_cs, target_cs)

        return loss

    def on_stage_start(self, stage, epoch):
        """Initialize metrics."""
        if stage != sb.Stage.TRAIN:
            self.wer_metric = self.hparams.error_rate_computer()
            self.cs_wer_metric = self.hparams.error_rate_computer()

    def on_fit_batch_end(self, batch, outputs, loss, should_update):
        """Update the learning rate schedule."""
        if should_update:
            self.hparams.lr_annealing(self.optimizer)

    def on_stage_end(self, stage, stage_loss, epoch):
        """Log stats and save checkpoints."""
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["cs-WER"] = self.cs_wer_metric.summarize("error_rate")

        if stage == sb.Stage.VALID:
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": current_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Test Stage": "Final Evaluation"},
                test_stats=stage_stats,
            )


def dataio_prepare(hparams):
    """Prepares the data IO pipelines."""
    # 1. Audio Pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # 2. Text Pipeline specific to Whisper
    tokenizer = hparams["tokenizer"]
    @sb.utils.data_pipeline.takes("words")
    @sb.utils.data_pipeline.provides("words", "tokens", "tokens_bos", "tokens_eos")
    def text_pipeline(words):
        yield words
        # Whisper tokenizer returns a dict. We want the input_ids.
        # Ensure we don't automatically add special tokens initially so we can control BOS/EOS
        tokens_list = tokenizer.encode(words, add_special_tokens=False)
        yield torch.LongTensor(tokens_list)
        
        # Whisper standard BOS is tokenizer.sot_token_id (often 50257) or similar.
        # SB HuggingFaceWhisper usually uses custom embeddings for multi-language.
        bos = hparams["bos_index"]
        eos = hparams["eos_index"]
        tokens_bos = torch.LongTensor([bos] + tokens_list)
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [eos])
        yield tokens_eos

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
            dynamic_items=[audio_pipeline, text_pipeline],
            output_keys=["id", "sig", "words", "tokens", "tokens_bos", "tokens_eos"],
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

    asr_brain = WhisperASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    asr_brain.fit(
        epoch_counter=asr_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    asr_brain.evaluate(
        test_set=datasets["test"],
        min_key="WER",
        test_loader_kwargs=hparams["valid_dataloader_opts"],
    )
