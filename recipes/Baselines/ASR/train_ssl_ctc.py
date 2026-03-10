#!/usr/bin/env python3
"""
Baseline SSL ASR Training Recipe (CTC Only)
===========================================
This script is used to fine-tune HuggingFace SSL models (Wav2Vec2, HuBERT, WavLM, XLS-R)
on the ViMedCSS dataset using a standard CTC objective.

It removes the proposed multi-task LID head and focuses strictly on Acoustic Modeling 
to serve as a fair baseline comparison for the thesis.
"""

import sys
import torch
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class BaselineASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward pass for CTC-only ASR."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        # Audio augmentations (training only)
        if stage == sb.Stage.TRAIN \
                and getattr(self.hparams, "enable_env_corrupt", False) \
                and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
        
        # 1. Forward pass through the HF SSL backbone
        # We extract the features from the CNN/Transformer stack
        features = self.modules.wav2vec2(wavs, wav_lens)

        # SpecAugment applied on features (after backbone, before CTC projection)
        # More standard for SSL fine-tuning than waveform augmentation.
        if stage == sb.Stage.TRAIN and getattr(self.hparams, "enable_spec_augment", False):
            features = self.modules.spec_augment(features, wav_lens)
        
        # 2. Linear projection to vocab size for CTC
        logits = self.modules.ctc_lin(features)
        p_ctc = self.hparams.log_softmax(logits)
        
        predictions = {"p_ctc": p_ctc}

        # 3. Decode during evaluation
        if stage != sb.Stage.TRAIN:
            # We use CTC greedy decoding for the baseline for simplicity and speed.
            # If using an external LM, ctc_search can be swapped here.
            p_ctc_det = p_ctc.detach()
            sequence, _, _, _ = self.hparams.valid_search(p_ctc_det, wav_lens)
            predictions["tokens"] = sequence

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        """Computes the CTC loss."""
        # Get target tokens
        tokens, tokens_lens = batch.tokens
        
        # CTC Loss calculation
        loss = sb.nnet.losses.ctc_loss(
            log_probs=predictions["p_ctc"],
            targets=tokens,
            input_lens=batch.sig[1],
            target_lens=tokens_lens,
            blank_index=self.hparams.blank_index,
        )

        # Compute Word Error Rate (WER) during evaluation
        if stage != sb.Stage.TRAIN and "tokens" in predictions:
            predicted_words = [
                self.hparams.tokenizer.decode_ids(prediction).split(" ")
                for prediction in predictions["tokens"]
            ]
            target_words = [words.split(" ") for words in batch.words]
            
            self.wer_metric.append(batch.id, predicted_words, target_words)
            
            # Code-Switch WER (English terms only)
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
            # Save checkpoints keyed by WER
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

    # 2. Text Pipeline
    @sb.utils.data_pipeline.takes("words")
    @sb.utils.data_pipeline.provides("words", "tokens")
    def text_pipeline(words):
        yield words
        tokens_list = hparams["tokenizer"].encode_as_ids(words)
        yield torch.LongTensor(tokens_list)

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
            output_keys=["id", "sig", "words", "tokens"],
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

    asr_brain = BaselineASR(
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
