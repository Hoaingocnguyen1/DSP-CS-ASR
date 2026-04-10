#!/usr/bin/env python3
"""
Whisper Small Full-Finetune Baseline
====================================
Baseline recipe for Vietnamese-English code-switching ASR.

- Backbone: openai/whisper-small
- Training: full fine-tuning of encoder + decoder
- Loss: seq2seq cross-entropy
- Evaluation: Whisper beam search
"""

import sys
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class WhisperBaselineASR(sb.Brain):
    def init_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.modules.whisper.parameters(),
            lr=self.hparams.lr_whisper,
            weight_decay=self.hparams.weight_decay,
        )
        self.optimizers_dict = {"optimizer": self.optimizer}
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens, token_lens = batch.tokens

        whisper = self.modules.whisper
        decoder_input_ids = self._get_decoder_input_tokens(tokens, token_lens)
        mel = whisper._get_mel(wavs)
        encoder_out = whisper.forward_encoder(mel)
        logits, _, _ = whisper.forward_decoder(encoder_out, decoder_input_ids)

        predictions = {"logits": logits}

        if stage != sb.Stage.TRAIN:
            search = (
                self.hparams.valid_search
                if stage == sb.Stage.VALID
                else self.hparams.test_search
            )
            hyps, _, _, _ = search(encoder_out.detach(), wav_lens)
            predictions["pred_tokens"] = hyps

        return predictions

    def _get_decoder_input_tokens(self, target_tokens, target_lens):
        whisper = self.modules.whisper
        batch_size = target_tokens.shape[0]
        device = target_tokens.device

        prefix = whisper.tokenizer.prefix_tokens
        prefix_tensor = torch.tensor(prefix, device=device).unsqueeze(0).expand(
            batch_size, -1
        )

        target_lengths = torch.round(target_lens * target_tokens.shape[1]).long()
        target_mask = (
            torch.arange(target_tokens.shape[1], device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        ) < target_lengths.unsqueeze(1)
        padded_targets = target_tokens.masked_fill(
            ~target_mask, whisper.tokenizer.pad_token_id
        )

        return torch.cat([prefix_tensor, padded_targets], dim=1)

    def compute_objectives(self, predictions, batch, stage):
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        logits = predictions["logits"]
        prefix_len = len(self.modules.whisper.tokenizer.prefix_tokens)
        pred_logits = logits[:, prefix_len - 1 :, :]

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
        loss = F.cross_entropy(
            pred_logits.reshape(-1, vocab_size),
            seq_targets.reshape(-1),
            ignore_index=-100,
            label_smoothing=getattr(self.hparams, "label_smoothing", 0.0),
        )
        loss = torch.nan_to_num(loss, nan=0.0, posinf=100.0, neginf=0.0)

        if stage != sb.Stage.TRAIN and "pred_tokens" in predictions:
            whisper = self.modules.whisper
            predicted_words = []
            for pred in predictions["pred_tokens"]:
                text = whisper.tokenizer.decode(
                    pred.tolist(), skip_special_tokens=True
                )
                predicted_words.append(text.strip().split())

            target_words = [words.split() for words in batch.words]
            self.wer_metric.append(batch.id, predicted_words, target_words)

            def filter_cs(words):
                return [
                    w
                    for w in words
                    if w.isascii() and w.isalpha() and len(w) > 2
                ]

            pred_cs = [filter_cs(pw) for pw in predicted_words]
            tgt_cs = [filter_cs(tw) for tw in target_words]
            self.cs_wer_metric.append(batch.id, pred_cs, tgt_cs)

        return loss

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.wer_metric = self.hparams.error_rate_computer()
            self.cs_wer_metric = self.hparams.error_rate_computer()

    def on_fit_batch_end(self, batch, outputs, loss, should_update):
        if should_update and hasattr(self.hparams, "lr_annealing"):
            scheduler = self.hparams.lr_annealing
            scheduler(self.optimizer)
            scale = scheduler.current_lr
            self.optimizer.param_groups[0]["lr"] = self.hparams.lr_whisper * scale

    def on_stage_end(self, stage, stage_loss, epoch):
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
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


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
        whisper = hparams["whisper"]
        token_ids = whisper.tokenizer.encode(words, add_special_tokens=False)
        yield torch.LongTensor(token_ids)
        yield torch.LongTensor(token_ids + [whisper.eos])

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
            output_keys=["id", "sig", "words", "tokens", "tokens_eos"],
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

    asr_brain = WhisperBaselineASR(
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
