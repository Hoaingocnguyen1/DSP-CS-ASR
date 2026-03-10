#!/usr/bin/env python3
import sys
import torch
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
        
        # Audio augmentations could be applied here
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
        
        # 1. Forward pass through DSP_W2VBERT
        # Returns adapted acoustic features and lid_logits
        encoded_signal, lid_logits = self.modules.dsp_w2vbert(wavs, wav_lens)
        lid_logprobs = self.hparams.log_softmax(lid_logits)

        # 2. Decoder
        # If in warm-up (Stage 1), we might not even care about decoder output
        # But for correctness, we build predictions.
        tokens_bos, _ = batch.tokens_bos
        embedded_tokens = self.modules.embedding(tokens_bos)
        
        decoder_outputs, _ = self.modules.decoder(
            embedded_tokens, encoded_signal, wav_lens
        )
        
        logits = self.modules.seq_lin(decoder_outputs)
        seq_logprobs = self.hparams.log_softmax(logits)
        
        predictions = {
            "seq_logprobs": seq_logprobs,
            "lid_logprobs": lid_logprobs
        }

        # Decoding in valid/test
        if stage != sb.Stage.TRAIN:
            if stage == sb.Stage.VALID:
                hyps, _, _, _ = self.hparams.valid_search(encoded_signal, wav_lens)
            else:
                hyps, _, _, _ = self.hparams.test_search(encoded_signal, wav_lens)
            predictions["tokens"] = hyps

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        # 1. LID Loss (Focal Loss or CrossEntropyLoss)
        # Using FocalLoss for class imbalance between VI/EN/Silence.
        # Here we simplify with NLL loss.
        lid_targets, lid_lens = batch.lid_ids
        
        # Warning: lid_logprobs has time dimension from backbone reduction, 
        # lid_targets has length from words. 
        # Ideally, we need Frame-level alignment, or simply average/pooling.
        # Since the thesis specifies frame-synchronous prompting:
        # ASR backbones heavily sub-sample audio (e.g., 320x for w2v). 
        # For simplicity, we interpolate LID targets to match backbone output length or calculate CTC loss on LID. 
        # Here we use CTC loss for LID tokens alignment!
        lid_loss = self.hparams.ctc_cost(
            predictions["lid_logprobs"], lid_targets, batch.sig[1], lid_lens
        )

        if self.hparams.warmup_only:
            return lid_loss
            
        # 2. ASR Seq2Seq Loss
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        asr_loss = sb.nnet.losses.nll_loss(
            log_probabilities=predictions["seq_logprobs"],
            targets=tokens_eos,
            length=tokens_eos_lens,
            label_smoothing=self.hparams.label_smoothing,
        )

        # Joint Training Loss (Stage 2)
        # Loss = ASR Loss + 0.2 * LID Loss
        loss = asr_loss + self.hparams.lid_loss_weight * lid_loss

        if stage != sb.Stage.TRAIN:
            # Word Error Rate Metrics
            predicted_words = [
                self.hparams.tokenizer.decode_ids(prediction).split(" ")
                for prediction in predictions["tokens"]
            ]
            target_words = [words.split(" ") for words in batch.words]

            self.wer_metric.append(batch.id, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        if stage != sb.Stage.TRAIN:
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
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
