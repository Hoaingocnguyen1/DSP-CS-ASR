#!/usr/bin/env python3
"""
DSP-CS Training Recipe (CTC-only)
=================================
This script trains the Dynamic Soft Prompting Code-Switching (DSP-CS) model
using a pure CTC objective, matching the architecture and convergence speed 
of the Baseline XLS-R model for a fair and direct comparison.
"""

import sys
import torch
import torch.nn.functional as F
from hyperpyyaml import load_hyperpyyaml
import speechbrain as sb
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


class DSP_CTC_ASR(sb.Brain):
    def init_optimizers(self):
        """Separate LRs for DSP backbone vs CTC head."""
        self.optimizer = torch.optim.Adam([
            {"params": self.modules.dsp_model.parameters(),
             "lr": self.hparams.lr_backbone},
            {"params": self.modules.ctc_lin.parameters(),
             "lr": self.hparams.lr_ctc},
        ])
        # Store initial LRs so scheduler can apply as scale factor
        self._base_lrs = [self.hparams.lr_backbone, self.hparams.lr_ctc]
        self.optimizers_dict = {"opt_class": self.optimizer}
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)

    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        
        # Audio augmentations (training only)
        if stage == sb.Stage.TRAIN \
                and getattr(self.hparams, "enable_env_corrupt", False) \
                and hasattr(self.hparams, "wav_augment"):
            wavs, wav_lens = self.hparams.wav_augment(wavs, wav_lens)
        
        # 1. Forward through DSP Model (LoRA + LID Prompting + LAL)
        encoded_signal, lid_logits, _, lal_logits = self.modules.dsp_model(wavs, wav_lens)
        lid_logprobs = self.hparams.log_softmax(lid_logits.clamp(min=-10, max=10))
        predictions = {"lid_logprobs": lid_logprobs, "encoded_signal": encoded_signal,
                       "lal_logits": lal_logits}

        # Log once which mode we're in
        if not hasattr(self, "_logged_mode"):
            mode = "WARMUP (LID only)" if self.hparams.warmup_only else "JOINT (CTC + LID)"
            logger.info(f"Training mode: {mode}")
            self._logged_mode = True

        # Stage 1: Warmup only runs the DSP component
        if not self.hparams.warmup_only:
            # SpecAugment applied on features (after backbone, before CTC)
            if stage == sb.Stage.TRAIN and getattr(self.hparams, "enable_spec_augment", False):
                encoded_signal, _ = self.modules.spec_augment(encoded_signal, wav_lens)
            
            # 2. Linear projection to vocab size for CTC
            logits = self.modules.ctc_lin(encoded_signal)
            p_ctc = self.hparams.log_softmax(logits)
            predictions["p_ctc"] = p_ctc
            
            # 3. CTC Greedy Decode during evaluation
            if stage != sb.Stage.TRAIN:
                from speechbrain.decoders.ctc import ctc_greedy_decode
                p_ctc_det = p_ctc.detach()
                sequence = ctc_greedy_decode(
                    p_ctc_det, wav_lens, blank_id=self.hparams.blank_index
                )
                predictions["tokens"] = sequence

        return predictions

    def compute_objectives(self, predictions, batch, stage):
        # --- 1. LID Loss (Frame-synchronous NLLLoss) ---
        lid_targets_padded, lid_lens = batch.lid_ids
        lid_targets = lid_targets_padded.data if hasattr(lid_targets_padded, "data") else lid_targets_padded

        # Stretch word-level LID targets to match backbone output frame count
        B, T_backbone, num_classes = predictions["lid_logprobs"].shape
        lid_targets_float = lid_targets.float().unsqueeze(1)  # [B, 1, N_words]
        lid_targets_interp = F.interpolate(
            lid_targets_float, size=T_backbone, mode="nearest"
        ).squeeze(1).long()  # [B, T_backbone]

        # v2: Focal Loss for LID (γ=2.0)
        # Down-weights easy frames (Vietnamese majority), up-weights hard frames
        # (code-switch boundaries) → LID focuses on what matters for cs-WER
        lid_logprobs = predictions["lid_logprobs"]  # [B, T, 3]
        B_lid, T_lid, C_lid = lid_logprobs.shape
        
        # Gather log-probs for the target class
        lid_targets_flat = lid_targets_interp.reshape(-1).clamp(0, C_lid - 1)
        logprobs_flat = lid_logprobs.reshape(-1, C_lid)
        target_logprobs = logprobs_flat.gather(1, lid_targets_flat.unsqueeze(1)).squeeze(1)
        
        # Focal weight: (1 - p_target)^gamma
        target_probs = target_logprobs.exp()   # p_target
        focal_weight = (1.0 - target_probs) ** 2.0  # gamma=2.0
        
        # Weighted NLL
        lid_loss = -(focal_weight * target_logprobs).mean()
        lid_loss = torch.nan_to_num(lid_loss, nan=0.0, posinf=0.0, neginf=0.0)

        if self.hparams.warmup_only:
            return lid_loss

        # --- 2. CTC ASR Loss (Joint Training) ---
        tokens, tokens_lens = batch.tokens
        
        ctc_loss = sb.nnet.losses.ctc_loss(
            log_probs=predictions["p_ctc"],
            targets=tokens,
            input_lens=batch.sig[1],
            target_lens=tokens_lens,
            blank_index=self.hparams.blank_index,
        )


        # --- Loss Combination with Annealing ---
        freeze_ep = getattr(self.hparams, "freeze_dsp_epochs", 0)
        current_epoch = self.hparams.epoch_counter.current
        if current_epoch <= freeze_ep:
            lid_w = 0.0  # Frozen epochs: pure CTC
        elif current_epoch <= freeze_ep + 7:
            lid_w = 0.1  # Early joint: 10% LID
        else:
            lid_w = self.hparams.lid_loss_weight  # Full: 20% LID
        
        # NaN protection on CTC loss (bf16 + long audio can overflow)
        ctc_loss = torch.nan_to_num(ctc_loss, nan=0.0, posinf=100.0, neginf=0.0)
        loss = lid_w * lid_loss + (1.0 - lid_w) * ctc_loss

        # --- 4. LAL Loss (Language Alignment Loss) ---
        # Cross-entropy between LAL predictions and ground-truth LID labels
        lal_w = getattr(self.hparams, "lal_loss_weight", 0.0)
        if lal_w > 0 and "lal_logits" in predictions:
            lal_logits = predictions["lal_logits"]  # [B, T, 3]
            T_lal = lal_logits.shape[1]
            # Align LID targets to LAL frame count
            lal_targets = lid_targets_interp[:, :T_lal].reshape(-1).clamp(0, 2)
            lal_logits_flat = lal_logits[:, :T_lal, :].reshape(-1, lal_logits.shape[-1])
            lal_loss = F.cross_entropy(lal_logits_flat, lal_targets)
            lal_loss = torch.nan_to_num(lal_loss, nan=0.0, posinf=0.0, neginf=0.0)
            loss = loss + lal_w * lal_loss
        
        # Final safety: ensure loss is always finite
        loss = torch.nan_to_num(loss, nan=0.0, posinf=100.0, neginf=0.0)

        # Log losses once to verify all paths
        if not hasattr(self, "_logged_ctc"):
            lal_info = f", lal={lal_loss.item():.4f}" if lal_w > 0 else ""
            logger.info(f"CTC loss active! lid={lid_loss.item():.4f}, ctc={ctc_loss.item():.4f}"
                        f"{lal_info}, total={loss.item():.4f}")
            self._logged_ctc = True

        # Compute Word Error Rate (WER) during evaluation
        if stage != sb.Stage.TRAIN and "tokens" in predictions:
            predicted_words = [
                self.hparams.tokenizer.sp.decode_ids(prediction).split(" ")
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
        # Sync model warmup flag with config
        if hasattr(self.modules.dsp_model, "warmup"):
            self.modules.dsp_model.warmup = self.hparams.warmup_only
        
        if stage != sb.Stage.TRAIN:
            self.wer_metric = self.hparams.error_rate_computer()
            self.cs_wer_metric = self.hparams.error_rate_computer()

        # Freeze DSP model (LoRA + GRU + Prompt) for first N epochs
        # so CTC head can learn stable pretrained features first.
        # This matches the baseline's freeze_wav2vec2_epochs mechanism.
        # Works with ALL model variants (DSP_XLSR, MoE, CrossAttn, Fusion).
        freeze_epochs = getattr(self.hparams, "freeze_dsp_epochs", 0)
        if stage == sb.Stage.TRAIN and freeze_epochs > 0:
            if epoch <= freeze_epochs:
                for p in self.modules.dsp_model.parameters():
                    p.requires_grad = False
                # Set backbone LR to 0
                self.optimizer.param_groups[0]["lr"] = 0.0
                if epoch == 1:
                    logger.info(f"DSP model FROZEN for epochs 1-{freeze_epochs} (CTC head trains alone)")
            else:
                # Generic unfreeze: scan ALL named parameters,
                # unfreeze LoRA/adapter weights + any custom layers.
                # Pretrained backbone weights stay frozen by design.
                dsp = self.modules.dsp_model
                for name, p in dsp.named_parameters():
                    # Always unfreeze: LoRA adapters, GRU, prompt, gate, lang_embed,
                    # fusion, cross_attn, layer_norm
                    trainable_keywords = [
                        'adapter', 'lora', 'prompt', 'gate', 'lang',
                        'gru', 'lid', 'fusion', 'cross_attn', 'layer_norm',
                    ]
                    # Partial Unfreezing: also unfreeze top-2 encoder layers (22, 23)
                    # for maximum convergence speed. These layers handle high-level
                    # language reasoning and add ~15M params (~5% of model).
                    partial_unfreeze = [
                        'layers.22.', 'layers.23.',
                    ]
                    if (any(kw in name.lower() for kw in trainable_keywords) or
                        any(puf in name for puf in partial_unfreeze)):
                        p.requires_grad = True
                if epoch == freeze_epochs + 1:
                    n_train = sum(p.numel() for p in dsp.parameters() if p.requires_grad)
                    logger.info(f"DSP model UNFROZEN — {n_train:,} trainable params")

    def on_fit_batch_end(self, batch, outputs, loss, should_update):
        """Step the LR scheduler — use output as SCALE FACTOR for per-group LRs."""
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
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]}, min_keys=["WER"],
            ) if not self.hparams.warmup_only else self.checkpointer.save_and_keep_only(
                meta={"loss": stage_loss}, min_keys=["loss"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )


def dataio_prepare(hparams):
    # 1. Audio Pipeline
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        if len(sig.shape) > 1 and sig.shape[1] > 1:
            sig = torch.mean(sig, dim=1)
        return sig

    # 2. Text Pipeline (ASR)
    @sb.utils.data_pipeline.takes("words")
    @sb.utils.data_pipeline.provides("words", "tokens")
    def text_pipeline(words):
        yield words
        tokens = hparams["tokenizer"].sp.encode_as_ids(words)
        yield torch.LongTensor(tokens)

    # 3. LID Pipeline (DSP-CS)
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
    
    asr_brain = DSP_CTC_ASR(
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
