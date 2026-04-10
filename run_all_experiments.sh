#!/bin/bash
# ==============================================================================
# QUẢN LÝ HUẤN LUYỆN - DSP-CS ASR THESIS EXPERIMENTS
# ==============================================================================
# Script này chứa các lệnh command chạy Baseline và các biến thể DSP-CS.
# Để chạy một mô hình, hãy bỏ comment (xóa dấu #) ở dòng tương ứng.
# Khuyến nghị: Chạy với môi trường đã kích hoạt (conda activate asr_env)
# ==============================================================================

# Thiết lập môi trường để chạy offline (tránh lỗi HuggingFace Timeout)
export HF_HUB_OFFLINE=1

echo "==========================================================="
echo " MENU HUẤN LUYỆN (EXPERIMENTS RUNNER) "
echo "==========================================================="

# ------------------------------------------------------------------------------
# 1. CHẠY BASELINE (XLS-R Full Fine-tuning)
# Sinh ra file log tại: recipes/Baselines/ASR/results/Baseline_XLSR_53/...
# ------------------------------------------------------------------------------
run_baseline() {
    echo "[!] Đang chạy Baseline XLS-R 315M params..."
    cd recipes/Baselines/ASR || exit
    python3 train_ssl_ctc.py hparams/xlsr.yaml \
        --max_grad_norm 1.0 \
        --precision bf16 \
        --grad_accumulation_factor 4
    cd ../../../
}

# ------------------------------------------------------------------------------
# 2. CHẠY DSP-CS GỐC (Top-6 LoRA + LayerNorm Trick + Gating)
# Sinh ra file log tại: recipes/DSP_CodeSwitch/ASR/results/DSP_XLSR_CTC/...
# ------------------------------------------------------------------------------
run_dsp_cs_original() {
    echo "[!] Đang chạy DSP-CS Gốc (Hướng hiện tại)..."
    cd recipes/DSP_CodeSwitch/ASR || exit
    python3 train_ctc.py train_xlsr_ctc.yaml \
        --max_grad_norm 1.0 \
        --precision bf16 \
        --grad_accumulation_factor 4
    cd ../../../
}

# ------------------------------------------------------------------------------
# 3. CHẠY DSP-CS VỚI MoE-LoRA (Chuyên gia Ngôn Ngữ)
# ------------------------------------------------------------------------------
run_dsp_moe() {
    echo "[!] Đang chạy DSP-CS nhánh MoE-LoRA..."
    cd recipes/DSP_CodeSwitch/ASR || exit
    python3 train_ctc.py train_moe.yaml \
        --max_grad_norm 1.0 \
        --precision bf16 \
        --grad_accumulation_factor 4
    cd ../../../
}

# ------------------------------------------------------------------------------
# 4. CHẠY DSP-CS VỚI CROSS-ATTENTION INJECTION (Hướng 5)
# ------------------------------------------------------------------------------
run_dsp_cross_attn() {
    echo "[!] Đang chạy DSP-CS nhánh Cross-Attention..."
    cd recipes/DSP_CodeSwitch/ASR || exit
    python3 train_ctc.py train_cross.yaml \
        --max_grad_norm 1.0 \
        --precision bf16 \
        --grad_accumulation_factor 4
    cd ../../../
}

# ------------------------------------------------------------------------------
# 5. CHẠY DSP-CS VỚI ADAPTER FUSION (Hướng 6)
# ------------------------------------------------------------------------------
run_dsp_fusion() {
    echo "[!] Đang chạy DSP-CS nhánh AdapterFusion..."
    cd recipes/DSP_CodeSwitch/ASR || exit
    python3 train_ctc.py train_fusion.yaml \
        --max_grad_norm 1.0 \
        --precision bf16 \
        --grad_accumulation_factor 4
    cd ../../../
}

# ==============================================================================
# HƯỚNG DẪN SỬ DỤNG:
# Xóa dấu '#' trước hàm bạn muốn chạy để tự động thực thi.
# ==============================================================================

# run_baseline
# run_dsp_cs_original
# run_dsp_moe
# run_dsp_cross_attn
# run_dsp_fusion

echo "Mở file run_all_experiments.sh bằng text editor, xóa dấu '#' ở dòng lệnh bạn muốn chạy ở cuối file, rồi thực thi lại lệnh: bash run_all_experiments.sh"
