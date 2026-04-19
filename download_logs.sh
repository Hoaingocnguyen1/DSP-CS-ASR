#!/bin/bash
# ==============================================================================
# SCRIPT TẢI LOGS QUA TAR + SCP (Vượt rào Rsync trên Vast.ai)
# ==============================================================================

PORT=57339
USER_HOST="root@74.48.78.46"
LOCAL_DIR="/home/hnn/Documents/kltn/DSP-CS-ASR/pulled_results/"

mkdir -p "$LOCAL_DIR"

echo "1. Đang ra lệnh ép Server tự gom các file Log thành 1 cục nén (trừ checkpoints bự)..."
ssh -p $PORT $USER_HOST 'cd /workspace/DSP-CS-ASR/recipes/DSP_CodeSwitch/ASR/ && find results/ -type f ! -name "*.ckpt" ! -name "*.pt" ! -name "*.safetensors" ! -name "*.bin" -print0 | tar -czvf results_logs.tar.gz --null -T -'

echo "2. Đang tải cục nén results_logs.tar.gz siêu tốc về máy..."
scp -P $PORT $USER_HOST:/workspace/DSP-CS-ASR/recipes/DSP_CodeSwitch/ASR/results_logs.tar.gz "$LOCAL_DIR"

echo "3. Đang giải nén tại Local..."
cd "$LOCAL_DIR" && tar -xzvf results_logs.tar.gz

echo "✅ HOÀN TẤT! Nhật ký của bác bao gồm toàn bộ folder results đã nằm rải thảm tại: $LOCAL_DIR"

