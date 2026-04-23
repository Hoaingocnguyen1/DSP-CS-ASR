#!/bin/bash
# ================================================================
# Run A/B Experiments (EXP-A → EXP-D)
# ================================================================
# Usage: bash run_experiments.sh [A|B|C|D|ALL]
# Chạy từ thư mục DSP-CS-ASR root trên server.
# ================================================================

set -e

EXPERIMENT=${1:-ALL}
BASE_DIR="/workspace/DSP-CS-ASR"
ASR_DIR="${BASE_DIR}/recipes/DSP_CodeSwitch/ASR"
DATA_FOLDER="${BASE_DIR}/data/vimedcss"

echo "=========================================="
echo "Running A/B Experiments: $EXPERIMENT"
echo "=========================================="

run_exp() {
    local name=$1
    local yaml=$2
    local out_dir=$3
    
    echo ""
    echo "[$name] Starting..."
    echo "  YAML: $yaml"
    echo "  Output: $out_dir"
    echo "=================================================="
    
    cd ${ASR_DIR}
    PYTHONPATH=${BASE_DIR} python3 train_whisper.py ${yaml} \
        --data_folder ${DATA_FOLDER} \
        --output_folder ${out_dir} \
        --save_folder ${out_dir}/save \
        --train_log ${out_dir}/train_log.txt
    
    echo "[$name] DONE! Results: ${out_dir}/train_log.txt"
}

# --- EXP-A: Bilingual Prompt ---
if [ "$EXPERIMENT" = "A" ] || [ "$EXPERIMENT" = "ALL" ]; then
    run_exp "EXP-A Bilingual" \
        "exp_a_bilingual.yaml" \
        "${BASE_DIR}/results/DSP_Whisper_EXP_A_bilingual/2025"
fi

# --- EXP-B: Dynamic LID Weights ---
if [ "$EXPERIMENT" = "B" ] || [ "$EXPERIMENT" = "ALL" ]; then
    run_exp "EXP-B Dynamic LID" \
        "exp_b_dynamic_lid.yaml" \
        "${BASE_DIR}/results/DSP_Whisper_EXP_B_dynamic_lid/2025"
fi

# --- EXP-C: CTC Auxiliary ---
if [ "$EXPERIMENT" = "C" ] || [ "$EXPERIMENT" = "ALL" ]; then
    run_exp "EXP-C CTC Aux" \
        "exp_c_ctc.yaml" \
        "${BASE_DIR}/results/DSP_Whisper_EXP_C_ctc/2025"
fi

# --- EXP-D: SPAL ---
if [ "$EXPERIMENT" = "D" ] || [ "$EXPERIMENT" = "ALL" ]; then
    run_exp "EXP-D SPAL" \
        "exp_d_spal.yaml" \
        "${BASE_DIR}/results/DSP_Whisper_EXP_D_spal/2025"
fi

echo ""
echo "=========================================="
echo "All requested experiments completed!"
echo "=========================================="
echo ""
echo "Kết quả:"
echo "  EXP-A: results/DSP_Whisper_EXP_A_bilingual/2025/train_log.txt"
echo "  EXP-B: results/DSP_Whisper_EXP_B_dynamic_lid/2025/train_log.txt"
echo "  EXP-C: results/DSP_Whisper_EXP_C_ctc/2025/train_log.txt"
echo "  EXP-D: results/DSP_Whisper_EXP_D_spal/2025/train_log.txt"
