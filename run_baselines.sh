#!/bin/bash
# ================================================================
# Run All Baseline Experiments
# ================================================================
# Usage: bash run_baselines.sh [LAL|AG|LAD|ALL]
# Must run from the DSP-CS-ASR root directory on the server.
# ================================================================

set -e

EXPERIMENT=${1:-ALL}
DATA_FOLDER="/workspace/DSP-CS-ASR/data/vimedcss"

echo "=========================================="
echo "Running Baseline Experiments: $EXPERIMENT"
echo "=========================================="

# --- Baseline B1: LAL (Liu et al., 2024) ---
if [ "$EXPERIMENT" = "LAL" ] || [ "$EXPERIMENT" = "ALL" ]; then
    echo ""
    echo "[B1] LAL Baseline (Liu et al., 2024)"
    echo "  - Linear LID head + CrossEntropy + Fixed β=0.05"
    echo "=================================================="
    cd /workspace/DSP-CS-ASR/recipes/Baselines/LAL
    PYTHONPATH=/workspace/DSP-CS-ASR python3 train_lal.py train_lal.yaml \
        --data_folder ${DATA_FOLDER} \
        --output_folder ../../../results/Baseline_LAL/2025 \
        --save_folder ../../../results/Baseline_LAL/2025/save \
        --train_log ../../../results/Baseline_LAL/2025/train_log.txt
    echo "[B1] LAL DONE!"
fi

# --- Baseline B2: AG (Aditya et al., ICASSP 2024) ---
if [ "$EXPERIMENT" = "AG" ] || [ "$EXPERIMENT" = "ALL" ]; then
    echo ""
    echo "[B2] AG Baseline (Aditya et al., ICASSP 2024)"
    echo "  - 2-stage: CE only (8ep) → CE + AG Loss"
    echo "  - Bilingual prompt <|vi|><|en|>"
    echo "=================================================="
    cd /workspace/DSP-CS-ASR/recipes/Baselines/AG
    PYTHONPATH=/workspace/DSP-CS-ASR python3 train_ag.py train_ag.yaml \
        --data_folder ${DATA_FOLDER} \
        --output_folder ../../../results/Baseline_AG/2025 \
        --save_folder ../../../results/Baseline_AG/2025/save \
        --train_log ../../../results/Baseline_AG/2025/train_log.txt
    echo "[B2] AG DONE!"
fi

# --- Baseline P+: LAD (Zhao et al., ICASSP 2025) ---
if [ "$EXPERIMENT" = "LAD" ] || [ "$EXPERIMENT" = "ALL" ]; then
    echo ""
    echo "[P+] LAD Baseline (Zhao et al., ICASSP 2025)"
    echo "  - Encoding Refiner + Per-layer Dual-path Adapters"
    echo "  - CTC auxiliary + Sigmoid fusion"
    echo "=================================================="
    cd /workspace/DSP-CS-ASR/recipes/Baselines/LAD
    PYTHONPATH=/workspace/DSP-CS-ASR python3 train_lad.py train_lad.yaml \
        --data_folder ${DATA_FOLDER} \
        --output_folder ../../../results/Baseline_LAD/2025 \
        --save_folder ../../../results/Baseline_LAD/2025/save \
        --train_log ../../../results/Baseline_LAD/2025/train_log.txt
    echo "[P+] LAD DONE!"
fi

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "=========================================="
echo "Results:"
echo "  LAL: results/Baseline_LAL/2025/train_log.txt"
echo "  AG:  results/Baseline_AG/2025/train_log.txt"
echo "  LAD: results/Baseline_LAD/2025/train_log.txt"
