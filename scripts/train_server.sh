#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-dsp}"
CONFIG_PATH="${2:-}"

if [[ -z "$CONFIG_PATH" ]]; then
    case "$MODE" in
        baseline)
            CONFIG_PATH="recipes/Baselines/ASR/hparams/xlsr.yaml"
            ;;
        dsp)
            CONFIG_PATH="recipes/DSP_CodeSwitch/ASR/train_xlsr_ctc.yaml"
            ;;
        whisper)
            CONFIG_PATH="recipes/DSP_CodeSwitch/ASR/train_whisper.yaml"
            ;;
        *)
            echo "[error] Unknown mode: $MODE" >&2
            echo "Usage: $0 <baseline|dsp|whisper> [config_path]" >&2
            exit 1
            ;;
    esac
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[error] Config not found: $CONFIG_PATH" >&2
    exit 1
fi

if [[ ! -d "$ROOT_DIR/data/vimedcss/wavs" ]]; then
    echo "[error] Missing data folder: $ROOT_DIR/data/vimedcss/wavs" >&2
    exit 1
fi

if [[ ! -f "$ROOT_DIR/recipes/DSP_CodeSwitch/Tokenizer/save/tokenizer_5000_bpe.model" ]]; then
    echo "[error] Missing tokenizer model in recipes/DSP_CodeSwitch/Tokenizer/save" >&2
    exit 1
fi

if [[ ! -d "$ROOT_DIR/.venv" ]]; then
    echo "[error] Missing virtualenv: $ROOT_DIR/.venv" >&2
    echo "[hint] Run scripts/setup_server_env.sh first." >&2
    exit 1
fi

source "$ROOT_DIR/.venv/bin/activate"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

if [[ "${HF_HUB_OFFLINE:-}" == "1" ]]; then
    echo "[info] HF_HUB_OFFLINE=1, training will use local HuggingFace cache only."
fi

if [[ "$MODE" == "baseline" ]]; then
    cd "$ROOT_DIR/recipes/Baselines/ASR"
    exec python train_ssl_ctc.py "$(realpath --relative-to="$PWD" "$ROOT_DIR/$CONFIG_PATH")"
fi

if [[ "$MODE" == "whisper" ]]; then
    cd "$ROOT_DIR/recipes/DSP_CodeSwitch/ASR"
    exec python train_whisper.py "$(realpath --relative-to="$PWD" "$ROOT_DIR/$CONFIG_PATH")"
fi

cd "$ROOT_DIR/recipes/DSP_CodeSwitch/ASR"
exec python train_ctc.py "$(realpath --relative-to="$PWD" "$ROOT_DIR/$CONFIG_PATH")"
