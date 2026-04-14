#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
CUDA_INDEX_URL="${CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

echo "[setup] root: $ROOT_DIR"
echo "[setup] python: $PYTHON_BIN"
echo "[setup] venv: $VENV_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[error] Cannot find Python executable: $PYTHON_BIN" >&2
    exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

echo "[setup] Installing PyTorch from $CUDA_INDEX_URL"
pip install torch torchaudio --index-url "$CUDA_INDEX_URL"

echo "[setup] Installing project in editable mode"
pip install -e "$ROOT_DIR"

echo "[setup] Installing extra training dependencies"
pip install "transformers>=4.38.0" "datasets>=2.18.0" jiwer pandas

echo "[done] Environment is ready."
echo "[next] source \"$VENV_DIR/bin/activate\""
