#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <old_prefix> <new_prefix>" >&2
    echo "Example: $0 /home/hnn/Documents/kltn/DSP-CS-ASR /data/projects/DSP-CS-ASR" >&2
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLD_PREFIX="$1"
NEW_PREFIX="$2"

for json_file in \
    "$ROOT_DIR/data/vimedcss/train.json" \
    "$ROOT_DIR/data/vimedcss/valid.json" \
    "$ROOT_DIR/data/vimedcss/test.json" \
    "$ROOT_DIR/data/vimedcss/train_sub.json" \
    "$ROOT_DIR/data/vimedcss/valid_sub.json" \
    "$ROOT_DIR/data/vimedcss/test_sub.json" \
    "$ROOT_DIR/data/vimedcss/test_sub200.json" \
    "$ROOT_DIR/data/vimedcss/train_sample.json" \
    "$ROOT_DIR/data/vimedcss/valid_sample.json" \
    "$ROOT_DIR/data/vimedcss/hard.json" \
    "$ROOT_DIR/data/vimedcss/validation.json"
do
    if [[ -f "$json_file" ]]; then
        sed -i "s|$OLD_PREFIX|$NEW_PREFIX|g" "$json_file"
        echo "[fixed] $json_file"
    fi
done

echo "[done] Updated dataset JSON paths."
