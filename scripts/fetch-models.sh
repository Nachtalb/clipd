#!/usr/bin/env bash
# Fetch CLIP ViT-B/32 int8 ONNX models + tokenizer into ./models
set -euo pipefail

REPO="Xenova/clip-vit-base-patch32"
BASE="https://huggingface.co/${REPO}/resolve/main"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"

mkdir -p "$DIR"

fetch() {
  local src="$1" dst="$2"
  if [ -f "$DIR/$dst" ]; then
    echo "have   $dst"
    return
  fi
  echo "get    $dst"
  curl -fsSL --retry 3 -o "$DIR/$dst.part" "$BASE/$src"
  mv "$DIR/$dst.part" "$DIR/$dst"
}

fetch "onnx/vision_model_quantized.onnx" "vision.onnx"
fetch "onnx/text_model_quantized.onnx"   "text.onnx"
fetch "tokenizer.json"                   "tokenizer.json"

cd "$DIR"
if [ -f SHA256SUMS ]; then
  sha256sum -c SHA256SUMS
else
  sha256sum vision.onnx text.onnx tokenizer.json > SHA256SUMS
  echo "wrote SHA256SUMS (first run — commit it to pin these files)"
fi

ls -l vision.onnx text.onnx tokenizer.json
