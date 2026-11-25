#!/usr/bin/env bash
set -euo pipefail
MODEL="${1:-qwen2.5-1.5b-instruct}"
mkdir -p models
case "$MODEL" in
  qwen2.5-1.5b-instruct) REPO=Qwen/Qwen2.5-1.5B-Instruct-GGUF; FILE=qwen2.5-1.5b-instruct-q4_k_m.gguf ;;
  qwen2.5-3b-instruct) REPO=Qwen/Qwen2.5-3B-Instruct-GGUF; FILE=qwen2.5-3b-instruct-q4_k_m.gguf ;;
  *) echo "unknown model $MODEL" >&2; exit 1 ;;
esac
curl -L --fail -o "models/$FILE" "https://huggingface.co/$REPO/resolve/main/$FILE"
echo "models/$FILE"
