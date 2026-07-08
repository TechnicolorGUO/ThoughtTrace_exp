#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/sft_qwen3_4b_lora_thought.yaml}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file does not exist: $CONFIG" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

swift sft --config "$CONFIG"
