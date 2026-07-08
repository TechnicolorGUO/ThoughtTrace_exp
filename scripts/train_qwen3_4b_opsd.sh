#!/usr/bin/env bash
# Train Qwen3-4B with OPSD (On-Policy Self-Distillation), LoRA.
# OPSD is TRL/accelerate-based (unlike the swift-based SFT pipeline in this repo).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/opsd_qwen3_4b_lora.yaml}"
ACCEL_CONFIG="${ACCEL_CONFIG:-configs/opsd_accelerate.yaml}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file does not exist: $CONFIG" >&2
  exit 1
fi
if [[ ! -f "$ACCEL_CONFIG" ]]; then
  echo "Accelerate config does not exist: $ACCEL_CONFIG" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NUM_PROCESSES="${NUM_PROCESSES:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

accelerate launch \
  --config_file "$ACCEL_CONFIG" \
  --num_processes "$NUM_PROCESSES" \
  src/opsd/opsd_train.py \
  --config "$CONFIG"
