#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/prompt_qwen3_4b_vllm.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file does not exist: $CONFIG" >&2
  exit 1
fi

export VLLM_HOST_IP="${VLLM_HOST_IP:-127.0.0.1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

python src/inference/generate_vllm.py --config "$CONFIG" "$@"
