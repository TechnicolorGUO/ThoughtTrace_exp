#!/usr/bin/env bash
# Train Qwen3-4B with the RLCSD/verl-style OPSD implementation adapted for ThoughtTrace.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/rlcsd_qwen3_4b_opsd_thoughttrace.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file does not exist: $CONFIG" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
MASTER_PORT="${MASTER_PORT:-29511}"

torchrun \
  --nproc_per_node "$NPROC_PER_NODE" \
  --master_port "$MASTER_PORT" \
  -m src.opsd_rlcsd.verl_main \
  --config "$CONFIG" \
  "$@"
