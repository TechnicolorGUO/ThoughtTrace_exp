#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/eval_qwen3_4b_lora_thought.yaml}"

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file does not exist: $CONFIG" >&2
  exit 1
fi

python src/eval/run_eval.py --config "$CONFIG"
