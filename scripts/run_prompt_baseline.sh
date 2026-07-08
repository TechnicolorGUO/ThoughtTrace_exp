#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG="${1:-configs/prompt_qwen3_4b.yaml}"
if [[ $# -gt 0 ]]; then
  shift
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Config file does not exist: $CONFIG" >&2
  exit 1
fi

python src/inference/generate.py --config "$CONFIG" "$@"
