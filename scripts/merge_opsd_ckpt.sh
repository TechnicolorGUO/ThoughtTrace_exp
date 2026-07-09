#!/usr/bin/env bash
# Merge a verl FSDP sharded checkpoint into a standard HuggingFace model dir.
# verl saves actor weights as model_world_size_N_rank_*.pt (FSDP shards); this
# reconstructs a single HF model (config + safetensors + tokenizer) that plain
# transformers / vLLM can load. With lora.merge=True the shards already hold the
# full merged weights, so the output is a complete Qwen3-4B, not an adapter.
#
# Usage:
#   bash scripts/merge_opsd_ckpt.sh <global_step_dir> [target_dir]
# Example:
#   bash scripts/merge_opsd_ckpt.sh \
#     outputs/qwen3_4b_rlcsd_opsd/20260709.130309/global_step_100
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
export PYTHONPATH="$(pwd)/third_party/verl:$(pwd):${PYTHONPATH}"

STEP_DIR="${1:?Usage: $0 <global_step_dir> [target_dir]}"
ACTOR_DIR="${STEP_DIR%/}/actor"
TARGET_DIR="${2:-${STEP_DIR%/}/merged_hf}"

if [ ! -d "$ACTOR_DIR" ]; then
    echo "actor dir not found: $ACTOR_DIR" >&2
    exit 1
fi

echo "Merging FSDP shards: $ACTOR_DIR -> $TARGET_DIR"
python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$TARGET_DIR"

echo "Done. Merged HF model at: $TARGET_DIR"
