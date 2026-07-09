#!/usr/bin/env bash
# Train Qwen3-4B with the migrated RLCSD OPSD implementation on ThoughtTrace data.
# Thin shim: forwards a YAML config to the vendored RLCSD launcher (_run_verl.sh),
# which reads the config, builds the hydra overrides, and runs
# src.opsd_rlcsd.self_distill_main with third_party/verl on PYTHONPATH.
#
# Usage:
#   bash scripts/train_qwen3_4b_rlcsd_opsd.sh [config.yaml] [extra hydra overrides...]
bash "$(dirname "$0")/_run_verl.sh" \
  "${1:-configs/rlcsd_qwen3_4b_opsd_thoughttrace.yaml}" \
  "${@:2}"
