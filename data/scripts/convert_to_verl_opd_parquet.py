#!/usr/bin/env python3
"""Convert ThoughtTrace SFT JSONL into verl OPD parquet files.

OPD uses the standard verl PPO/distillation pipeline. The policy prompt is the
conversation history plus the assistant's latest reply, and the reference next
user message is stored only as metadata/reward ground truth. No private thought
fields are exposed to either student or teacher in this baseline.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def convert_rows(rows: list[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        messages = row.get("messages") or []
        if len(messages) < 2:
            continue

        reference = ""
        if messages[-1].get("role") == "assistant":
            reference = str(messages[-1].get("content", "")).strip()
            prompt = messages[:-1]
        else:
            prompt = messages

        metadata = row.get("metadata") or {}
        uid = (
            f"{metadata.get('conversation_id', split)}:"
            f"{metadata.get('assistant_message_id', index)}:"
            f"{metadata.get('next_user_message_id', index)}"
        )

        converted.append(
            {
                "data_source": "thoughttrace",
                "prompt": prompt,
                "ability": "user_simulation",
                "reward_model": {
                    "style": "reference",
                    "ground_truth": reference,
                },
                "extra_info": {
                    "index": index,
                    "split": split,
                    "uid": uid,
                    "reference": reference,
                    **metadata,
                },
            }
        )
    return converted


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit(
            "pandas is required to write parquet. Install pandas and pyarrow in the verl environment."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ThoughtTrace no-thought SFT JSONL to verl OPD parquet.")
    parser.add_argument(
        "--train-input",
        default=str(PROJECT_ROOT / "data/processed/user_sim_no_thought_train.jsonl"),
    )
    parser.add_argument(
        "--test-input",
        "--val-input",
        dest="test_input",
        default=str(PROJECT_ROOT / "data/processed/user_sim_no_thought_test.jsonl"),
    )
    parser.add_argument(
        "--train-output",
        default=str(PROJECT_ROOT / "data/processed/user_sim_opd_train.parquet"),
    )
    parser.add_argument(
        "--test-output",
        "--val-output",
        dest="test_output",
        default=str(PROJECT_ROOT / "data/processed/user_sim_opd_test.parquet"),
    )
    args = parser.parse_args()

    train_rows = convert_rows(load_jsonl(Path(args.train_input)), split="train")
    test_rows = convert_rows(load_jsonl(Path(args.test_input)), split="test")

    write_parquet(train_rows, Path(args.train_output))
    write_parquet(test_rows, Path(args.test_output))

    stats = {
        "train_input": args.train_input,
        "test_input": args.test_input,
        "train_output": args.train_output,
        "test_output": args.test_output,
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
    }
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
