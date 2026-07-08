#!/usr/bin/env python3
"""Convert ThoughtTrace conversations into OPSD (problem/solution) JSONL.

OPSD (On-Policy Self-Distillation) needs two flat columns:
  - problem:  the STUDENT-visible prompt. Here = conversation history + the
              assistant's latest reply (everything the user sees before replying).
  - solution: the PRIVILEGED reference only the TEACHER sees. Here = the user's
              private thought ([Reaction] + [Motivation]).

The student rolls out the user's next message conditioned only on `problem`;
the teacher scores those same tokens with the private thought in context. Training
distills "thought-informed" next-message generation into a student that never
sees the thought.

We reuse the SAME train/test conversation split produced by
convert_to_swift_sft.py (train_conversations.jsonl / test_conversations.jsonl)
so OPSD and SFT stay comparable. The `reply` (ground-truth next message) is kept
as reference metadata only; the OPSD trainer does not consume it (on-policy).
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


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def thought_text(items: list[dict[str, Any]]) -> str:
    return "；".join(
        str(item.get("content", "")).strip()
        for item in items
        if str(item.get("content", "")).strip()
    )


def format_history(messages: list[dict[str, Any]], *, max_history_turns: int | None) -> str:
    if max_history_turns is not None and max_history_turns > 0:
        messages = messages[-max_history_turns * 2 :]
    lines: list[str] = []
    for message in messages:
        role = "User" if message.get("type") == "user" else "Assistant"
        content = str(message.get("content", "")).strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_problem(history: list[dict[str, Any]], assistant_message: dict[str, Any], *, max_history_turns: int | None) -> str:
    """Student-visible prompt: history + assistant latest reply (no thought)."""
    history_text = format_history(history, max_history_turns=max_history_turns)
    latest_reply = str(assistant_message.get("content", "")).strip()
    if history_text:
        return f"[Conversation History]\n{history_text}\n\n[Assistant Latest Reply]\nAssistant: {latest_reply}"
    return f"[Conversation History]\n\n[Assistant Latest Reply]\nAssistant: {latest_reply}"


def build_solution(assistant_message: dict[str, Any], next_user_message: dict[str, Any]) -> str:
    """Privileged reference (teacher-only): the user's private thought."""
    reaction = thought_text(assistant_message.get("reactions") or [])
    motivation = thought_text(next_user_message.get("reasons") or [])
    return f"[Reaction]: {reaction}\n[Motivation]: {motivation}"


def build_samples(
    conversations: list[dict[str, Any]],
    *,
    max_history_turns: int | None,
    require_reaction: bool,
    require_reason: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    samples: list[dict[str, Any]] = []
    stats = {
        "conversations": len(conversations),
        "assistant_to_user_pairs": 0,
        "skipped_missing_reaction": 0,
        "skipped_missing_reason": 0,
        "samples": 0,
    }

    for conversation in conversations:
        messages = conversation.get("messages") or []
        for index, assistant_message in enumerate(messages[:-1]):
            next_user_message = messages[index + 1]
            if assistant_message.get("type") != "assistant" or next_user_message.get("type") != "user":
                continue

            stats["assistant_to_user_pairs"] += 1
            has_reaction = bool(thought_text(assistant_message.get("reactions") or []))
            has_reason = bool(thought_text(next_user_message.get("reasons") or []))
            if require_reaction and not has_reaction:
                stats["skipped_missing_reaction"] += 1
                continue
            if require_reason and not has_reason:
                stats["skipped_missing_reason"] += 1
                continue

            samples.append(
                {
                    "problem": build_problem(
                        messages[:index], assistant_message, max_history_turns=max_history_turns
                    ),
                    "solution": build_solution(assistant_message, next_user_message),
                    # reference only — the on-policy trainer does not read this
                    "reply": str(next_user_message.get("content", "")).strip(),
                    "conversation_id": conversation.get("id"),
                    "assistant_message_id": assistant_message.get("id"),
                    "next_user_message_id": next_user_message.get("id"),
                }
            )

    stats["samples"] = len(samples)
    return samples, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ThoughtTrace conversations to OPSD problem/solution JSONL.")
    parser.add_argument("--processed-dir", default=str(PROJECT_ROOT / "data/processed"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data/processed"))
    parser.add_argument("--max-history-turns", type=int, default=6)
    parser.add_argument("--allow-missing-reaction", action="store_true")
    parser.add_argument("--allow-missing-reason", action="store_true")
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    train_conversations = load_jsonl(processed_dir / "train_conversations.jsonl")
    test_conversations = load_jsonl(processed_dir / "test_conversations.jsonl")

    train_rows, train_stats = build_samples(
        train_conversations,
        max_history_turns=args.max_history_turns,
        require_reaction=not args.allow_missing_reaction,
        require_reason=not args.allow_missing_reason,
    )
    test_rows, test_stats = build_samples(
        test_conversations,
        max_history_turns=args.max_history_turns,
        require_reaction=not args.allow_missing_reaction,
        require_reason=not args.allow_missing_reason,
    )

    write_jsonl(train_rows, output_dir / "user_sim_opsd_train.jsonl")
    write_jsonl(test_rows, output_dir / "user_sim_opsd_test.jsonl")
    write_jsonl(train_rows[:20], output_dir / "user_sim_opsd_preview.jsonl")

    stats = {
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "max_history_turns": args.max_history_turns,
        "require_reaction": not args.allow_missing_reaction,
        "require_reason": not args.allow_missing_reason,
        "train_stats": train_stats,
        "test_stats": test_stats,
    }
    (output_dir / "stats_opsd.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

