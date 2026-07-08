#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SYSTEM_PROMPT = (
    "You are simulating a real user in a human-AI conversation. Given the conversation history and the "
    "assistant's latest reply, write the user's next message only."
)


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


def split_conversations(
    conversations: list[dict[str, Any]],
    *,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(conversations)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    test_size = max(1, round(len(shuffled) * test_ratio)) if shuffled else 0
    test_conversations = shuffled[:test_size]
    train_conversations = shuffled[test_size:]
    return train_conversations, test_conversations


def thought_text(items: list[dict[str, Any]]) -> str:
    return "；".join(str(item.get("content", "")).strip() for item in items if str(item.get("content", "")).strip())


def format_history(messages: list[dict[str, Any]], *, max_history_turns: int | None = None) -> str:
    if max_history_turns is not None and max_history_turns > 0:
        messages = messages[-max_history_turns * 2 :]

    lines: list[str] = []
    for message in messages:
        role = "User" if message.get("type") == "user" else "Assistant"
        content = str(message.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_input(history: list[dict[str, Any]], assistant_message: dict[str, Any], *, max_history_turns: int | None) -> str:
    history_text = format_history(history, max_history_turns=max_history_turns)
    latest_reply = str(assistant_message.get("content", "")).strip()
    if history_text:
        return f"[Conversation History]\n{history_text}\n\n[Assistant Latest Reply]\nAssistant: {latest_reply}"
    return f"[Conversation History]\n\n[Assistant Latest Reply]\nAssistant: {latest_reply}"


def build_output(next_user_message: dict[str, Any]) -> str:
    return str(next_user_message.get("content", "")).strip()


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
        "skipped_empty_reply": 0,
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

            output = build_output(next_user_message)
            if not output:
                stats["skipped_empty_reply"] += 1
                continue

            sample = {
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_input(
                            messages[:index],
                            assistant_message,
                            max_history_turns=max_history_turns,
                        ),
                    },
                    {"role": "assistant", "content": output},
                ],
                "metadata": {
                    "conversation_id": conversation.get("id"),
                    "assistant_message_id": assistant_message.get("id"),
                    "next_user_message_id": next_user_message.get("id"),
                    "model_name": conversation.get("model_name"),
                    "model_provider": conversation.get("model_provider"),
                },
            }
            samples.append(sample)

    stats["samples"] = len(samples)
    return samples, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ThoughtTrace JSONL to no-thought ms-swift chat SFT JSONL.")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data/raw/ThoughtTrace.jsonl"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data/processed"))
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-history-turns", type=int, default=6)
    parser.add_argument("--allow-missing-reaction", action="store_true")
    parser.add_argument("--allow-missing-reason", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    conversations = load_jsonl(input_path)
    train_conversations, test_conversations = split_conversations(
        conversations,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
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

    write_jsonl(train_conversations, output_dir / "train_conversations.jsonl")
    write_jsonl(test_conversations, output_dir / "test_conversations.jsonl")
    write_jsonl(train_rows, output_dir / "user_sim_no_thought_train.jsonl")
    write_jsonl(test_rows, output_dir / "user_sim_no_thought_test.jsonl")
    write_jsonl(train_rows[:20], output_dir / "user_sim_no_thought_preview.jsonl")

    stats = {
        "total_conversations": len(conversations),
        "train_conversations": len(train_conversations),
        "test_conversations": len(test_conversations),
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "max_history_turns": args.max_history_turns,
        "require_reaction": not args.allow_missing_reaction,
        "require_reason": not args.allow_missing_reason,
        "train_stats": train_stats,
        "test_stats": test_stats,
    }
    (output_dir / "split_ids.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "test_ratio": args.test_ratio,
                "train_conversation_ids": [row.get("id") for row in train_conversations],
                "test_conversation_ids": [row.get("id") for row in test_conversations],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "stats_no_thought.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
