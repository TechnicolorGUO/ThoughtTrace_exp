#!/usr/bin/env python3
"""Convert ThoughtTrace conversations directly into RLCSD-OPSD verl parquet.

OPSD (On-Policy Self-Distillation) trains a student to imitate a *thought-
informed teacher* WITHOUT ever giving the student the thought:

  - student sees only the conversation history + the assistant's latest reply,
    and rolls out the user's next message on-policy.
  - teacher sees the same context PLUS the user's private thought
    ([Reaction] + [Motivation]) and scores the student's rolled-out tokens.
  - loss = forward-KL(teacher || student) on those shared response tokens.

This script reads the SAME train/test conversation split produced by
convert_to_swift_sft.py (train_conversations.jsonl / test_conversations.jsonl)
so OPSD/SFT/OPD stay comparable, and writes verl parquet directly -- no
intermediate problem/solution JSONL.

RLCSD verl schema (per row), consumed by RLCSD/src/self_distill_main.py:
  data_source   str    -- NON-math tag, so the math prompt monkey-patch is skipped
  prompt        list   -- student chat messages (instruction + context)
  ability       str
  reward_model  dict   -- {"style": "reference", "ground_truth": <next msg>}
                          reference/eval metadata only; teacher must NOT see it
                          (enforced by the solution-only teacher template).
  extra_info    dict   -- {"problem": <context>, "solution": <thought>, ...}
                          problem+solution are what the teacher prompt is rebuilt
                          from at run time.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_SOURCE = "thoughttrace_user_sim"

# Student-side instruction. The student is asked to play the *user* and produce
# the next user message given the visible context. Keep this parallel to the
# teacher framing in opsd_format so the shared response tokens line up.
STUDENT_INSTRUCTION = (
    "You are role-playing as the user in the conversation below. "
    "Based on the conversation history and the assistant's latest reply, "
    "write the user's next message."
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def build_context(history: list[dict[str, Any]], assistant_message: dict[str, Any], *, max_history_turns: int | None) -> str:
    """Shared context both student and teacher condition on (no thought)."""
    history_text = format_history(history, max_history_turns=max_history_turns)
    latest_reply = str(assistant_message.get("content", "")).strip()
    return (
        f"[Conversation History]\n{history_text}\n\n"
        f"[Assistant Latest Reply]\nAssistant: {latest_reply}"
    )


def build_thought(assistant_message: dict[str, Any], next_user_message: dict[str, Any]) -> str:
    """Privileged reference (teacher-only): the user's private thought."""
    reaction = thought_text(assistant_message.get("reactions") or [])
    motivation = thought_text(next_user_message.get("reasons") or [])
    return f"[Reaction]: {reaction}\n[Motivation]: {motivation}"

def build_samples(
    conversations: list[dict[str, Any]],
    *,
    split: str,
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

            context = build_context(
                messages[:index], assistant_message, max_history_turns=max_history_turns
            )
            thought = build_thought(assistant_message, next_user_message)
            reply = str(next_user_message.get("content", "")).strip()
            uid = (
                f"{conversation.get('id')}:"
                f"{assistant_message.get('id')}:"
                f"{next_user_message.get('id')}"
            )

            samples.append(
                {
                    "data_source": DATA_SOURCE,
                    # Student rollout prompt: instruction + shared context. The
                    # non-math data_source keeps self_distill_main from rewriting
                    # this with the \boxed{} math template.
                    "prompt": [
                        {"role": "user", "content": f"{STUDENT_INSTRUCTION}\n\n{context}"}
                    ],
                    "ability": "user_simulation",
                    # ground_truth is the real next message. It is reference/eval
                    # metadata ONLY -- the solution-only teacher template must not
                    # feed it to the teacher, or the gold target leaks.
                    "reward_model": {"style": "reference", "ground_truth": reply},
                    "extra_info": {
                        "split": split,
                        "uid": uid,
                        # problem = shared context; solution = privileged thought.
                        # The teacher prompt is rebuilt at run time from these two.
                        "problem": context,
                        "solution": thought,
                        "reference": reply,
                        "conversation_id": conversation.get("id"),
                        "assistant_message_id": assistant_message.get("id"),
                        "next_user_message_id": next_user_message.get("id"),
                    },
                }
            )

    stats["samples"] = len(samples)
    return samples, stats


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
    parser = argparse.ArgumentParser(description="Convert ThoughtTrace split conversations to RLCSD-OPSD verl parquet.")
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
        split="train",
        max_history_turns=args.max_history_turns,
        require_reaction=not args.allow_missing_reaction,
        require_reason=not args.allow_missing_reason,
    )
    test_rows, test_stats = build_samples(
        test_conversations,
        split="test",
        max_history_turns=args.max_history_turns,
        require_reaction=not args.allow_missing_reaction,
        require_reason=not args.allow_missing_reason,
    )

    write_parquet(train_rows, output_dir / "user_sim_opsd_train.parquet")
    write_parquet(test_rows, output_dir / "user_sim_opsd_test.parquet")

    stats = {
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "max_history_turns": args.max_history_turns,
        "require_reaction": not args.allow_missing_reaction,
        "require_reason": not args.allow_missing_reason,
        "train_stats": train_stats,
        "test_stats": test_stats,
        "train_output": str(output_dir / "user_sim_opsd_train.parquet"),
        "test_output": str(output_dir / "user_sim_opsd_test.parquet"),
    }
    (output_dir / "stats_opsd.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
