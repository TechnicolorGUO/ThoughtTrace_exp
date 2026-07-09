"""Convert ms-swift inference JSONL outputs to the project eval schema."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from parse_outputs import extract_reply


BRACKET_REPLY_PATTERN = re.compile(r"\[Reply\]\s*(.*?)(?:\n?\]\s*)?$", re.DOTALL | re.IGNORECASE)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_swift_response(text: str, *, parse_reply: bool) -> str:
    text = str(text or "").strip()
    if not parse_reply:
        return text

    reply = extract_reply(text)
    if reply != text:
        return reply

    match = BRACKET_REPLY_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text


def prompt_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        return []
    if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "assistant":
        messages = messages[:-1]
    return [
        {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
        for message in messages
        if isinstance(message, dict)
    ]


def convert(rows: list[dict[str, Any]], *, parse_reply: bool) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        response = row.get("response", row.get("prediction", ""))
        converted.append(
            {
                "prediction": parse_swift_response(response, parse_reply=parse_reply),
                "reference": str(row.get("labels", row.get("reference", ""))),
                "metadata": row.get("metadata", {}),
                "prompt_messages": prompt_messages(row.get("messages")),
                "example_index": index,
                "raw_response": response,
            }
        )
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ms-swift inference outputs to eval predictions JSONL.")
    parser.add_argument("--input", required=True, help="Raw JSONL written by swift infer --result_path.")
    parser.add_argument("--output", required=True, help="Project predictions JSONL output path.")
    parser.add_argument(
        "--parse-reply",
        action="store_true",
        help="Extract <reply>...</reply> or [Reply] sections before evaluation.",
    )
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    converted = convert(rows, parse_reply=bool(args.parse_reply))
    write_jsonl(converted, Path(args.output))
    print(f"Wrote {len(converted)} predictions to {args.output}")


if __name__ == "__main__":
    main()
