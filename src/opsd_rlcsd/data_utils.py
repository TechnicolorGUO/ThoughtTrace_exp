"""Data loading for ThoughtTrace RLCSD/verl OPSD."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
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


def normalize_item(item: dict, source: str) -> dict:
    problem = str(item.get("problem", "")).strip()
    solution = str(item.get("solution", "")).strip()
    answer = str(item.get("answer", item.get("reply", ""))).strip()
    if not problem:
        raise ValueError(f"Missing problem in {source}")
    if not solution:
        raise ValueError(f"Missing solution/private thought in {source}")
    return {
        "problem": problem,
        "solution": solution,
        "answer": answer,
        "source": "thoughttrace",
        "conversation_id": item.get("conversation_id"),
        "assistant_message_id": item.get("assistant_message_id"),
        "next_user_message_id": item.get("next_user_message_id"),
    }


def load_local_dataset(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return [normalize_item(row, str(path)) for row in load_jsonl(path)]
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"JSON dataset must be a list: {path}")
        return [normalize_item(row, str(path)) for row in data]
    if path.suffix == ".parquet":
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=str(path), split="train")
        return [normalize_item(row, str(path)) for row in dataset]
    raise ValueError(f"Unsupported dataset format: {path}")


def prepare_training_data(
    data_dir: Optional[str] = None,
    dataset_name: str = "data/processed/user_sim_opsd_train.jsonl",
) -> list[dict]:
    del data_dir
    path = Path(dataset_name)
    if not path.exists():
        raise FileNotFoundError(f"ThoughtTrace OPSD train dataset not found: {path}")
    return load_local_dataset(path)


def prepare_eval_data(
    data_dir: Optional[str] = None,
    val_dataset: Optional[str] = None,
) -> dict[str, list[dict]]:
    del data_dir
    if not val_dataset:
        return {}
    path = Path(val_dataset)
    if not path.exists():
        raise FileNotFoundError(f"ThoughtTrace OPSD eval dataset not found: {path}")
    return {"thoughttrace": load_local_dataset(path)}


def extract_answer_from_boxed(text: str) -> str:
    return str(text or "").strip()


def check_answer(prediction: str, ground_truth: str) -> bool:
    return str(prediction or "").strip() == str(ground_truth or "").strip()
