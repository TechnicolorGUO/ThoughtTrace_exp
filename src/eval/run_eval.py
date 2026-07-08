"""Unified evaluation entry point for ThoughtTrace experiments.

This script will load prediction JSONL files, normalize or parse generated
outputs, run configured metrics, and write a compact evaluation summary for
each baseline or checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.bleu import corpus_bleu
from src.eval.embedding_sim import embedding_similarity
from src.inference.parse_outputs import extract_reply


DEFAULT_CONFIG = {
    "prediction_key": "prediction",
    "reference_key": "reference",
    "parse_thought": False,
    "skip_embedding": False,
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "embedding_batch_size": 32,
}


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


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return load_flat_yaml(path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Eval config must be a mapping: {path}")
    return data


def load_flat_yaml(path: Path) -> dict[str, Any]:
    """Load simple top-level key-value YAML configs without PyYAML."""
    data: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"Unsupported YAML line at {path}:{line_number}: {raw_line.rstrip()}")
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                raise ValueError(f"Empty YAML key at {path}:{line_number}")
            data[key] = parse_scalar(value)
    return data


def parse_scalar(value: str) -> Any:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() in {"null", "none", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def get_nested(row: dict[str, Any], key: str) -> Any:
    value: Any = row
    for part in key.split("."):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            return None
    return value


def collect_texts(
    rows: list[dict[str, Any]],
    *,
    prediction_key: str,
    reference_key: str,
    parse_thought: bool,
) -> tuple[list[str], list[str]]:
    predictions: list[str] = []
    references: list[str] = []
    for index, row in enumerate(rows):
        prediction = get_nested(row, prediction_key)
        reference = get_nested(row, reference_key)
        if prediction is None:
            raise KeyError(f"Missing prediction key '{prediction_key}' in row {index}")
        if reference is None:
            raise KeyError(f"Missing reference key '{reference_key}' in row {index}")
        prediction_text = str(prediction)
        reference_text = str(reference)
        if parse_thought:
            prediction_text = extract_reply(prediction_text)
            reference_text = extract_reply(reference_text)
        predictions.append(prediction_text)
        references.append(reference_text)
    return predictions, references


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if args.config:
        config.update(load_yaml(Path(args.config)))

    cli_overrides = {
        "predictions": args.predictions,
        "output": args.output,
        "prediction_key": args.prediction_key,
        "reference_key": args.reference_key,
        "parse_thought": args.parse_thought,
        "skip_embedding": args.skip_embedding,
        "embedding_model": args.embedding_model,
        "embedding_batch_size": args.embedding_batch_size,
    }
    config.update({key: value for key, value in cli_overrides.items() if value is not None})
    if not config.get("predictions"):
        raise ValueError("Missing required field: predictions. Provide it in --config or with --predictions.")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ThoughtTrace prediction JSONL files.")
    parser.add_argument("--config", help="Evaluation YAML config.")
    parser.add_argument("--predictions", help="Prediction JSONL file.")
    parser.add_argument("--output", help="Optional path for evaluation summary JSON.")
    parser.add_argument("--prediction-key")
    parser.add_argument("--reference-key")
    parser.add_argument("--parse-thought", action="store_true", default=None)
    parser.add_argument("--skip-embedding", action="store_true", default=None)
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-batch-size", type=int)
    args = parser.parse_args()
    config = build_config(args)

    prediction_path = Path(config["predictions"])
    rows = load_jsonl(prediction_path)
    predictions, references = collect_texts(
        rows,
        prediction_key=str(config["prediction_key"]),
        reference_key=str(config["reference_key"]),
        parse_thought=bool(config["parse_thought"]),
    )

    metrics: dict[str, Any] = {
        "prediction_file": str(prediction_path),
        "num_examples": len(predictions),
        "bleu": corpus_bleu(predictions, references),
    }
    if not config["skip_embedding"]:
        metrics["embedding"] = embedding_similarity(
            predictions,
            references,
            model_name=str(config["embedding_model"]),
            batch_size=int(config["embedding_batch_size"]),
        )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if config.get("output"):
        write_json(Path(config["output"]), metrics)


if __name__ == "__main__":
    main()
