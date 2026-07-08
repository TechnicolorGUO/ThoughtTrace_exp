"""Run model inference on processed ThoughtTrace test data.

This script will load a trained base model or LoRA checkpoint, generate user
next-message predictions from fixed test-set inputs, and write prediction
records with metadata, references, and raw model outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = {
    "adapter_path": None,
    "dataset": "data/processed/user_sim_no_thought_test.jsonl",
    "output": "outputs/predictions.jsonl",
    "max_new_tokens": 256,
    "temperature": 0.0,
    "top_p": 1.0,
    "do_sample": False,
    "torch_dtype": "bfloat16",
    "device_map": "auto",
    "trust_remote_code": True,
    "limit": None,
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


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return load_flat_yaml(path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Generation config must be a mapping: {path}")
    return data


def load_flat_yaml(path: Path) -> dict[str, Any]:
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


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if args.config:
        config.update(load_yaml(Path(args.config)))

    cli_overrides = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "dataset": args.dataset,
        "output": args.output,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "limit": args.limit,
    }
    config.update({key: value for key, value in cli_overrides.items() if value is not None})
    if not config.get("model"):
        raise ValueError("Missing required field: model. Provide it in --config or with --model.")
    return config


def get_torch_dtype(dtype_name: str):
    import torch

    dtype_name = str(dtype_name)
    if dtype_name in {"auto", "None", "none"}:
        return "auto"
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def prompt_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) < 3:
        raise ValueError("Each input row must contain at least system, user, and assistant messages.")
    return [
        {"role": str(message["role"]), "content": str(message["content"])}
        for message in messages[:-1]
    ]


def reference_text(row: dict[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    return str(messages[-1].get("content", ""))


def load_model_and_tokenizer(config: dict[str, Any]):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(config["model"]),
        trust_remote_code=bool(config["trust_remote_code"]),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(config["model"]),
        torch_dtype=get_torch_dtype(str(config["torch_dtype"])),
        device_map=str(config["device_map"]),
        trust_remote_code=bool(config["trust_remote_code"]),
    )

    adapter_path = config.get("adapter_path")
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, tokenizer


def generate_one(model, tokenizer, messages: list[dict[str, str]], config: dict[str, Any]) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(model.device) for key, value in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": int(config["max_new_tokens"]),
        "do_sample": bool(config["do_sample"]),
        "top_p": float(config["top_p"]),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if bool(config["do_sample"]):
        generation_kwargs["temperature"] = float(config["temperature"])

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    generated_ids = output_ids[0, inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def run_generation(config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = load_jsonl(Path(config["dataset"]))
    if config.get("limit") is not None:
        rows = rows[: int(config["limit"])]

    model, tokenizer = load_model_and_tokenizer(config)
    outputs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        messages = prompt_messages(row)
        prediction = generate_one(model, tokenizer, messages, config)
        outputs.append(
            {
                "prediction": prediction,
                "reference": reference_text(row),
                "metadata": row.get("metadata", {}),
                "prompt_messages": messages,
                "example_index": index,
            }
        )
        if (index + 1) % 10 == 0:
            print(f"Generated {index + 1}/{len(rows)} examples", flush=True)

    write_jsonl(outputs, Path(config["output"]))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ThoughtTrace predictions.")
    parser.add_argument("--config", help="Generation YAML config.")
    parser.add_argument("--model")
    parser.add_argument("--adapter-path")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--do-sample", action="store_true", default=None)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = build_config(args)
    outputs = run_generation(config)
    print(f"Wrote {len(outputs)} predictions to {config['output']}")


if __name__ == "__main__":
    main()
