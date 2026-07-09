"""Run vLLM inference on processed ThoughtTrace test data.

The output schema matches `src/inference/generate.py`, so downstream evaluation
can consume predictions from either Hugging Face Transformers or vLLM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from generate import (
    DEFAULT_CONFIG,
    load_jsonl,
    load_yaml,
    prompt_messages,
    reference_text,
    write_jsonl,
)


VLLM_DEFAULTS = {
    **DEFAULT_CONFIG,
    "output": "outputs/qwen3_4b_prompt_vllm/predictions.jsonl",
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.85,
    "batch_size": 32,
    "max_model_len": None,
}


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(VLLM_DEFAULTS)
    if args.config:
        config.update(load_yaml(Path(args.config)))

    cli_overrides = {
        "model": args.model,
        "dataset": args.dataset,
        "output": args.output,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "limit": args.limit,
        "tensor_parallel_size": args.tensor_parallel_size,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "batch_size": args.batch_size,
        "max_model_len": args.max_model_len,
    }
    config.update({key: value for key, value in cli_overrides.items() if value is not None})
    if not config.get("model"):
        raise ValueError("Missing required field: model. Provide it in --config or with --model.")
    if config.get("adapter_path"):
        raise ValueError("vLLM prompt baseline does not support adapter_path here. Use generate.py for LoRA adapters.")
    return config


def build_prompts(tokenizer: Any, rows: list[dict[str, Any]]) -> list[list[dict[str, str]]]:
    return [prompt_messages(row) for row in rows]


def render_prompts(tokenizer: Any, messages_batch: list[list[dict[str, str]]]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in messages_batch
    ]


def run_generation(config: dict[str, Any]) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    rows = load_jsonl(Path(config["dataset"]))
    if config.get("limit") is not None:
        rows = rows[: int(config["limit"])]

    llm_kwargs = {
        "model": str(config["model"]),
        "tensor_parallel_size": int(config["tensor_parallel_size"]),
        "gpu_memory_utilization": float(config["gpu_memory_utilization"]),
        "dtype": str(config.get("torch_dtype", "bfloat16")),
        "trust_remote_code": bool(config.get("trust_remote_code", True)),
    }
    if config.get("max_model_len") is not None:
        llm_kwargs["max_model_len"] = int(config["max_model_len"])

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    sampling_kwargs = {
        "max_tokens": int(config["max_new_tokens"]),
        "temperature": float(config["temperature"]) if bool(config.get("do_sample")) else 0.0,
        "top_p": float(config["top_p"]),
    }
    sampling_params = SamplingParams(**sampling_kwargs)

    outputs: list[dict[str, Any]] = []
    batch_size = int(config["batch_size"])
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        messages_batch = build_prompts(tokenizer, batch_rows)
        prompts = render_prompts(tokenizer, messages_batch)
        generations = llm.generate(prompts, sampling_params)

        for offset, (row, messages, generation) in enumerate(zip(batch_rows, messages_batch, generations)):
            prediction = generation.outputs[0].text.strip() if generation.outputs else ""
            outputs.append(
                {
                    "prediction": prediction,
                    "reference": reference_text(row),
                    "metadata": row.get("metadata", {}),
                    "prompt_messages": messages,
                    "example_index": start + offset,
                }
            )

        print(f"Generated {min(start + batch_size, len(rows))}/{len(rows)} examples", flush=True)

    write_jsonl(outputs, Path(config["output"]))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ThoughtTrace predictions with vLLM.")
    parser.add_argument("--config", help="Generation YAML config.")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--do-sample", action="store_true", default=None)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--gpu-memory-utilization", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-model-len", type=int)
    args = parser.parse_args()

    config = build_config(args)
    outputs = run_generation(config)
    print(f"Wrote {len(outputs)} predictions to {config['output']}")


if __name__ == "__main__":
    main()
