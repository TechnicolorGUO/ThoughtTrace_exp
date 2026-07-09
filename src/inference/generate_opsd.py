"""Run inference with an OPSD/RLCSD checkpoint on the ThoughtTrace test set.

Companion to src/inference/generate.py, but built for the verl OPSD pipeline:

  * Loads a MERGED HuggingFace model dir (produce it first with
    scripts/merge_opsd_ckpt.sh from a verl global_step_*/actor checkpoint).
  * Reads the SAME test parquet the trainer uses
    (data/processed/user_sim_opsd_test.parquet) and drives generation from its
    `prompt` column, so the prompt construction exactly matches training. This
    matters: the student was trained on that role-play user prompt, so eval must
    reuse it rather than re-wrapping with a different template.
  * `reference` is taken from reward_model.ground_truth (the real next user
    message), kept for eval only.

Output records match generate.py so downstream eval (src/eval/run_eval.py) works
unchanged:
    {prediction, reference, metadata, prompt_messages, example_index}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_parquet_rows(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_parquet(path)
    return df.to_dict(orient="records")


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _to_message_list(prompt_field: Any) -> list[dict[str, str]]:
    """Normalize the parquet `prompt` cell into a list[{role, content}]."""
    import numpy as np

    if isinstance(prompt_field, np.ndarray):
        prompt_field = prompt_field.tolist()
    if not isinstance(prompt_field, (list, tuple)):
        raise ValueError(f"Unexpected prompt field type: {type(prompt_field)}")
    messages = []
    for msg in prompt_field:
        messages.append({"role": str(msg["role"]), "content": str(msg["content"])})
    return messages


def _reference(row: dict[str, Any]) -> str:
    rm = row.get("reward_model")
    if isinstance(rm, dict) and rm.get("ground_truth") is not None:
        return str(rm["ground_truth"])
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("reference") is not None:
        return str(extra["reference"])
    return ""


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    extra = row.get("extra_info")
    if isinstance(extra, dict):
        # keep it JSON-serializable and small
        return {k: extra[k] for k in ("uid", "split", "conversation_id",
                                      "assistant_message_id", "next_user_message_id")
                if k in extra}
    return {}


def load_model_and_tokenizer(model_path: str, dtype: str, device_map: str, trust_remote_code: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = "auto" if dtype in {"auto", "none", "None"} else getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer

def generate_one(model, tokenizer, messages: list[dict[str, str]], gen_kwargs: dict[str, Any]) -> str:
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)
    generated = output_ids[0, inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="OPSD/RLCSD checkpoint inference on ThoughtTrace test parquet.")
    parser.add_argument("--model", required=True, help="Merged HF model dir (from scripts/merge_opsd_ckpt.sh)")
    parser.add_argument("--dataset", default="data/processed/user_sim_opsd_test.parquet",
                        help="Test parquet with a `prompt` column (training-matched construction)")
    parser.add_argument("--output", default="outputs/opsd_predictions.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--torch-dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_parquet_rows(Path(args.dataset))
    if args.limit is not None:
        rows = rows[: args.limit]

    model, tokenizer = load_model_and_tokenizer(
        args.model, args.torch_dtype, args.device_map, args.trust_remote_code)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        gen_kwargs.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)

    outputs: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        messages = _to_message_list(row["prompt"])
        prediction = generate_one(model, tokenizer, messages, gen_kwargs)
        outputs.append({
            "prediction": prediction,
            "reference": _reference(row),
            "metadata": _metadata(row),
            "prompt_messages": messages,
            "example_index": index,
        })
        if (index + 1) % 10 == 0:
            print(f"Generated {index + 1}/{len(rows)} examples", flush=True)

    write_jsonl(outputs, Path(args.output))
    print(f"Wrote {len(outputs)} predictions to {args.output}")


if __name__ == "__main__":
    main()
