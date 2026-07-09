# ThoughtTrace_exp

Privileged distillation for user simulation in dialogue. The project uses the ThoughtTrace dataset to train and evaluate models that predict a user's next utterance from the conversation context and the assistant's latest reply.

## Structure

```
├── README.md
│
├── configs/
│   ├── sft_qwen3_4b_lora_thought.yaml
│   ├── sft_qwen3_4b_lora_no_thought.yaml
│   ├── prompt_qwen3_4b.yaml
│   ├── prompt_qwen3_4b_vllm.yaml
│   ├── eval_qwen3_4b_lora_thought.yaml
│   ├── eval_qwen3_4b_lora_no_thought.yaml
│   ├── eval_qwen3_4b_prompt.yaml
│   └── eval_qwen3_4b_prompt_vllm.yaml
│
├── data/
│   ├── raw/
│   │   └── ThoughtTrace.jsonl
│   ├── processed/
│   │   ├── train_conversations.jsonl
│   │   ├── test_conversations.jsonl
│   │   ├── split_ids.json
│   │   ├── user_sim_train.jsonl
│   │   ├── user_sim_test.jsonl
│   │   ├── user_sim_no_thought_train.jsonl
│   │   └── user_sim_no_thought_test.jsonl
│   └── scripts/
│       ├── convert_to_swift_sft.py
│       └── convert_to_swift_sft_no_thought.py
│
├── scripts/
│   ├── train_qwen3_4b_lora.sh
│   ├── run_prompt_baseline.sh
│   ├── run_prompt_baseline_vllm.sh
│   └── eval.sh
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── prompts.py
│   ├── inference/
│   │   ├── generate.py
│   │   └── parse_outputs.py
│   └── eval/
│       ├── bleu.py
│       ├── embedding_sim.py
│       └── run_eval.py
│
└── outputs/
    ├── qwen3_4b_lora_thought/
    ├── qwen3_4b_lora_no_thought/
    └── qwen3_4b_prompt/
```

## Data

Put the raw ThoughtTrace file at:

```bash
data/raw/ThoughtTrace.jsonl
```

The conversion scripts first split the raw data at the conversation level, then construct `assistant -> user` prediction examples. This avoids leakage where turns from the same conversation appear in both train and test.

Generate the thought-augmented SFT data:

```bash
python data/scripts/convert_to_swift_sft.py
```

Generate the no-thought SFT data:

```bash
python data/scripts/convert_to_swift_sft_no_thought.py
```

Both scripts use the same default split:

```text
test_ratio: 0.10
seed: 42
```

The thought and no-thought SFT files are pair-level aligned. The no-thought version does not expose thought text in the input or output, but it uses the same examples as the thought version for a fair comparison.

### Datapoint Examples

All baselines are constructed from the same underlying prediction event:

```text
conversation history + assistant latest reply -> user's next message
```

The formats differ only in what supervision or privileged context is exposed.

No-thought SFT / prompt baseline datapoint:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are simulating a real user in a human-AI conversation. Given the conversation history and the assistant's latest reply, write the user's next message only."
    },
    {
      "role": "user",
      "content": "[Conversation History]\nUser: I want to find the better price for a new couch. How could I do it ?\n\n[Assistant Latest Reply]\nAssistant: A good way is to compare the total value, not just the sticker price. ..."
    },
    {
      "role": "assistant",
      "content": "step-by-step template please"
    }
  ],
  "metadata": {
    "conversation_id": "user12_task2_conversation1",
    "assistant_message_id": 1773849009461,
    "next_user_message_id": 1773849254132
  }
}
```

Thought SFT datapoint:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are simulating a real user in a human-AI conversation. Given the conversation history and the assistant's latest reply, first write the user's private thoughts, then write the user's next message. Follow the exact output format."
    },
    {
      "role": "user",
      "content": "[Conversation History]\nUser: I want to find the better price for a new couch. How could I do it ?\n\n[Assistant Latest Reply]\nAssistant: A good way is to compare the total value, not just the sticker price. ..."
    },
    {
      "role": "assistant",
      "content": "<thought>\n[Reaction]: I like that it propose advices regarding a variety of criterias.\n[Motivation]: I want to see the format of the template proposed.\n</thought>\n<reply>\nstep-by-step template please\n</reply>"
    }
  ]
}
```

OPSD datapoint:

```json
{
  "problem": "[Conversation History]\nUser: I want to find the better price for a new couch. How could I do it ?\n\n[Assistant Latest Reply]\nAssistant: A good way is to compare the total value, not just the sticker price. ...",
  "solution": "[Reaction]: I like that it propose advices regarding a variety of criterias.\n[Motivation]: I want to see the format of the template proposed.",
  "reply": "step-by-step template please"
}
```

In this project, the OPSD field `solution` means `user_thought`. The intended privileged setup is:

```text
student input = problem
teacher input = problem + user_thought
```

So `problem` is the only information visible to the student. The teacher additionally receives `solution`, which contains the user's private ThoughtTrace annotations (`[Reaction]` + `[Motivation]`). The `reply` field is the ground-truth next user message and is kept as reference metadata; it should not be appended to either the student or teacher prompt during on-policy distillation.

OPD parquet datapoint:

```json
{
  "data_source": "thoughttrace",
  "prompt": [
    {
      "role": "system",
      "content": "You are simulating a real user in a human-AI conversation..."
    },
    {
      "role": "user",
      "content": "[Conversation History]\nUser: ...\n\n[Assistant Latest Reply]\nAssistant: ..."
    }
  ],
  "ability": "user_simulation",
  "reward_model": {
    "style": "reference",
    "ground_truth": "step-by-step template please"
  },
  "extra_info": {
    "split": "train",
    "reference": "step-by-step template please"
  }
}
```

OPD does not expose ThoughtTrace private thoughts. The teacher is a separate model that distills its behavior on the same no-thought prompt.

## Baselines

| Method | Description | Train / Generation Config | Eval Config |
|--------|-------------|---------------------------|-------------|
| Prompt | Base Qwen3-4B, zero-shot prompt inference | `configs/prompt_qwen3_4b.yaml` | `configs/eval_qwen3_4b_prompt.yaml` |
| Prompt vLLM | Same prompt baseline, batched vLLM inference | `configs/prompt_qwen3_4b_vllm.yaml` | `configs/eval_qwen3_4b_prompt_vllm.yaml` |
| No-thought SFT | LoRA SFT, context + assistant reply -> user reply | `configs/sft_qwen3_4b_lora_no_thought.yaml` | `configs/eval_qwen3_4b_lora_no_thought.yaml` |
| Thought SFT | LoRA SFT, context + assistant reply -> thought + user reply | `configs/sft_qwen3_4b_lora_thought.yaml` | `configs/eval_qwen3_4b_lora_thought.yaml` |

## Training

Activate the server environment:

```bash
useenv swift
cd /autodl-fs/data/beichen/projects/ThoughtTrace_exp
```

Train the thought SFT model:

```bash
bash scripts/train_qwen3_4b_lora.sh
```

Train the no-thought SFT model:

```bash
bash scripts/train_qwen3_4b_lora.sh configs/sft_qwen3_4b_lora_no_thought.yaml
```

LoRA outputs are written to:

```text
outputs/qwen3_4b_lora_thought/
outputs/qwen3_4b_lora_no_thought/
```

The training script reads the YAML config and expands it into `swift sft --key value` arguments for compatibility with the server's ms-swift version.

## Prompt Baseline

The prompt baseline uses the same Qwen3-4B base model without LoRA training.

Run with Hugging Face Transformers:

```bash
bash scripts/run_prompt_baseline.sh
```

Run with vLLM batched inference:

```bash
bash scripts/run_prompt_baseline_vllm.sh
```

Quick smoke test on 5 examples:

```bash
bash scripts/run_prompt_baseline.sh configs/prompt_qwen3_4b.yaml --limit 5
bash scripts/run_prompt_baseline_vllm.sh configs/prompt_qwen3_4b_vllm.yaml --limit 5
```

Predictions are written to:

```text
outputs/qwen3_4b_prompt/predictions.jsonl
outputs/qwen3_4b_prompt_vllm/predictions.jsonl
```

Each prediction record contains:

```json
{
  "prediction": "model output",
  "reference": "ground-truth user next message",
  "metadata": {},
  "prompt_messages": [],
  "example_index": 0
}
```

## Evaluation

Evaluation expects prediction JSONL files with `prediction` and `reference` fields.

Metrics:

- **BLEU**: corpus-level n-gram overlap between predicted and reference user messages.
- **Embedding similarity**: cosine similarity between sentence embeddings of predicted and reference messages.

Evaluate the prompt baseline:

```bash
bash scripts/eval.sh configs/eval_qwen3_4b_prompt.yaml
```

Evaluate the thought SFT model:

```bash
bash scripts/eval.sh configs/eval_qwen3_4b_lora_thought.yaml
```

Evaluate the no-thought SFT model:

```bash
bash scripts/eval.sh configs/eval_qwen3_4b_lora_no_thought.yaml
```

If `sentence-transformers` is not installed, run BLEU-only evaluation:

```bash
bash scripts/eval.sh configs/eval_qwen3_4b_prompt.yaml --skip-embedding
```

For thought-model outputs, evaluation extracts only the `<reply>...</reply>` span before comparing against the reference. No-thought and prompt baselines use the full generated text as the predicted user reply.

## Results

| Method | Model | Examples | BLEU | Embedding Sim | Notes |
|--------|-------|----------|------|---------------|-------|
| Prompt vLLM | Qwen3-4B | 339 | 0.0028 | 0.2438 | `max_new_tokens=256`; generated replies are much longer than references |
| Prompt | Qwen3-4B | — | — | — | HF Transformers version, same prompt format |
| No-thought SFT | Qwen3-4B + LoRA | — | — | — | Pending |
| Thought SFT | Qwen3-4B + LoRA | — | — | — | Pending |

Prompt vLLM token stats:

```json
{
  "prediction_tokens": 86710.0,
  "reference_tokens": 7000.0,
  "precision_1": 0.0379,
  "precision_2": 0.0054,
  "precision_3": 0.0010,
  "precision_4": 0.0003
}
```

## Todo

- [ ] Generate predictions for trained LoRA checkpoints with `src/inference/generate.py`.
- [ ] Add OPSD / privileged-distillation experiments.
- [ ] Add longer-context ablations such as `max_length=8192`.
- [ ] Add human or LLM-judge evaluation for semantic and behavioral quality.

## License

TBD
