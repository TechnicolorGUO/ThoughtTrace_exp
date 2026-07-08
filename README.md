# ThoughtTrace_exp

Privileged distillation for user simulation in dialogue — using the SOC dataset to train models that predict a user's next utterance given conversation context.

## Structure

```
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── sft_4b.yaml
│   ├── sft_8b.yaml
│   ├── prompt.yaml
│   └── opsd_4b_4b.yaml
│
├── data/
│   ├── raw/                    # SOC raw data (gitignored)
│   ├── processed/
│   │   ├── train.json
│   │   └── test_260.json       # fixed test set (10%)
│   └── scripts/
│       ├── split_data.py
│       └── augment.py
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── prompts.py
│   ├── sft/
│   │   ├── train.py
│   │   └── inference.py
│   ├── prompt/
│   │   └── inference.py
│   ├── opsd/
│   │   ├── train.py
│   │   └── inference.py
│   └── eval/
│       ├── bleu.py
│       ├── embedding_sim.py
│       └── run_eval.py
│
├── scripts/
│   ├── run_sft.sh
│   ├── run_prompt.sh
│   └── run_opsd.sh
│
└── outputs/                    # checkpoints & logs (gitignored)
    ├── sft_4b/
    ├── sft_8b/
    ├── prompt/
    └── opsd_4b_4b/
```

## Setup

```bash
pip install -r requirements.txt
```

## Data

The ThoughtTrace dataset provides conversation context paired with user thoughts (reaction + motivation) and the user's next message.

- Raw data: place under `data/raw/` (not tracked by Git).
- Run `data/scripts/split_data.py` to produce `data/processed/train.json` and `data/processed/test_260.json`.

## Baselines

| Method | Description | Config |
|--------|-------------|--------|
| Prompt | Zero-shot — context → user message | `configs/prompt.yaml` |
| SFT    | Fine-tuned on (context → user message) | `configs/sft_4b.yaml` / `sft_8b.yaml` |
| OPSD   | Teacher sees user thought as privileged info; student does not. Both output user message only. | `configs/opsd_4b_4b.yaml` |

## Running

```bash
# SFT
bash scripts/run_sft.sh

# Prompt (zero-shot)
bash scripts/run_prompt.sh

# OPSD
bash scripts/run_opsd.sh
```

## Evaluation

Two metrics run on the fixed test set:

- **BLEU** — n-gram overlap via NLTK (`src/eval/bleu.py`)
- **Embedding similarity** — cosine similarity between generated and ground-truth embeddings (`src/eval/embedding_sim.py`)

```bash
# Evaluate all baselines
python src/eval/run_eval.py --outputs outputs/
```

## Results

| Method | Model Size | BLEU | Embedding Sim |
|--------|-----------|------|---------------|
| Prompt | 4B       | —    | —             |
| Prompt | 8B       | —    | —             |
| SFT    | 4B       | —    | —             |
| SFT    | 8B       | —    | —             |
| OPSD   | 4B→4B    | —    | —             |
| OPSD   | 8B→8B    | —    | —             |

## Todo

- [ ] OPSD 4B/8B formal experiments
- [ ] Scheme A: generate thought alongside user message
- [ ] Data augmentation (dialogue truncation, meta-data based)
- [ ] Extend to datasets without thought annotations

## License

TBD
