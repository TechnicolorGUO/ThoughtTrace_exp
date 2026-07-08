"""BLEU evaluation for ThoughtTrace user-message predictions.

This module will compute n-gram overlap between generated user replies and
ground-truth next user messages, with tokenization and smoothing choices kept
consistent across baselines.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Tokenize text for lightweight BLEU evaluation."""
    return TOKEN_PATTERN.findall(str(text or "").lower())


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    predictions: Iterable[str],
    references: Iterable[str],
    *,
    max_order: int = 4,
    smooth: bool = True,
) -> dict[str, float]:
    """Compute corpus BLEU with clipped n-gram precision."""
    predictions = list(predictions)
    references = list(references)
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    matches = [0] * max_order
    possible = [0] * max_order
    pred_len = 0
    ref_len = 0

    for prediction, reference in zip(predictions, references):
        pred_tokens = tokenize(prediction)
        ref_tokens = tokenize(reference)
        pred_len += len(pred_tokens)
        ref_len += len(ref_tokens)

        for order in range(1, max_order + 1):
            pred_ngrams = ngrams(pred_tokens, order)
            ref_ngrams = ngrams(ref_tokens, order)
            overlap = pred_ngrams & ref_ngrams
            matches[order - 1] += sum(overlap.values())
            possible[order - 1] += max(len(pred_tokens) - order + 1, 0)

    precisions: list[float] = []
    for match_count, possible_count in zip(matches, possible):
        if smooth:
            precisions.append((match_count + 1.0) / (possible_count + 1.0))
        elif possible_count == 0:
            precisions.append(0.0)
        else:
            precisions.append(match_count / possible_count)

    if pred_len == 0:
        bleu = 0.0
        brevity_penalty = 0.0
    else:
        brevity_penalty = 1.0 if pred_len > ref_len else math.exp(1.0 - ref_len / pred_len)
        if min(precisions) <= 0:
            bleu = 0.0
        else:
            bleu = brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_order)

    return {
        "bleu": bleu,
        "bleu_percent": bleu * 100.0,
        "brevity_penalty": brevity_penalty,
        "prediction_tokens": float(pred_len),
        "reference_tokens": float(ref_len),
        **{f"precision_{i + 1}": precisions[i] for i in range(max_order)},
    }
