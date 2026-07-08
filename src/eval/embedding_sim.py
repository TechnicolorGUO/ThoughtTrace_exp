"""Embedding similarity evaluation for ThoughtTrace predictions.

This module will compute semantic similarity between generated user replies
and references using sentence embeddings and cosine similarity.
"""

from __future__ import annotations

from typing import Iterable


def cosine_similarity_rows(left, right) -> list[float]:
    """Compute row-wise cosine similarity for two embedding matrices."""
    import numpy as np

    left = np.asarray(left)
    right = np.asarray(right)
    numerators = (left * right).sum(axis=1)
    denominators = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    denominators = np.maximum(denominators, 1e-12)
    return (numerators / denominators).tolist()


def embedding_similarity(
    predictions: Iterable[str],
    references: Iterable[str],
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
) -> dict[str, float]:
    """Compute mean sentence-embedding cosine similarity."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Embedding similarity requires sentence-transformers. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    predictions = [str(item or "") for item in predictions]
    references = [str(item or "") for item in references]
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        return {"embedding_similarity": 0.0, "num_examples": 0.0}

    model = SentenceTransformer(model_name)
    pred_embeddings = model.encode(
        predictions,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    ref_embeddings = model.encode(
        references,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    )
    scores = cosine_similarity_rows(pred_embeddings, ref_embeddings)
    return {
        "embedding_similarity": sum(scores) / len(scores),
        "embedding_similarity_min": min(scores),
        "embedding_similarity_max": max(scores),
        "num_examples": float(len(scores)),
    }
