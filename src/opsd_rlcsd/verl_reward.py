"""Dummy reward for ThoughtTrace OPSD.

ThoughtTrace next-message prediction has no unique exact-match reward like
math boxed-answer tasks. The first OPSD version uses teacher-student
distillation only, so task reward is intentionally neutral.
"""

from __future__ import annotations


def compute_reward(responses: list[str], ground_truths: list[str]) -> list[float]:
    del responses, ground_truths
    return [0.0 for _ in ground_truths]
