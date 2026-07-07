"""Small answer-normalization metrics for baseline tests."""

from __future__ import annotations

import re
import string
from collections import Counter


def exact_match(prediction: str, gold_answer: str) -> int:
    """Return 1 only for exact string equality."""
    return int(str(prediction) == str(gold_answer))


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation/articles, and normalize whitespace."""
    lowered = str(text).lower()
    without_punctuation = lowered.translate(str.maketrans("", "", string.punctuation))
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def normalized_exact_match(prediction: str, gold_answer: str) -> int:
    """Return 1 when normalized answers match exactly."""
    return int(normalize_answer(prediction) == normalize_answer(gold_answer))


def token_f1(prediction: str, gold_answer: str) -> float:
    """Return token-level F1 over normalized answer text."""
    prediction_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold_answer).split()
    if not prediction_tokens and not gold_tokens:
        return 1.0
    if not prediction_tokens or not gold_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
