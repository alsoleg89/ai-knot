"""Shared tokenization utilities for retrieval and extraction."""

from __future__ import annotations

import re


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens with basic normalization.

    Splits camelCase (``FastAPI`` → ``["fast", "api"]``) and strips trailing
    ``"s"`` from tokens longer than 3 characters for basic plural handling.
    Also handles Cyrillic characters.

    Args:
        text: Input text to tokenize.

    Returns:
        List of normalized tokens.
    """
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    tokens = re.findall(r"[a-zA-Z0-9\u0400-\u04FF]+", text.lower())
    return [t[:-1] if t.endswith("s") and len(t) > 3 else t for t in tokens]
