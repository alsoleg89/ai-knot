"""Shared tokenization utilities for retrieval and extraction."""

from __future__ import annotations

import re

# Pre-compiled patterns.
_CAMEL_RE = re.compile(r"([a-z])([A-Z])")
_TOKEN_RE = re.compile(r"[^\W_]+")


def _stem(token: str) -> str:
    """Lightweight suffix stemmer (Porter 1980 step-1 subset).

    Handles common English suffixes without any hardcoded word lists.
    Rules are applied in order; first match wins.
    """
    if len(token) <= 3:
        return token

    # -ment → remove (deployment → deploy)
    if token.endswith("ment") and len(token) > 6:
        return token[:-4]

    # -tion / -sion → remove (creation → crea, but that's ok for matching)
    if (token.endswith("tion") or token.endswith("sion")) and len(token) > 6:
        return token[:-4]

    # -ing → remove (caching → cach, running → runn → run via double-consonant)
    if token.endswith("ing") and len(token) > 5:
        stem = token[:-3]
        # Handle doubled consonant: runn → run
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        return stem

    # -ed → remove (deployed → deploy, walked → walk)
    if token.endswith("ed") and len(token) > 4:
        stem = token[:-2]
        if len(stem) >= 2 and stem[-1] == stem[-2] and stem[-1] not in "aeiou":
            stem = stem[:-1]
        return stem

    # -ly → remove (quickly → quick)
    if token.endswith("ly") and len(token) > 4:
        return token[:-2]

    # -er → remove (faster → fast)
    if token.endswith("er") and len(token) > 4:
        return token[:-2]

    # -est → remove (fastest → fast)
    if token.endswith("est") and len(token) > 5:
        return token[:-3]

    # -s → remove (original rule, plural stripping)
    if token.endswith("s"):
        return token[:-1]

    return token


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens with stemming.

    Splits camelCase (``FastAPI`` → ``["fast", "api"]``), applies lightweight
    suffix stemming (Porter 1980 subset), and works with any Unicode script.

    Args:
        text: Input text to tokenize.

    Returns:
        List of normalized, stemmed tokens.
    """
    text = _CAMEL_RE.sub(r"\1 \2", text)
    tokens = _TOKEN_RE.findall(text.lower())
    return [_stem(t) for t in tokens]
