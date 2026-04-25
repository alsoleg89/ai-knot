"""Lexicon loader — language-pluggable vocabulary for the v2 bridge.

A lexicon is a JSON file at ``bench/lexicons/<lang>.json`` with this schema::

    {
      "_meta": {"language": "en", ...},
      "aggregation_tokens": ["list", "all", ...],
      "aggregation_phrases": ["how many", ...],
      "temporal_tokens": ["when", "before", ...],
      "single_fact_phrases": ["what is", ...],
      "stopwords_cap": ["What", "When", ...],
      "factual_predicates": ["is", "has", ...],
      "common_words_path": "/usr/share/dict/words"
    }

Adding a new language: drop in ``ru.json`` (or any code) with the same keys.
The bridge picks it up via ``--lang ru``. No code changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_LEX_DIR = Path(__file__).parent / "lexicons"


@dataclass(frozen=True)
class Lexicon:
    language: str
    aggregation_tokens: frozenset[str]
    aggregation_phrases: tuple[str, ...]
    temporal_tokens: frozenset[str]
    single_fact_phrases: tuple[str, ...]
    stopwords_cap: frozenset[str]
    factual_predicates: frozenset[str]
    common_words: frozenset[str] = field(default_factory=frozenset)


def _load_common_words(path_str: str) -> frozenset[str]:
    if not path_str:
        return frozenset()
    p = Path(path_str)
    if not p.exists():
        return frozenset()
    try:
        return frozenset(w.strip().lower() for w in p.read_text().splitlines() if w.strip())
    except OSError:
        return frozenset()


@lru_cache(maxsize=8)
def load_lexicon(lang: str = "en") -> Lexicon:
    """Load lexicon for the given language code. Cached."""
    path = _LEX_DIR / f"{lang}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Lexicon not found: {path}. Available: "
            f"{[p.stem for p in _LEX_DIR.glob('*.json')]}"
        )
    with open(path) as f:
        data = json.load(f)
    return Lexicon(
        language=str(data.get("_meta", {}).get("language", lang)),
        aggregation_tokens=frozenset(data.get("aggregation_tokens", [])),
        aggregation_phrases=tuple(data.get("aggregation_phrases", [])),
        temporal_tokens=frozenset(data.get("temporal_tokens", [])),
        single_fact_phrases=tuple(data.get("single_fact_phrases", [])),
        stopwords_cap=frozenset(data.get("stopwords_cap", [])),
        factual_predicates=frozenset(data.get("factual_predicates", [])),
        common_words=_load_common_words(data.get("common_words_path", "")),
    )


def list_available_languages() -> list[str]:
    return sorted(p.stem for p in _LEX_DIR.glob("*.json"))
