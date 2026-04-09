"""Post-retrieval scoring helpers for multi-agent assembly.

Provides specificity scoring, near-miss detection, and diversity
policies.  All scorers are model-free and deterministic.
"""

from __future__ import annotations

import math

from ai_knot.tokenizer import tokenize as _tokenize
from ai_knot.types import Fact

# ---------------------------------------------------------------------------
# Generic cue phrases that indicate overview / near-miss content.
# Checked via substring match on lowercased fact content.
# ---------------------------------------------------------------------------
_GENERIC_CUE_PHRASES: tuple[str, ...] = (
    "overview",
    "conceptual level",
    "without implementation specifics",
    "without implementation detail",
    "general introduction",
    "at a high level",
    "broad perspective",
    "introductory guide",
    "conceptual understanding",
    "general purpose",
)

# Stopwords for technical-density calculation (stemmed).
_DENSITY_STOP_TOKENS = frozenset(
    stem
    for word in [
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "and",
        "but",
        "or",
        "not",
        "no",
    ]
    for stem in _tokenize(word)
)


class SpecificityScorer:
    """Score how implementation-specific a fact is (vs overview-like).

    Higher score = more specific / technical.

    Phase-1 algorithm:
    - ``technical_density = technical_tokens / total_tokens``
    - where technical tokens = non-stopword, stemmed tokens
    - Bonus for facts with populated slot_key (structured addressing).
    """

    def score(self, fact: Fact) -> float:
        """Return specificity score in [0.0, 1.0]."""
        tokens = _tokenize(fact.content)
        if not tokens:
            return 0.0

        technical = [t for t in tokens if t not in _DENSITY_STOP_TOKENS]
        density = len(technical) / len(tokens)

        # Bonus for structured facts (slot_key present → entity-addressed).
        slot_bonus = 0.1 if fact.slot_key else 0.0

        # Bonus for facts with evidence grounding.
        evidence_bonus = 0.05 if fact.source_snippets else 0.0

        return min(1.0, density + slot_bonus + evidence_bonus)


class NearMissDetector:
    """Detect overview / near-miss facts and assign a ranking penalty.

    Near-miss facts partially overlap query facets but lack implementation
    specifics.  They are penalised (not filtered) to allow the assembler
    to still use them as backfill when no better candidate exists.

    Phase-1 algorithm:
    - Check for generic cue phrases → base penalty 0.4
    - Compute technical density → low density adds up to 0.3 penalty
    - Combined penalty capped at 0.7 (never fully suppress)
    """

    def penalty(self, fact: Fact) -> float:
        """Return near-miss penalty in [0.0, 0.7]. Higher = more overview-like."""
        content_lower = fact.content.lower()
        penalty = 0.0

        # Generic cue phrase detection.
        for cue in _GENERIC_CUE_PHRASES:
            if cue in content_lower:
                penalty += 0.4
                break

        # Technical density check — low density = more generic.
        tokens = _tokenize(fact.content)
        if tokens:
            technical = [t for t in tokens if t not in _DENSITY_STOP_TOKENS]
            density = len(technical) / len(tokens)
            # Density < 0.5 → increasingly penalised (max 0.3 at density=0).
            if density < 0.5:
                penalty += 0.3 * (1.0 - density / 0.5)

        return min(0.7, penalty)


class DiversityPolicy:
    """Caps for per-agent and per-domain representation in results.

    Prevents a single agent or domain cluster from monopolising top-k.
    """

    def per_agent_cap(self, *, top_k: int, n_publishers: int = 3) -> int:
        """Maximum results from a single agent.

        Formula: ceil(top_k / n_publishers) + 1, minimum 1.
        Ensures at least n_publishers can contribute.
        """
        if n_publishers <= 1:
            return top_k
        return max(1, math.ceil(top_k / n_publishers) + 1)

    def per_domain_cap(self, *, top_k: int, n_facets: int = 3) -> int:
        """Maximum results covering the same facet.

        Formula: ceil(top_k / n_facets) + 1, minimum 2.
        Ensures each facet gets at least some representation.
        """
        if n_facets <= 1:
            return top_k
        return max(2, math.ceil(top_k / n_facets) + 1)
