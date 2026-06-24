"""Conjunctive facet decomposition for multi-source pool queries.

Splits queries like "integrate X, Y, and Z into a unified pipeline"
into independent retrieval facets.  Each facet is searched separately
and results are assembled via coverage-aware selection.

Independence guardrails prevent false-positive decomposition:
- Each clause must contain enough content-bearing tokens (>= 2).
- Explanatory tails ("how does it work") stay single-facet.
- Ambiguous decompositions fall back to the original query.
"""

from __future__ import annotations

import re

from ai_knot.multi_agent.models import QueryFacet, RoutedPoolQuery
from ai_knot.tokenizer import tokenize as _tokenize

# Minimum content-bearing tokens per facet clause to be considered independent.
_MIN_CONTENT_TOKENS = 2

# Stopwords / common verbs that don't count as content-bearing.
# Kept small and targeted — stemmed forms only.
_STOP_TOKENS = frozenset(
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
        "being",
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
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "what",
        "which",
        "who",
        "whom",
        "how",
        "why",
        "when",
        "where",
        "not",
        "no",
        "nor",
        "and",
        "but",
        "or",
        "so",
        "if",
        "because",
        "about",
        "up",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "only",
        "same",
        "than",
        "too",
        "very",
        "just",
        "also",
    ]
    for stem in _tokenize(word)
)

# Meta-explanatory patterns — clauses matching these are NOT independent facets.
_META_PATTERNS = [
    re.compile(r"^how\b.*\b(work|function|operat|behav)", re.IGNORECASE),
    re.compile(r"^why\b.*\b(important|useful|need|matter)", re.IGNORECASE),
    re.compile(r"^what\b.*\b(is|are|mean)", re.IGNORECASE),
    re.compile(r"^(explain|describe|tell me about)\b", re.IGNORECASE),
]

# Conjunction splitting patterns — ordered by specificity.
# Pattern 1: "X, Y, and Z" or "X, Y and Z"
_COMMA_AND_RE = re.compile(r",\s*(?:and\s+)?", re.IGNORECASE)
# Pattern 2: standalone "and" between clauses (only for long queries)
_AND_RE = re.compile(r"\band\b", re.IGNORECASE)


def _content_tokens(text: str) -> tuple[str, ...]:
    """Return only content-bearing stemmed tokens (no stopwords)."""
    return tuple(t for t in _tokenize(text) if t not in _STOP_TOKENS)


def _is_meta_explanatory(clause: str) -> bool:
    """Return True if clause is an explanatory tail, not a retrieval target."""
    stripped = clause.strip()
    return any(p.search(stripped) for p in _META_PATTERNS)


def _extract_framing(query: str) -> tuple[str, str]:
    """Split off a framing prefix/suffix from the query.

    Queries like "How can a system integrate X, Y, and Z into a unified pipeline?"
    have a framing prefix ("How can a system integrate") and suffix ("into a
    unified pipeline") that are NOT facets.

    Returns:
        (framing_context, core_clause) where core_clause contains the facets.
    """
    # Common framing patterns that wrap around the facet list.
    patterns = [
        # "How can/does X integrate A, B, and C into/for Y?"
        re.compile(
            r"^(.*?\b(?:integrat|combin|unif|merg|assembl|incorporat)\w*)\s+"
            r"(.*?)\s+"
            r"(into\s+.*|for\s+.*|within\s+.*)?$",
            re.IGNORECASE,
        ),
        # "What approach combines A, B, and C?"
        re.compile(
            r"^(.*?\b(?:approach|method|way|strategy|technique)\w*\s+\w+\s+)"
            r"(.*?)(\??)$",
            re.IGNORECASE,
        ),
    ]
    for p in patterns:
        m = p.match(query)
        if m:
            prefix = m.group(1).strip()
            core = m.group(2).strip()
            suffix = (m.group(3) or "").strip()
            framing = f"{prefix} ... {suffix}".strip()
            if core:
                return framing, core

    return "", query


class ConjunctiveFacetPlanner:
    """Decompose conjunctive multi-source queries into independent facets.

    Only decomposes when each candidate clause has enough content-bearing
    tokens.  Falls back to a single whole-query facet when decomposition
    is ambiguous.
    """

    def decompose(self, routed: RoutedPoolQuery) -> tuple[QueryFacet, ...]:
        """Split a routed query into independent facets.

        Args:
            routed: The routed pool query (intent already classified).

        Returns:
            Tuple of QueryFacet instances. Always contains at least one
            facet (the whole query as fallback).
        """
        query = routed.raw_query

        # Strip framing to isolate the facet list.
        _framing, core = _extract_framing(query)
        # Remove trailing punctuation.
        core = re.sub(r"[?.!]+$", "", core).strip()

        # Try comma-and splitting first (most explicit signal).
        clauses = _COMMA_AND_RE.split(core)
        clauses = [c.strip() for c in clauses if c.strip()]

        # If comma splitting produced < 2 clauses, try "and" splitting
        # but only for longer queries (> 8 words in core).
        if len(clauses) < 2 and len(core.split()) > 8:
            clauses = _AND_RE.split(core)
            clauses = [c.strip() for c in clauses if c.strip()]

        # Apply independence guardrails.
        valid_clauses: list[tuple[str, tuple[str, ...]]] = []
        for clause in clauses:
            # Reject meta-explanatory clauses.
            if _is_meta_explanatory(clause):
                continue
            tokens = _content_tokens(clause)
            if len(tokens) >= _MIN_CONTENT_TOKENS:
                valid_clauses.append((clause, tokens))

        # Need at least 2 valid clauses for decomposition to be worthwhile.
        if len(valid_clauses) < 2:
            tokens = tuple(_tokenize(query))
            return (
                QueryFacet(
                    facet_id="f0",
                    text=query,
                    tokens=tokens,
                    facet_type="general",
                ),
            )

        # Cap at 5 facets per plan spec.
        valid_clauses = valid_clauses[:5]

        facets: list[QueryFacet] = []
        for i, (clause_text, clause_tokens) in enumerate(valid_clauses):
            facets.append(
                QueryFacet(
                    facet_id=f"f{i}",
                    text=clause_text,
                    tokens=clause_tokens,
                    facet_type="domain",
                )
            )

        return tuple(facets)
