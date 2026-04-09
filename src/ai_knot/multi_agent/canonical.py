"""Canonical claim family resolution for shared-pool retrieval.

Groups candidate facts into claim families using IDF-weighted content-token
overlap, then resolves competing claims before the final top-k cutoff.

IDF is computed over the candidate set itself — no keyword lists or domain
vocabulary.  High-IDF tokens (rare in the candidate pool) carry more weight
than generic operational terms shared by many facts.  This discriminates
between same-entity competing claims (share rare entity identifiers → high
weighted overlap) and different-entity complementary claims (share only
common operational vocabulary → low weighted overlap).
"""

from __future__ import annotations

import math
from collections.abc import Callable

from ai_knot.tokenizer import tokenize as _tokenize
from ai_knot.types import Fact

# ---------------------------------------------------------------------------
# Conflict-signal stems — words that indicate one fact explicitly supersedes
# or contradicts another (e.g. policy updates, deprecations, corrections).
# Canonical resolution is only applied to clusters that contain at least one
# fact with one of these stems; clusters of complementary facts (same topic,
# different scope) are kept intact.
# ---------------------------------------------------------------------------
_CONFLICT_SIGNAL_STEMS: frozenset[str] = frozenset(
    stem
    for kw in {
        "deprecated",
        "tightened",
        "expanded",
        "revised",
        "superseded",
        "replaced",
        "obsolete",
        "updated",
        "removed",
        "changed",
    }
    for stem in _tokenize(kw)
)

# ---------------------------------------------------------------------------
# Stopwords shared with scoring.py — kept local to avoid circular import.
# ---------------------------------------------------------------------------
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


def _content_tokens(text: str) -> frozenset[str]:
    """Return stemmed content-bearing tokens (stopwords excluded)."""
    return frozenset(t for t in _tokenize(text) if t not in _STOP_TOKENS)


def _build_idf(token_sets: list[frozenset[str]]) -> dict[str, float]:
    """Compute IDF for each token over the candidate set.

    IDF(t) = log((N + 1) / (df(t) + 1)) + 1  (smoothed, always positive).

    Tokens that appear in many candidate facts get low IDF (generic terms).
    Tokens that appear in few candidate facts get high IDF (specific entities).
    """
    n = len(token_sets)
    df: dict[str, int] = {}
    for ts in token_sets:
        for t in ts:
            df[t] = df.get(t, 0) + 1
    return {t: math.log((n + 1) / (count + 1)) + 1.0 for t, count in df.items()}


def _idf_weighted_overlap(
    a: frozenset[str],
    b: frozenset[str],
    idf: dict[str, float],
) -> float:
    """IDF-weighted overlap coefficient.

    = Σ IDF(t) for t in A∩B  /  Σ IDF(t) for t in the smaller set

    Generic terms shared by many facts (low IDF) contribute little.
    Specific entity identifiers shared by few facts (high IDF) contribute a lot.

    Returns 0.0 if either set is empty.
    """
    if not a or not b:
        return 0.0
    intersection = a & b
    if not intersection:
        return 0.0
    # Denominator: sum of IDF over the smaller set (overlap coefficient style).
    smaller = a if len(a) <= len(b) else b
    denom = sum(idf.get(t, 1.0) for t in smaller)
    if denom == 0.0:
        return 0.0
    numer = sum(idf.get(t, 1.0) for t in intersection)
    return numer / denom


class ClaimFamilyResolver:
    """Pre-ranking canonical resolution using IDF-weighted content-similarity grouping.

    Algorithm
    ---------
    1. Compute content-bearing tokens for each candidate fact.
    2. Build per-token IDF over the candidate set.
       Tokens rare in the candidate pool (specific entity identifiers) get high
       IDF; tokens common across many facts (generic operational terms) get low IDF.
    3. Build pairwise IDF-weighted overlap matrix.
       This discriminates same-entity competing claims (share rare entity tokens
       → high score) from different-entity complementary claims (share only
       common vocabulary → low score).
    4. Derive a grouping threshold from the distribution of pairwise similarities
       in the candidate set (median of non-zero pairs, floored at 0.35).
    5. Union-find clustering: two facts are in the same family when their
       IDF-weighted overlap >= threshold.
    6. Resolve each cluster:
       - canonical_mode: keep only the authoritative winner
         (slot_key fact > most-trusted-and-newest unslotted fact).
       - non-canonical mode: deduplicate near-identical claims (high
         similarity >= 0.8), keep multi-viewpoint evidence otherwise.
    """

    # Hard floor: never group facts whose weighted token overlap is below this.
    _FLOOR: float = 0.35
    # Near-duplicate threshold for non-canonical deduplication.
    _NEAR_DUPLICATE: float = 0.80

    def resolve(
        self,
        candidates: list[tuple[Fact, float]],
        *,
        canonical_mode: bool,
        get_trust: Callable[[str], float] | None = None,
    ) -> list[tuple[Fact, float]]:
        """Group candidates by claim family and resolve competing claims.

        Args:
            candidates: (Fact, score) pairs in any order.
            canonical_mode: When True, keep only the authoritative winner
                per family. When False, deduplicate near-duplicates only.
            get_trust: Optional trust function for winner selection.
                Falls back to recency-only when not provided.

        Returns:
            Filtered list of (Fact, score) pairs.
        """
        if len(candidates) <= 1:
            return candidates

        token_sets = [_content_tokens(f.content) for f, _ in candidates]
        n = len(candidates)

        # IDF over the candidate corpus — discriminates entity tokens from
        # generic operational vocabulary.
        idf = _build_idf(token_sets)

        # Compute pairwise similarities for union-find clustering.
        sim_matrix: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                sim_matrix[(i, j)] = _idf_weighted_overlap(token_sets[i], token_sets[j], idf)

        # Threshold: always the hard floor.
        # A dynamic median-based threshold inflates when many candidates share
        # common vocabulary (e.g. "endpoint", "collector", "REST"), pushing the
        # threshold well above 0.35 and preventing same-entity competing claims
        # from being grouped.  Using the floor gives consistent grouping
        # behaviour across pool sizes and vocabulary distributions.
        threshold = self._FLOOR

        # For non-canonical mode, only deduplicate near-duplicates.
        if not canonical_mode:
            threshold = max(threshold, self._NEAR_DUPLICATE)

        # Union-find clustering.
        parent = list(range(n))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(x: int, y: int) -> None:
            rx, ry = _find(x), _find(y)
            if rx != ry:
                parent[rx] = ry

        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix.get((i, j), 0.0) >= threshold:
                    _union(i, j)

        # Group by cluster root.
        clusters: dict[int, list[int]] = {}
        for i in range(n):
            root = _find(i)
            clusters.setdefault(root, []).append(i)

        # Resolve each cluster.
        kept: list[tuple[Fact, float]] = []
        for _root, members in clusters.items():
            if len(members) == 1:
                kept.append(candidates[members[0]])
                continue

            member_pairs = [candidates[i] for i in members]

            if canonical_mode:
                # Only apply canonical resolution when at least one fact in the
                # cluster contains an explicit conflict/update signal (deprecated,
                # tightened, expanded, etc.).  Clusters of complementary facts
                # that share topic vocabulary but different scopes (e.g. two
                # deployment facts about FluxCD from different teams) are kept
                # intact — both facts carry useful information.
                cluster_tokens = frozenset(
                    tok for f, _ in member_pairs for tok in _tokenize(f.content)
                )
                if not (cluster_tokens & _CONFLICT_SIGNAL_STEMS):
                    # No conflict signal — complementary facts, keep all.
                    kept.extend(member_pairs)
                    continue

            if canonical_mode:
                winner_fact, winner_score = self._canonical_winner(
                    member_pairs, get_trust=get_trust
                )
                # Promote the winner's score to the max score in the cluster.
                # Rationale: BM25 scores across the cluster reflect how relevant
                # the shared topic is to the query — a competing claim with high
                # BM25 signals that the query is specifically about this fact
                # family.  After eliminating competitors, the authoritative winner
                # should carry that full relevance signal; otherwise trust-discount
                # applied later can push the correct answer below unrelated facts
                # from higher-trust agents.
                max_cluster_score = max(s for _, s in member_pairs)
                kept.append((winner_fact, max(winner_score, max_cluster_score)))
            else:
                # Non-canonical: all members pass (only near-duplicates were
                # grouped at the higher threshold), keep the highest-scored.
                kept.append(max(member_pairs, key=lambda x: x[1]))

        return kept

    @staticmethod
    def _canonical_winner(
        members: list[tuple[Fact, float]],
        *,
        get_trust: Callable[[str], float] | None,
    ) -> tuple[Fact, float]:
        """Select the authoritative fact from a cluster.

        Priority:
        1. Slotted fact (CAS-addressed, highest explicit version wins).
        2. Most-trusted × most-recent unslotted fact.
        3. Tie-break: highest retrieval score.
        """
        slotted = [(f, s) for f, s in members if f.slot_key]
        if slotted:
            if len(slotted) == 1:
                return slotted[0]
            # Among slotted: highest version wins; fall back to score.
            return max(slotted, key=lambda x: (x[0].version, x[1]))

        def _rank(pair: tuple[Fact, float]) -> tuple[float, float, float]:
            f, score = pair
            trust = get_trust(f.origin_agent_id) if (get_trust and f.origin_agent_id) else 1.0
            recency = f.created_at.timestamp()
            return trust, recency, score

        return max(members, key=_rank)
