"""Agent expertise index for route-before-retrieve in MULTI_SOURCE queries.

Builds per-agent profiles from published pool facts, enabling fast
narrowing of the search space before BM25 retrieval.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from ai_knot.tokenizer import tokenize as _tokenize
from ai_knot.types import Fact


@dataclass(slots=True)
class ExpertiseProfile:
    """Aggregated expertise profile for a single agent."""

    agent_id: str
    domains: Counter[str] = field(default_factory=Counter)
    tags: Counter[str] = field(default_factory=Counter)
    canonical_terms: Counter[str] = field(default_factory=Counter)
    content_terms: Counter[str] = field(default_factory=Counter)
    published_facts: int = 0
    trust_score: float = 0.8


@dataclass(slots=True)
class ExpertiseHit:
    """A ranked agent result from the expertise index."""

    agent_id: str
    score: float
    matched_terms: tuple[str, ...]


class AgentExpertiseIndex:
    """In-memory index of agent expertise derived from active pool facts.

    Cache lifecycle:
    - Build lazily on first MULTI_SOURCE recall.
    - Cache with a version marker (pool fact count + max version).
    - Invalidate when pool changes (publish, CAS, promote, GC).
    - Do NOT rebuild on every recall.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, ExpertiseProfile] = {}
        self._cache_version: int = 0
        self._built: bool = False

    @property
    def built(self) -> bool:
        """Whether the index has been built at least once."""
        return self._built

    @property
    def profiles(self) -> dict[str, ExpertiseProfile]:
        """Read-only access to current profiles."""
        return self._profiles

    def build(
        self,
        active_facts: list[Fact],
        get_trust: Callable[[str], float],
    ) -> None:
        """Build expertise profiles from active pool facts.

        Groups facts by ``origin_agent_id`` and accumulates token counts
        for content, canonical surfaces, tags, and entity domains.
        """
        self._profiles.clear()

        # Group facts by origin agent.
        by_agent: dict[str, list[Fact]] = {}
        for fact in active_facts:
            agent_id = fact.origin_agent_id or "unknown"
            by_agent.setdefault(agent_id, []).append(fact)

        for agent_id, facts in by_agent.items():
            profile = ExpertiseProfile(agent_id=agent_id)
            profile.published_facts = len(facts)
            profile.trust_score = get_trust(agent_id)

            for fact in facts:
                # Content terms from the main content field.
                for token in _tokenize(fact.content):
                    profile.content_terms[token] += 1

                # Canonical surface terms (if available).
                if fact.canonical_surface:
                    for token in _tokenize(fact.canonical_surface):
                        profile.canonical_terms[token] += 1

                # Tags.
                for tag in fact.tags:
                    profile.tags[tag] += 1

                # Entity names as domain indicators.
                if fact.entity:
                    for token in _tokenize(fact.entity):
                        profile.domains[token] += 1

            self._profiles[agent_id] = profile

        # Compute cache version.
        self._cache_version = len(active_facts) + sum(f.version for f in active_facts)
        self._built = True

    def is_stale(self, active_facts: list[Fact]) -> bool:
        """Check if the cache needs rebuilding."""
        version = len(active_facts) + sum(f.version for f in active_facts)
        return version != self._cache_version or not self._built

    def top_agents_for_facet(
        self,
        facet_tokens: tuple[str, ...],
        *,
        top_n: int = 8,
    ) -> list[ExpertiseHit]:
        """Return top-N agents most likely to have relevant facts.

        Scoring: sum of token overlap between *facet_tokens* and the
        agent's ``content_terms``, weighted by ``trust_score`` and
        ``log(published_facts + 1)``.
        """
        if not self._built or not facet_tokens:
            return []

        token_set = set(facet_tokens)
        hits: list[ExpertiseHit] = []

        for profile in self._profiles.values():
            matched: list[str] = []
            raw_score = 0.0

            for token in token_set:
                count = profile.content_terms.get(token, 0)
                count += profile.canonical_terms.get(token, 0)
                count += profile.domains.get(token, 0)
                if count > 0:
                    matched.append(token)
                    raw_score += count

            if not matched:
                continue

            # Weight by trust and volume.
            score = raw_score * profile.trust_score * math.log(profile.published_facts + 1)
            hits.append(
                ExpertiseHit(
                    agent_id=profile.agent_id,
                    score=score,
                    matched_terms=tuple(sorted(matched)),
                )
            )

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_n]

    def top_agents_for_query(
        self,
        query: str,
        *,
        top_n: int = 8,
    ) -> list[ExpertiseHit]:
        """Convenience: tokenize *query* and delegate to ``top_agents_for_facet``."""
        tokens = tuple(_tokenize(query))
        return self.top_agents_for_facet(tokens, top_n=top_n)
