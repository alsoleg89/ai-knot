"""Explicit, budgeted semantic candidates; similarity never certifies a task."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Literal

from ai_knot.multi_agent.assembly import CoverageAwareAssembler
from ai_knot.multi_agent.models import CandidateFact
from ai_knot.types import Fact

from .changes import canonical, evidence_revision
from .context import task_applicable, task_scopes, task_sources
from .memory import AlwaysOnMemory
from .models import (
    BasisIssue,
    ContextFact,
    ContextReview,
    EvidenceRef,
    context_label,
    positive_integer,
)

EmbeddingPort = Callable[[list[str]], Awaitable[list[list[float]]]]
Source = tuple[EvidenceRef, Fact]
Status = Literal["snapshot", "ok", "input_limit", "budget_exhausted", "unavailable", "changed"]


@dataclass(frozen=True)
class SemanticBudget:
    max_candidates: int = 128
    max_input_bytes: int = 65536
    max_provider_calls: int = 0
    timeout: float = 5.0

    def __post_init__(self) -> None:
        positive_integer(self.max_candidates, "max_candidates", 128)
        positive_integer(self.max_input_bytes, "max_input_bytes", 1048576)
        if type(self.max_provider_calls) is not int or self.max_provider_calls < 0:
            raise ValueError("max_provider_calls must be a nonnegative integer")
        if (
            type(self.timeout) not in (int, float)
            or not math.isfinite(self.timeout)
            or self.timeout <= 0
        ):
            raise ValueError("timeout must be positive and finite")


@dataclass(frozen=True)
class SemanticCandidates:
    status: Status
    review: ContextReview
    provider_calls_reserved: int = 0
    input_bytes_reserved: int = 0


def _fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _vectors(value: object, count: int, dimension: int | None) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != count:
        raise ValueError("Invalid embedding count")
    result = []
    for vector in value:
        if (
            not isinstance(vector, list)
            or not vector
            or any(type(n) not in (float, int) or not math.isfinite(n) for n in vector)
        ):
            raise ValueError("Invalid embedding vector")
        dimension = dimension or len(vector)
        if len(vector) != dimension:
            raise ValueError("Embedding dimensions changed")
        norm = math.hypot(*vector)
        if not math.isfinite(norm) or norm == 0:
            raise ValueError("Invalid embedding norm")
        result.append([float(n) / norm for n in vector])
    return result


class SemanticContext:
    """One owner and fixed embedding space; reuse this helper within one event loop.

    The callback must make at most one request per invocation without retries.
    Its authoritative meter owns aggregate spending; SemanticBudget only bounds
    this acquisition. Nothing runs until acquire is explicitly awaited.
    """

    def __init__(self, memory: AlwaysOnMemory, *, embed: EmbeddingPort) -> None:
        if not callable(embed):
            raise ValueError("An explicit async embedding callback is required")
        self._memory = memory
        self._embed = embed
        self._lock = asyncio.Lock()
        self._documents: dict[str, tuple[str, list[float]]] = {}
        self._queries: tuple[tuple[str, ...], list[list[float]]] | None = None
        self._dimension: int | None = None

    def _snapshot(self, task_key: str, limit: int) -> tuple[datetime, dict[str, Source]]:
        memory = self._memory
        with memory._transaction(task_scopes(memory), write=False) as session:
            now = memory._now()
            sources = {}
            for ref, fact in task_sources(memory, session, task_key, now):
                sources[canonical([ref.namespace, fact.id])] = (ref, fact)
                if len(sources) > limit:
                    return now, sources  # Reject the universe, never embed this prefix.
            return now, sources

    async def acquire(
        self,
        requirements: tuple[str, ...],
        *,
        budget: SemanticBudget,
        task_key: str = "",
        role: str = "",
        max_facts: int = 8,
        max_chars: int = 12000,
    ) -> SemanticCandidates:
        if (
            not isinstance(requirements, (tuple, list))
            or not 1 <= len(requirements) <= 5
            or any(not isinstance(q, str) or not q.strip() or len(q) > 4096 for q in requirements)
            or len(set(requirements)) != len(requirements)
        ):
            raise ValueError("One to five unique nonempty requirements required")
        requirements = tuple(requirements)
        if not isinstance(budget, SemanticBudget):
            raise ValueError("SemanticBudget required")
        context_label(task_key, "task_key")
        context_label(role, "role")
        positive_integer(max_facts, "max_facts", 128)
        positive_integer(max_chars, "max_chars", 65536)
        async with self._lock:
            return await self._acquire(requirements, budget, task_key, role, max_facts, max_chars)

    async def _acquire(
        self,
        requirements: tuple[str, ...],
        budget: SemanticBudget,
        task_key: str,
        role: str,
        max_facts: int,
        max_chars: int,
    ) -> SemanticCandidates:
        memory = self._memory
        checked, sources = self._snapshot(task_key, budget.max_candidates)
        empty = ContextReview((), (), (), checked, False)
        if len(sources) > budget.max_candidates:
            self._documents.clear()
            return SemanticCandidates("input_limit", empty)
        fingerprints = {key: _fingerprint(fact.content) for key, (_, fact) in sources.items()}
        self._documents = {
            key: value
            for key, value in self._documents.items()
            if fingerprints.get(key) == value[0]
        }
        if (
            len(sources) <= max_facts
            and sum(len(f.content) for _, f in sources.values()) <= max_chars
        ):
            accepted = tuple(
                ContextFact(ref, fact.content, fact.slot_key)
                for _, (ref, fact) in sorted(sources.items())
            )
            return SemanticCandidates("snapshot", ContextReview(accepted, (), (), checked, True))

        missing = {
            key: fact.content for key, (_, fact) in sources.items() if key not in self._documents
        }
        query_vectors = (
            self._queries[1] if self._queries and self._queries[0] == requirements else []
        )
        inputs = ([] if query_vectors else list(requirements)) + list(missing.values())
        calls = size = 0
        fresh: dict[str, list[float]] = {}
        if inputs:
            size = len(json.dumps(inputs, ensure_ascii=False, separators=(",", ":")).encode())
            if size > budget.max_input_bytes:
                return SemanticCandidates("input_limit", empty)
            if budget.max_provider_calls < 1:
                return SemanticCandidates("budget_exhausted", empty)
            calls = 1  # Reserve before awaiting; failures never refund an unknown call.
            try:
                vectors = _vectors(
                    await asyncio.wait_for(self._embed(inputs), budget.timeout),
                    len(inputs),
                    self._dimension,
                )
            except Exception:
                return SemanticCandidates("unavailable", empty, calls, size)
            query_count = 0 if query_vectors else len(requirements)
            if query_count:
                query_vectors = vectors[:query_count]
            fresh = dict(zip(missing, vectors[query_count:], strict=True))
            self._dimension = len(vectors[0])
            self._queries = requirements, query_vectors

        with memory._transaction(task_scopes(memory), write=False) as session:
            now = memory._now()
            valid: dict[str, Source] = {}
            from .reads import ReadCache

            session = ReadCache(session)
            issues: list[BasisIssue] = []
            for key, (ref, _) in sources.items():
                problems = memory._validate_basis(session, (ref,), now)
                if problems:
                    issues.extend(problems)
                    continue
                fact = session.load_facts(ref.namespace, fact_id=ref.fact_id)[0]
                if not task_applicable(fact, task_key) or evidence_revision(fact) != ref.revision:
                    issues.append(BasisIssue("context_changed", ref))
                    continue
                valid[key] = ref, fact
                if key in fresh:
                    self._documents[key] = fingerprints[key], fresh[key]
            self._documents = {key: value for key, value in self._documents.items() if key in valid}
            facets: dict[str, list[CandidateFact]] = {}
            candidates: dict[str, CandidateFact] = {}
            for i, query in enumerate(query_vectors):
                facet = f"q{i}"
                facets[facet] = []
                for key, (_, fact) in valid.items():
                    vector = self._documents[key][1]
                    # Unit-normalized vectors make their dot product cosine similarity.
                    score = sum(q * v for q, v in zip(query, vector, strict=True))
                    if score <= 0:
                        continue
                    boost = 1.25 if role and fact.qualifiers.get("for_role") == role else 1.0
                    candidate = candidates.setdefault(key, CandidateFact(replace(fact, id=key)))
                    candidate.base_score = max(candidate.base_score, score * boost)
                    candidate.facet_scores[facet] = score
                    facets[facet].append(candidate)
            selected = (
                CoverageAwareAssembler()
                .assemble(
                    candidates_by_facet=facets,
                    top_k=max_facts,
                    max_chars=max_chars,
                    n_publishers=1,
                    min_backfill_score_frac=0.25,
                )
                .selected
            )
            selected_keys = {candidate.fact.id for candidate in selected}
            accepted = tuple(
                ContextFact(
                    valid[c.fact.id][0], valid[c.fact.id][1].content, valid[c.fact.id][1].slot_key
                )
                for c in selected
            )
            omitted = tuple(valid[key][0] for key in sorted(candidates) if key not in selected_keys)
            review = ContextReview(
                accepted, tuple(issues), omitted, now, not issues and not omitted
            )
            return SemanticCandidates("changed" if issues else "ok", review, calls, size)
