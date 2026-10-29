"""Read-only task working sets; applicability and role never grant authority."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import replace
from datetime import datetime
from typing import TYPE_CHECKING

from ai_knot._inverted_index import InvertedIndex
from ai_knot.multi_agent.assembly import CoverageAwareAssembler
from ai_knot.multi_agent.facets import _content_tokens
from ai_knot.multi_agent.models import AssemblyResult, CandidateFact
from ai_knot.types import Fact, MemoryType

from .changes import access_projection, canonical, evidence_revision, visible
from .lineage import lineage_status
from .models import (
    ContextFact,
    EvidenceRef,
    SnapshotTooLarge,
    StaleEvidence,
    TaskContext,
    TaskContextGuard,
    context_label,
    positive_integer,
)
from .reads import ReadCache

if TYPE_CHECKING:
    from .memory import AlwaysOnMemory
    from .storage import AlwaysOnSession


def task_scopes(memory: AlwaysOnMemory) -> tuple[str, ...]:
    if memory._pool is not None and memory._pool._persist_stats:
        memory._check_namespace("__shared__")
        return tuple(dict.fromkeys((memory._owner, "__shared__")))
    return (memory._owner,)


def task_applicable(fact: Fact, task_key: str) -> bool:
    return fact.type != MemoryType.EPISODIC and fact.qualifiers.get("task_key", "") in (
        "",
        task_key,
    )


def task_sources(
    memory: AlwaysOnMemory,
    session: AlwaysOnSession,
    task_key: str,
    now: datetime,
) -> Iterator[tuple[EvidenceRef, Fact]]:
    session = ReadCache(session)
    for namespace in task_scopes(memory):
        for fact in session.load_task_facts(namespace, task_key=task_key):
            if not task_applicable(fact, task_key) or not fact.supported or not fact.is_active(now):
                continue
            ref = EvidenceRef(namespace, fact.id, evidence_revision(fact))
            if not memory._validate_basis(session, (ref,), now):
                yield ref, fact


def task_snapshot(
    memory: AlwaysOnMemory,
    *,
    task_key: str,
    max_facts: int,
    max_chars: int,
) -> tuple[ContextFact, ...]:
    context_label(task_key, "task_key")
    positive_integer(max_facts, "max_facts", 128)
    positive_integer(max_chars, "max_chars", 65536)
    with memory._transaction(task_scopes(memory), write=False) as session:
        result: list[ContextFact] = []
        remaining = max_chars
        for ref, fact in task_sources(memory, session, task_key, memory._now()):
            if len(result) == max_facts or len(fact.content) > remaining:
                raise SnapshotTooLarge("Task snapshot exceeds the output budget")
            result.append(ContextFact(ref, fact.content, fact.slot_key))
            remaining -= len(fact.content)
        return tuple(sorted(result, key=lambda item: (item.ref.namespace, item.ref.fact_id)))


def task_guard(
    memory: AlwaysOnMemory,
    session: AlwaysOnSession,
    *,
    task_key: str,
    max_facts: int,
    now: datetime,
) -> TaskContextGuard:
    refs: list[tuple[str, str, str]] = []
    for ref, _ in task_sources(memory, session, task_key, now):
        if len(refs) == max_facts:
            raise SnapshotTooLarge("Task guard exceeds the candidate limit")
        refs.append((ref.namespace, ref.fact_id, ref.revision))
    identity = {
        "policy": "task-context-v1",
        "owner": memory._owner,
        "scopes": task_scopes(memory),
        "task_key": task_key,
        "max_facts": max_facts,
        "refs": sorted(refs),
    }
    return TaskContextGuard(
        task_key, hashlib.sha256(canonical(identity).encode()).hexdigest(), max_facts
    )


def select_context(
    memory: AlwaysOnMemory,
    requirements: tuple[str, ...],
    *,
    task_key: str,
    role: str,
    max_facts: int,
    max_chars: int,
    required_refs: tuple[EvidenceRef, ...] = (),
) -> TaskContext:
    if (
        not isinstance(requirements, (tuple, list))
        or not 1 <= len(requirements) <= 5
        or any(
            not isinstance(text, str) or not 1 <= len(text) <= 4096 or not _content_tokens(text)
            for text in requirements
        )
        or len(set(requirements)) != len(requirements)
    ):
        raise ValueError("one to five unique content-bearing requirements required")
    context_label(task_key, "task_key")
    context_label(role, "role")
    positive_integer(max_facts, "max_facts", 128)
    positive_integer(max_chars, "max_chars", 65536)
    required_refs = validate_required_refs(required_refs)
    scopes = task_scopes(memory)
    shared = "__shared__" in scopes
    with memory._transaction(scopes, write=False) as session:
        session = ReadCache(session)
        now = memory._now()
        all_facts = {
            namespace: {fact.id: fact for fact in session.load_facts(namespace)}
            for namespace in scopes
        }
        grants = memory._evidence_grants(session) if shared else set()

        def load(namespace: str, fact_id: str) -> list[Fact]:
            fact = all_facts.get(namespace, {}).get(fact_id)
            return [fact] if fact is not None else []

        originals: dict[str, tuple[EvidenceRef, Fact]] = {}
        indexed: list[Fact] = []
        for namespace, facts in all_facts.items():
            for fact in facts.values():
                if (
                    not fact.supported
                    or not fact.is_active(now)
                    or not task_applicable(fact, task_key)
                ):
                    continue
                if namespace != memory._owner and not visible(
                    access_projection(fact), memory._owner, grants
                ):
                    continue
                if not all(lineage_status(namespace, fact, memory._owner, grants, load)):
                    continue
                # The existing string-keyed index must distinguish equal IDs in
                # private/shared scopes. Original refs and revisions stay intact.
                key = canonical([namespace, fact.id])
                originals[key] = EvidenceRef(namespace, fact.id, evidence_revision(fact)), fact
                indexed.append(replace(fact, id=key))

        for ref in required_refs:
            key = canonical([ref.namespace, ref.fact_id])
            if key not in originals or memory._validate_basis(session, (ref,), now):
                raise StaleEvidence("Required task context unavailable")
        required = required_context(required_refs, originals, max_facts, max_chars)
        required_keys = {canonical([ref.namespace, ref.fact_id]) for ref in required_refs}
        index = InvertedIndex(indexed)
        by_id = {fact.id: fact for fact in indexed}
        content_terms = {fact.id: set(_content_tokens(fact.content)) for fact in indexed}
        candidates: dict[str, CandidateFact] = {}
        facets: dict[str, list[CandidateFact]] = {}
        for number, requirement in enumerate(requirements):
            facet_id = f"f{number}"
            query_tokens = set(_content_tokens(requirement))
            matches = {
                key: len(query_tokens & terms)
                for key, terms in content_terms.items()
                if query_tokens & terms
            }
            scores = {
                key: score
                for key, score in index.score(requirement).items()
                if key in matches and score > 0
            }
            if not scores:
                # BM25 suppresses ubiquitous terms in large corpora. A direct
                # original-token match is evidence; a role-only match is not.
                scores = {key: hits / len(query_tokens) for key, hits in matches.items()}
            facets[facet_id] = []
            for key, relevance in scores.items():
                if relevance <= 0:
                    continue
                fact = by_id[key]
                multiplier = 1.25 if role and fact.qualifiers.get("for_role") == role else 1.0
                candidate = candidates.setdefault(key, CandidateFact(fact))
                candidate.base_score = max(candidate.base_score, relevance * multiplier)
                candidate.facet_scores[facet_id] = relevance
                facets[facet_id].append(candidate)

        admissible: dict[str, bool] = {}

        def admit(candidate: CandidateFact) -> bool:
            key = candidate.fact.id
            if key not in admissible:
                ref, _ = originals[key]
                admissible[key] = not memory._validate_basis(session, (ref,), now)
            return admissible[key]

        remaining_chars = max_chars - sum(len(f.content) for f in required)
        assembled = (
            CoverageAwareAssembler().assemble(
                candidates_by_facet={
                    fid: [c for c in group if c.fact.id not in required_keys]
                    for fid, group in facets.items()
                },
                top_k=max_facts - len(required),
                # Explicit task prerequisites take priority over publisher quotas.
                n_publishers=1,
                max_chars=remaining_chars,
                admit=admit,
                min_backfill_score_frac=0.25,
            )
            if remaining_chars and len(required) < max_facts
            else AssemblyResult(
                budget_limited=not remaining_chars
                and len(required) < max_facts
                and any(key not in required_keys and admit(c) for key, c in candidates.items())
            )
        )
        assembled.covered_facets.update(
            facet
            for key in required_keys
            if key in candidates
            for facet, score in candidates[key].facet_scores.items()
            if score > 0
        )
        selected = required + tuple(
            ContextFact(
                originals[item.fact.id][0],
                originals[item.fact.id][1].content,
                originals[item.fact.id][1].slot_key,
            )
            for item in assembled.selected
        )
        covered = tuple(
            text for i, text in enumerate(requirements) if f"f{i}" in assembled.covered_facets
        )
        missing = tuple(
            text for i, text in enumerate(requirements) if f"f{i}" not in assembled.covered_facets
        )
        return TaskContext(selected, covered, missing, assembled.budget_limited, now, not missing)


def validate_required_refs(refs: tuple[EvidenceRef, ...]) -> tuple[EvidenceRef, ...]:
    from ._state import union_basis

    if (
        not isinstance(refs, (tuple, list))
        or len(refs) > 128
        or any(not isinstance(ref, EvidenceRef) for ref in refs)
    ):
        raise ValueError("required_refs requires at most 128 evidence references")
    union_basis(tuple(refs))
    return tuple(dict.fromkeys(refs))


def required_context(
    refs: tuple[EvidenceRef, ...],
    sources: dict[str, tuple[EvidenceRef, Fact]],
    max_facts: int,
    max_chars: int,
) -> tuple[ContextFact, ...]:
    result = []
    for ref in refs:
        source = sources.get(canonical([ref.namespace, ref.fact_id]))
        if source is None or source[0] != ref:
            raise StaleEvidence("Required task context unavailable")
        result.append(ContextFact(ref, source[1].content, source[1].slot_key))
    if len(result) > max_facts or sum(len(f.content) for f in result) > max_chars:
        raise SnapshotTooLarge("Required task context exceeds output budget")
    return tuple(result)
