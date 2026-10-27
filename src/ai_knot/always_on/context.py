"""Read-only task working sets; applicability and role never grant authority."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ai_knot._inverted_index import InvertedIndex
from ai_knot.multi_agent.assembly import CoverageAwareAssembler
from ai_knot.multi_agent.facets import _content_tokens
from ai_knot.multi_agent.models import CandidateFact
from ai_knot.types import Fact, MemoryType

from .changes import access_projection, canonical, evidence_revision, visible
from .lineage import lineage_status
from .models import ContextFact, EvidenceRef, TaskContext, context_label, positive_integer

if TYPE_CHECKING:
    from .memory import AlwaysOnMemory


def select_context(
    memory: AlwaysOnMemory,
    requirements: tuple[str, ...],
    *,
    task_key: str,
    role: str,
    max_facts: int,
    max_chars: int,
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
    shared = memory._pool is not None and memory._pool._persist_stats
    if shared:
        memory._check_namespace("__shared__")
    scopes = (memory._owner, "__shared__") if shared else (memory._owner,)
    with memory._transaction(scopes, write=False) as session:
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
                    or fact.type == MemoryType.EPISODIC
                    or fact.qualifiers.get("task_key", "") not in ("", task_key)
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

        assembled = CoverageAwareAssembler().assemble(
            candidates_by_facet=facets,
            top_k=max_facts,
            # Explicit task prerequisites take priority over publisher quotas.
            n_publishers=1,
            max_chars=max_chars,
            admit=admit,
            min_backfill_score_frac=0.25,
        )
        selected = tuple(
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
