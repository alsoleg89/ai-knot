"""Resolver functions for fact deduplication and slot-addressed resolution.

Contains all the deduplication helpers (Jaccard, containment, ATC-based) and
the three resolver functions used by KnowledgeBase to handle fact lifecycle:
``resolve_against_existing``, ``resolve_by_slot``, and ``resolve_structured``.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ai_knot.tokenizer import tokenize
from ai_knot.types import Fact


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two strings using stemmed tokens.

    Uses the shared stemmed tokenizer (Broder 1997) so that morphological
    variants like "deployed"/"deploying" are recognized as identical.
    This keeps deduplication consistent with the retriever's BM25F scoring.
    """
    words_a = set(tokenize(a))
    words_b = set(tokenize(b))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _containment_similarity(a: str, b: str) -> float:
    """Max asymmetric token containment between two strings.

    containment(A, B) = |A ∩ B| / min(|A|, |B|)

    Catches cases where a short fact is a subset of a longer one, or
    two facts express the same idea with different amounts of detail.
    Jaccard penalises length asymmetry; containment does not.
    """
    words_a = set(tokenize(a))
    words_b = set(tokenize(b))
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    return intersection / min(len(words_a), len(words_b))


def _dedup_similarity(a: str, b: str) -> float:
    """Combined dedup similarity: max(jaccard, containment).

    Using the max of both metrics catches both symmetric overlap
    (Jaccard) and asymmetric subset relationships (containment).
    """
    return max(_jaccard_similarity(a, b), _containment_similarity(a, b))


# Unified dedup threshold for both within-batch and cross-store deduplication.
# 0.85 avoids false-positive merges of short but semantically distinct rules
# (e.g. "keep posts under 300 words" vs "use fenced code blocks"), which occur
# at the original 0.7 threshold when containment similarity exceeds min-token ratio.
_DEDUP_THRESHOLD: float = 0.85


def deduplicate_facts(facts: list[Fact], *, threshold: float = _DEDUP_THRESHOLD) -> list[Fact]:
    """Remove near-duplicate facts by combined similarity (Jaccard + containment).

    Args:
        facts: List of facts to deduplicate.
        threshold: Similarity threshold above which facts are considered duplicates.

    Returns:
        Deduplicated list, keeping the first occurrence.
    """
    if not facts:
        return []

    unique: list[Fact] = []
    for fact in facts:
        is_dup = False
        for existing in unique:
            if _dedup_similarity(fact.content, existing.content) >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(fact)
    return unique


def resolve_against_existing(
    new_facts: list[Fact],
    existing: list[Fact],
    *,
    threshold: float = _DEDUP_THRESHOLD,
) -> tuple[list[Fact], list[Fact]]:
    """Separate new facts into inserts and temporal closes relative to existing facts.

    For each new fact, if a sufficiently similar existing fact is found
    (combined similarity >= threshold), the old fact is **temporally closed**
    (``valid_until`` set to now) and the new fact is queued for insertion with
    bumped importance. This preserves the full fact history instead of mutating
    it in place.

    If no match is found, the new fact is inserted unchanged.

    Args:
        new_facts: Facts extracted from the latest conversation.
        existing: Active facts already stored for this agent.
        threshold: Combined similarity threshold to consider two facts duplicates.

    Returns:
        A 2-tuple ``(to_insert, closed_existing)`` where:
        - ``to_insert``: facts to insert (both genuinely new and new-version replacements).
        - ``closed_existing``: old facts that were temporally closed (``valid_until`` set).
    """
    to_insert: list[Fact] = []
    closed: list[Fact] = []
    now = datetime.now(UTC)

    for new in new_facts:
        matched: Fact | None = None
        best_sim = 0.0
        for old in existing:
            sim = _dedup_similarity(new.content, old.content)
            if sim >= threshold and sim > best_sim:
                best_sim = sim
                matched = old
        if matched is not None:
            # Temporal close: seal the old version without mutating its content.
            matched.valid_until = now
            closed.append(matched)
            # Carry forward importance and version from the old fact.
            new.importance = min(1.0, matched.importance + 0.05)
            new.version = matched.version + 1
            # Carry over evidence trail from the old fact.
            if matched.source_snippets:
                existing_snips = set(new.source_snippets)
                carried = [s for s in matched.source_snippets if s not in existing_snips]
                new.source_snippets = (new.source_snippets + carried)[:5]
            to_insert.append(new)
        else:
            to_insert.append(new)

    return to_insert, closed


_PRONOUNS = frozenset({"he", "she", "it", "they", "i", "we", "you", "him", "her", "them"})


def entity_match(a: str, b: str) -> bool:
    """Return True if two entity strings refer to the same real-world entity.

    Uses containment (substring in both directions) and Jaccard similarity.
    Guards against pronoun-based false matches (e.g. "he" != "Alex Chen").
    """
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return False
    if a in _PRONOUNS or b in _PRONOUNS:
        return False
    if a in b or b in a:
        return True
    return _jaccard_similarity(a, b) > 0.5


def resolve_structured(
    new_fact: Fact,
    existing: list[Fact],
) -> Fact | None:
    """Entity-addressed dedup: find existing fact with same entity+attribute.

    Returns the existing Fact if a match is found, or None if no match
    (meaning this fact should be treated as a new insert).

    Only fires when both new_fact.entity and new_fact.attribute are non-empty.
    Falls back to cosine-based dedup otherwise.
    """
    if not new_fact.entity or not new_fact.attribute:
        return None
    for existing_fact in existing:
        if not existing_fact.entity or not existing_fact.attribute:
            continue
        if (
            entity_match(new_fact.entity, existing_fact.entity)
            and new_fact.attribute.lower().strip() == existing_fact.attribute.lower().strip()
            and existing_fact.is_active()
        ):
            return existing_fact
    return None


def resolve_by_slot(
    new_fact: Fact,
    existing: list[Fact],
) -> tuple[str, Fact | None]:
    """Exact-match slot resolution against existing active facts.

    Returns ``(op, matched)`` where *op* is one of:

    - ``'branch'``: ``new_fact`` has no ``slot_key``, or no existing fact shares it → insert as new.
    - ``'reinforce'``: same slot, same ``value_text`` → bump confidence on existing, skip insert.
    - ``'supersede'``: same slot, different (or missing) ``value_text`` → close old, insert new.

    This is faster and more reliable than Jaccard-based ``resolve_structured`` because
    ``slot_key`` is a deterministic ``"{entity}::{attribute}"`` address — no fuzzy matching needed.

    Args:
        new_fact: Newly extracted fact to resolve.
        existing: Active facts to search. Caller is responsible for passing only active facts.
    """
    if not new_fact.slot_key:
        return "branch", None

    for old in existing:
        if old.slot_key != new_fact.slot_key:
            continue
        # Same slot — compare values to decide reinforce vs supersede.
        if (
            new_fact.value_text
            and old.value_text
            and new_fact.value_text.strip().lower() == old.value_text.strip().lower()
        ):
            return "reinforce", old
        return "supersede", old

    return "branch", None
