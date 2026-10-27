"""Bounded, revision-addressed roots for explicitly derived facts.

The host records dependencies; this does not prove that a model's text follows
from them or discover missing dependencies in arbitrary imported prose.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from ai_knot.types import Fact

from .changes import access_projection, evidence_revision, visible
from .models import EvidenceRef, canonical

if TYPE_CHECKING:
    from ai_knot.storage.base import StorageBackend

ROOTS_KEY = "_ai_knot_evidence_roots"
DEPENDENCIES_KEY = "_ai_knot_evidence_dependencies"
PUBLICATION_KEY = "_ai_knot_publication_identity"
MAX_ROOTS = 128


class LineageCapacityExceeded(ValueError):
    """Exact lineage exceeds the bounded evidence representation."""


def observation_identity(ref: EvidenceRef) -> str:
    """Opaque identity of one source revision, without a private root pointer."""
    return hashlib.sha256(
        canonical([ref.namespace, ref.fact_id, ref.revision]).encode()
    ).hexdigest()


def canonical_observations(
    roots: set[EvidenceRef],
    load: Callable[[str, str], list[Fact]],
) -> tuple[EvidenceRef, ...]:
    """Collapse a private original only against an already accessible public copy.

    This changes evidence counting, not dependency/access checks. Ordinary public
    publications remain self-contained roots; private derivations keep theirs.
    """
    published = set()
    for root in roots:
        if root.namespace != "__shared__":
            continue
        fact = next(iter(load(root.namespace, root.fact_id)), None)
        if fact is not None and ROOTS_KEY not in fact.qualifiers:
            identity = fact.qualifiers.get(PUBLICATION_KEY)
            if isinstance(identity, str):
                published.add(identity)
    return tuple(
        sorted(
            root
            for root in roots
            if root.namespace == "__shared__" or observation_identity(root) not in published
        )
    )


def is_derived(fact: Fact) -> bool:
    """Recognize persisted automatic projections without guessing their revisions."""
    return (
        ROOTS_KEY in fact.qualifiers
        or fact.verification_source == "aggregate"
        or fact.attribute.startswith("_aggregate")
        or "consolidation_ids" in fact.qualifiers
    )


def evidence_roots(namespace: str, fact: Fact) -> tuple[EvidenceRef, ...]:
    """Original facts are one root; derived facts carry a flattened unique set."""
    raw = fact.qualifiers.get(ROOTS_KEY)
    if raw is None:
        if is_derived(fact):
            raise ValueError("generated fact has no exact evidence roots")
        return (EvidenceRef(namespace, fact.id, evidence_revision(fact)),)
    return _decode_refs(raw)


def evidence_dependencies(fact: Fact) -> tuple[EvidenceRef, ...]:
    """Direct and transitive input revisions, distinct from independent roots."""
    raw = fact.qualifiers.get(DEPENDENCIES_KEY)
    return () if raw is None else _decode_refs(raw)


def _decode_refs(raw: str) -> tuple[EvidenceRef, ...]:
    data = json.loads(raw)
    if not isinstance(data, list) or not 1 <= len(data) <= MAX_ROOTS:
        raise ValueError("invalid evidence roots")
    roots = []
    for item in data:
        if not isinstance(item, dict) or set(item) != {"namespace", "fact_id", "revision"}:
            raise ValueError("invalid evidence root")
        roots.append(EvidenceRef(**item))
    return tuple(sorted(set(roots)))


def encode_roots(roots: tuple[EvidenceRef, ...]) -> str:
    roots = tuple(sorted(set(roots)))
    if len(roots) > MAX_ROOTS:
        raise LineageCapacityExceeded("evidence roots must contain 1..128 entries")
    if not roots:
        raise ValueError("evidence roots must contain 1..128 entries")
    return canonical(
        [
            {"namespace": ref.namespace, "fact_id": ref.fact_id, "revision": ref.revision}
            for ref in roots
        ]
    )


def derivation_qualifiers(namespace: str, sources: list[Fact]) -> dict[str, str]:
    """Record flattened roots and every input revision, including intermediate views."""
    roots = {ref for source in sources for ref in evidence_roots(namespace, source)}
    dependencies = {
        ref
        for source in sources
        for ref in (
            EvidenceRef(namespace, source.id, evidence_revision(source)),
            *evidence_dependencies(source),
        )
    }
    return {
        ROOTS_KEY: encode_roots(tuple(roots)),
        DEPENDENCIES_KEY: encode_roots(tuple(dependencies)),
    }


def lineage_status(
    namespace: str,
    fact: Fact,
    reader: str,
    grants: set[str],
    load: Callable[[str, str], list[Fact]],
) -> tuple[bool, bool]:
    """Return (readable, current) without exposing inaccessible root identities.

    A historical source may be closed by consolidation. Its exact archived
    revision remains usable; changing/removing that revision invalidates descendants.
    """
    if ROOTS_KEY not in fact.qualifiers:
        return True, not is_derived(fact)
    try:
        roots = evidence_roots(namespace, fact)
        dependencies = evidence_dependencies(fact)
    except (TypeError, ValueError, KeyError):
        return True, False
    for root in sorted({*roots, *dependencies}):
        if root.namespace not in (reader, "__shared__"):
            return False, False
        try:
            matches = load(root.namespace, root.fact_id)
        except ValueError:
            # A newly changed dependency outside the caller's captured lock set
            # cannot be accepted by a pre-action guard.
            return True, False
        original = next(iter(matches), None)
        if original is None:
            return True, False
        if root.namespace != reader and not visible(access_projection(original), reader, grants):
            return False, False
        if is_derived(original) and (root in roots or ROOTS_KEY not in original.qualifiers):
            return True, False
        if evidence_revision(original) != root.revision:
            return True, False
    return True, True


def filter_current_lineage(
    storage: StorageBackend,
    namespace: str,
    reader: str,
    grants: set[str],
    facts: list[Fact],
) -> list[Fact]:
    """Apply the same dependency gate to private and shared recall paths."""
    cache: dict[str, dict[str, Fact]] = {}

    def load(scope: str, fact_id: str) -> list[Fact]:
        if scope not in cache:
            cache[scope] = {f.id: f for f in storage.load(scope)}
        found = cache[scope].get(fact_id)
        return [found] if found else []

    return [f for f in facts if all(lineage_status(namespace, f, reader, grants, load))]
