"""Support bundle builders — coarse retrieval units for the query plane.

Bundles group AtomicClaims by (entity, kind, topic) for efficient coarse-to-fine
retrieval.  They are built lazily on first query and invalidated by DirtyKeys.

Four bundle kinds:
  ENTITY_TOPIC        — all claims about a given entity (any relation)
  STATE_TIMELINE      — active state claims for a given (entity, relation) slot
  EVENT_NEIGHBORHOOD  — event claims temporally proximate to a subject
  RELATION_SUPPORT    — claims that express a specific entity-to-entity relation
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

from ai_knot.query_types import (
    AtomicClaim,
    BundleKind,
    ClaimKind,
    RawEpisode,
    SupportBundle,
    stable_bundle_id,
)

# ---------------------------------------------------------------------------
# Discourse subjects that should never anchor an entity bundle.
# ---------------------------------------------------------------------------

_DISCOURSE_SUBJECT_BLOCKLIST: frozenset[str] = frozenset(
    {
        "It",
        "This",
        "That",
        "These",
        "Those",
        "My",
        "Our",
        "Your",
        "Their",
        "His",
        "Her",
        "I",
        "We",
        "You",
        "They",
    }
)

# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_entity_topic_bundles(
    claims: list[AtomicClaim],
    *,
    agent_id: str,
    materialization_version: int,
) -> tuple[list[SupportBundle], dict[str, list[str]]]:
    """Group claims by subject; one bundle per distinct subject.

    Entity filter: skip noise subjects.  Only create a bundle if the subject
    is a proper noun (1-3 capitalised words, not in the discourse blocklist)
    OR there are at least 2 claims about it.

    Returns (bundles, memberships) where memberships = {bundle_id: [claim_id]}.
    """
    groups: dict[str, list[AtomicClaim]] = {}
    for c in claims:
        if not c.subject:
            continue
        groups.setdefault(c.subject, []).append(c)

    bundles: list[SupportBundle] = []
    memberships: dict[str, list[str]] = {}
    now = datetime.now(UTC)

    for subject, group in groups.items():
        # Entity filter: skip noise subjects.
        words = subject.split()
        is_proper_noun = (
            len(words) <= 3
            and subject[0].isupper()
            and words[0].istitle()
            and words[0] not in _DISCOURSE_SUBJECT_BLOCKLIST
        )
        is_multi_claim = len(group) >= 2
        if not (is_proper_noun or is_multi_claim):
            continue

        score = _aggregate_score(group)
        bid = stable_bundle_id(BundleKind.ENTITY_TOPIC, subject)
        b = SupportBundle(
            id=bid,
            agent_id=agent_id,
            kind=BundleKind.ENTITY_TOPIC,
            topic=subject,
            member_claim_ids=tuple(c.id for c in group),
            score_formula="mean(salience*confidence)",
            bundle_score=score,
            built_from_materialization_version=materialization_version,
            built_at=now,
        )
        bundles.append(b)
        memberships[bid] = [c.id for c in group]

    return bundles, memberships


def build_state_timeline_bundles(
    claims: list[AtomicClaim],
    *,
    agent_id: str,
    materialization_version: int,
) -> tuple[list[SupportBundle], dict[str, list[str]]]:
    """Group STATE claims by slot_key (subject::relation).

    Only includes STATE and TRANSITION kinds.
    """
    groups: dict[str, list[AtomicClaim]] = {}
    for c in claims:
        if c.kind not in (ClaimKind.STATE, ClaimKind.TRANSITION):
            continue
        key = c.slot_key or f"{c.subject}::{c.relation}"
        if not key or key == "::":
            continue
        groups.setdefault(key, []).append(c)

    bundles: list[SupportBundle] = []
    memberships: dict[str, list[str]] = {}
    now = datetime.now(UTC)

    for slot_key, group in groups.items():
        # Sort by valid_from desc so most recent state is first.
        sorted_group = sorted(group, key=lambda c: c.valid_from, reverse=True)
        score = _aggregate_score(sorted_group)
        bid = stable_bundle_id(BundleKind.STATE_TIMELINE, slot_key)
        b = SupportBundle(
            id=bid,
            agent_id=agent_id,
            kind=BundleKind.STATE_TIMELINE,
            topic=slot_key,
            member_claim_ids=tuple(c.id for c in sorted_group),
            score_formula="mean(salience*confidence)|sorted_by_valid_from_desc",
            bundle_score=score,
            built_from_materialization_version=materialization_version,
            built_at=now,
        )
        bundles.append(b)
        memberships[bid] = [c.id for c in sorted_group]

    return bundles, memberships


def build_event_neighborhood_bundles(
    claims: list[AtomicClaim],
    raw_episodes: list[RawEpisode],
    *,
    agent_id: str,
    materialization_version: int,
) -> tuple[list[SupportBundle], dict[str, list[str]]]:
    """Group EVENT claims by subject (or subject::relation when flag is set), sorted by event_time.

    Episode context is used to enrich event_time when claims lack it.

    When AI_KNOT_EVENT_BUNDLE_BY_RELATION=1 (experimental), each distinct
    (subject, relation) pair gets its own bundle instead of all events for a
    subject sharing one bundle.  Default is subject-only grouping (flag=0).
    """
    ep_times: dict[str, datetime] = {
        ep.id: ep.session_date or ep.observed_at for ep in raw_episodes
    }

    event_bundle_by_relation = os.environ.get("AI_KNOT_EVENT_BUNDLE_BY_RELATION", "0") == "1"

    groups: dict[str, list[AtomicClaim]] = {}
    for c in claims:
        if c.kind is not ClaimKind.EVENT:
            continue
        if not c.subject:
            continue
        group_key = (
            f"{c.subject}::{c.relation}" if (event_bundle_by_relation and c.relation) else c.subject
        )
        groups.setdefault(group_key, []).append(c)

    bundles: list[SupportBundle] = []
    memberships: dict[str, list[str]] = {}
    now = datetime.now(UTC)

    for group_key, group in groups.items():

        def _event_sort_key(c: AtomicClaim) -> datetime:
            return c.event_time or ep_times.get(c.source_episode_id, c.observed_at)

        sorted_group = sorted(group, key=_event_sort_key)
        score = _aggregate_score(sorted_group)
        bid = stable_bundle_id(BundleKind.EVENT_NEIGHBORHOOD, group_key)
        b = SupportBundle(
            id=bid,
            agent_id=agent_id,
            kind=BundleKind.EVENT_NEIGHBORHOOD,
            topic=group_key,
            member_claim_ids=tuple(c.id for c in sorted_group),
            score_formula="mean(salience*confidence)|sorted_by_event_time",
            bundle_score=score,
            built_from_materialization_version=materialization_version,
            built_at=now,
        )
        bundles.append(b)
        memberships[bid] = [c.id for c in sorted_group]

    return bundles, memberships


def build_relation_support_bundles(
    claims: list[AtomicClaim],
    *,
    agent_id: str,
    materialization_version: int,
) -> tuple[list[SupportBundle], dict[str, list[str]]]:
    """Group RELATION claims by "{subject}::{relation}" topic."""
    groups: dict[str, list[AtomicClaim]] = {}
    for c in claims:
        if c.kind is not ClaimKind.RELATION:
            continue
        if not (c.subject and c.relation):
            continue
        key = f"{c.subject}::{c.relation}"
        groups.setdefault(key, []).append(c)

    bundles: list[SupportBundle] = []
    memberships: dict[str, list[str]] = {}
    now = datetime.now(UTC)

    for key, group in groups.items():
        score = _aggregate_score(group)
        bid = stable_bundle_id(BundleKind.RELATION_SUPPORT, key)
        b = SupportBundle(
            id=bid,
            agent_id=agent_id,
            kind=BundleKind.RELATION_SUPPORT,
            topic=key,
            member_claim_ids=tuple(c.id for c in group),
            score_formula="mean(salience*confidence)",
            bundle_score=score,
            built_from_materialization_version=materialization_version,
            built_at=now,
        )
        bundles.append(b)
        memberships[bid] = [c.id for c in group]

    return bundles, memberships


def build_all_bundles(
    claims: list[AtomicClaim],
    raw_episodes: list[RawEpisode],
    *,
    agent_id: str,
    materialization_version: int,
) -> tuple[list[SupportBundle], dict[str, list[str]]]:
    """Run all four bundle builders and merge results."""
    all_bundles: list[SupportBundle] = []
    all_memberships: dict[str, list[str]] = {}

    for bundles, memberships in [
        build_entity_topic_bundles(
            claims, agent_id=agent_id, materialization_version=materialization_version
        ),
        build_state_timeline_bundles(
            claims, agent_id=agent_id, materialization_version=materialization_version
        ),
        build_relation_support_bundles(
            claims, agent_id=agent_id, materialization_version=materialization_version
        ),
    ]:
        all_bundles.extend(bundles)
        all_memberships.update(memberships)

    # Event neighborhood needs raw_episodes.
    bundles, memberships = build_event_neighborhood_bundles(
        claims,
        raw_episodes,
        agent_id=agent_id,
        materialization_version=materialization_version,
    )
    all_bundles.extend(bundles)
    all_memberships.update(memberships)

    return all_bundles, all_memberships


# ---------------------------------------------------------------------------
# Score helper
# ---------------------------------------------------------------------------


def _aggregate_score(claims: list[AtomicClaim]) -> float:
    """Mean of (salience * confidence) across claims; returns 0.0 for empty."""
    if not claims:
        return 0.0
    return sum(c.salience * c.confidence for c in claims) / len(claims)
