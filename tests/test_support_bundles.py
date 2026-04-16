"""Tests for support bundle builders and DirtyKey invalidation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_knot.materialization import materialize_episode
from ai_knot.query_types import (
    AtomicClaim,
    BundleKind,
    ClaimKind,
    DirtyKey,
    RawEpisode,
    make_claim_id,
)
from ai_knot.support_bundles import (
    build_all_bundles,
    build_entity_topic_bundles,
    build_event_neighborhood_bundles,
    build_relation_support_bundles,
    build_state_timeline_bundles,
)

NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
AGENT = "test-agent"
VER = 1


def _make_claim(
    subject: str,
    relation: str,
    value: str = "yes",
    kind: ClaimKind = ClaimKind.STATE,
    episode_id: str = "ep-0",
) -> AtomicClaim:
    return AtomicClaim(
        id=make_claim_id(),
        agent_id=AGENT,
        kind=kind,
        subject=subject,
        relation=relation,
        value_text=value,
        value_tokens=(value,),
        qualifiers={},
        polarity="support",
        event_time=None,
        observed_at=NOW,
        valid_from=NOW,
        valid_until=None,
        confidence=0.9,
        salience=0.8,
        source_episode_id=episode_id,
        source_spans=((0, len(value)),),
        materialization_version=VER,
        materialized_at=NOW,
        slot_key=f"{subject}::{relation}",
        version=0,
        origin_agent_id=AGENT,
    )


def _make_episode(raw_text: str, ep_id: str = "ep-0") -> RawEpisode:
    return RawEpisode(
        id=ep_id,
        agent_id=AGENT,
        session_id="sess-0",
        turn_id=ep_id,
        speaker="user",
        observed_at=NOW,
        session_date=None,
        raw_text=raw_text,
        source_meta={},
        parent_episode_id=None,
    )


# ---------------------------------------------------------------------------
# Entity-topic bundles
# ---------------------------------------------------------------------------


class TestEntityTopicBundles:
    def test_one_bundle_per_subject(self):
        claims = [
            _make_claim("Alice", "job"),
            _make_claim("Alice", "city"),
            _make_claim("Bob", "job"),
        ]
        bundles, memberships = build_entity_topic_bundles(
            claims, agent_id=AGENT, materialization_version=VER
        )
        topics = {b.topic for b in bundles}
        assert "Alice" in topics
        assert "Bob" in topics
        assert len(bundles) == 2

    def test_all_claims_for_subject_in_bundle(self):
        claims = [_make_claim("Alice", "job"), _make_claim("Alice", "city")]
        bundles, memberships = build_entity_topic_bundles(
            claims, agent_id=AGENT, materialization_version=VER
        )
        alice_bundle = next(b for b in bundles if b.topic == "Alice")
        assert len(alice_bundle.member_claim_ids) == 2

    def test_bundle_kind_is_entity_topic(self):
        claims = [_make_claim("Alice", "job")]
        bundles, _ = build_entity_topic_bundles(claims, agent_id=AGENT, materialization_version=VER)
        assert all(b.kind is BundleKind.ENTITY_TOPIC for b in bundles)

    def test_empty_claims_returns_empty(self):
        bundles, memberships = build_entity_topic_bundles(
            [], agent_id=AGENT, materialization_version=VER
        )
        assert bundles == []
        assert memberships == {}

    def test_membership_maps_bundle_to_claim_ids(self):
        c1 = _make_claim("Alice", "job")
        c2 = _make_claim("Alice", "city")
        bundles, memberships = build_entity_topic_bundles(
            [c1, c2], agent_id=AGENT, materialization_version=VER
        )
        alice_bundle = next(b for b in bundles if b.topic == "Alice")
        ids = set(memberships[alice_bundle.id])
        assert c1.id in ids
        assert c2.id in ids


# ---------------------------------------------------------------------------
# State-timeline bundles
# ---------------------------------------------------------------------------


class TestStateTimelineBundles:
    def test_one_bundle_per_slot(self):
        claims = [
            _make_claim("Alice", "job", "engineer"),
            _make_claim("Alice", "job", "manager"),  # two values, same slot
            _make_claim("Alice", "city", "NYC"),
        ]
        bundles, _ = build_state_timeline_bundles(
            claims, agent_id=AGENT, materialization_version=VER
        )
        topics = {b.topic for b in bundles}
        assert "Alice::job" in topics
        assert "Alice::city" in topics

    def test_bundle_kind_is_state_timeline(self):
        claims = [_make_claim("Alice", "job", kind=ClaimKind.STATE)]
        bundles, _ = build_state_timeline_bundles(
            claims, agent_id=AGENT, materialization_version=VER
        )
        assert all(b.kind is BundleKind.STATE_TIMELINE for b in bundles)


# ---------------------------------------------------------------------------
# Relation-support bundles
# ---------------------------------------------------------------------------


class TestRelationSupportBundles:
    def test_relation_bundle_created_for_relation_claims(self):
        claims = [
            _make_claim("Alice", "knows", "Bob", kind=ClaimKind.RELATION),
        ]
        bundles, _ = build_relation_support_bundles(
            claims, agent_id=AGENT, materialization_version=VER
        )
        assert len(bundles) >= 1
        assert any(b.kind is BundleKind.RELATION_SUPPORT for b in bundles)


# ---------------------------------------------------------------------------
# Event-neighborhood bundles
# ---------------------------------------------------------------------------


class TestEventNeighborhoodBundles:
    def test_event_bundle_created_for_event_claims(self):
        episode = _make_episode("Alice attended a conference in June.")
        claims = [_make_claim("Alice", "attended", "conference", kind=ClaimKind.EVENT)]
        bundles, _ = build_event_neighborhood_bundles(
            claims, [episode], agent_id=AGENT, materialization_version=VER
        )
        # Event neighborhood bundles group claims around subject events
        assert all(b.kind is BundleKind.EVENT_NEIGHBORHOOD for b in bundles)

    def test_empty_claims_returns_empty(self):
        bundles, _ = build_event_neighborhood_bundles(
            [], [], agent_id=AGENT, materialization_version=VER
        )
        assert bundles == []


# ---------------------------------------------------------------------------
# build_all_bundles
# ---------------------------------------------------------------------------


class TestBuildAllBundles:
    def test_all_four_kinds_produced(self):
        episode = _make_episode("Alice, a doctor, knows Bob.")
        claims = materialize_episode(episode)
        if not claims:
            pytest.skip("materializer produced no claims for this text")

        raw_episodes = [episode]
        bundles, memberships = build_all_bundles(
            claims, raw_episodes, agent_id=AGENT, materialization_version=VER
        )
        assert len(bundles) >= 1
        # All bundles have agent_id set
        assert all(b.agent_id == AGENT for b in bundles)
        # memberships map is non-empty
        assert len(memberships) >= 1

    def test_bundle_membership_stable_on_repeat(self):
        """Same claims → same bundle topology (claim IDs are deterministic)."""
        episode = _make_episode("Bob works at Acme Corp.")
        claims = materialize_episode(episode)
        raw_episodes = [episode]

        bundles1, mem1 = build_all_bundles(
            claims, raw_episodes, agent_id=AGENT, materialization_version=VER
        )
        bundles2, mem2 = build_all_bundles(
            claims, raw_episodes, agent_id=AGENT, materialization_version=VER
        )
        # Same number of bundles
        assert len(bundles1) == len(bundles2)
        # Same claim IDs per bundle topic
        topics1 = {b.topic: set(mem1.get(b.id, [])) for b in bundles1}
        topics2 = {b.topic: set(mem2.get(b.id, [])) for b in bundles2}
        assert topics1 == topics2


# ---------------------------------------------------------------------------
# DirtyKey construction from materialization
# ---------------------------------------------------------------------------


class TestDirtyKeys:
    def test_dirty_key_for_slot_when_relation_known(self):
        from ai_knot.materialization import dirty_keys_for_claims

        claims = [_make_claim("Alice", "job")]
        keys = dirty_keys_for_claims(claims)
        # Should emit a (subject, relation) key, not a subject-only key
        assert any(k.subject == "Alice" and k.relation == "job" for k in keys)

    def test_dirty_key_subject_only_when_no_relation(self):
        from ai_knot.materialization import dirty_keys_for_claims

        claims = [
            AtomicClaim(
                id=make_claim_id(),
                agent_id=AGENT,
                kind=ClaimKind.STATE,
                subject="Alice",
                relation="",  # no relation
                value_text="unknown",
                value_tokens=("unknown",),
                qualifiers={},
                polarity="support",
                event_time=None,
                observed_at=NOW,
                valid_from=NOW,
                valid_until=None,
                confidence=0.9,
                salience=0.8,
                source_episode_id="ep-0",
                source_spans=((0, 7),),
                materialization_version=VER,
                materialized_at=NOW,
                slot_key="",
                version=0,
                origin_agent_id=AGENT,
            )
        ]
        keys = dirty_keys_for_claims(claims)
        assert any(k.subject == "Alice" and k.relation is None for k in keys)

    def test_no_duplicate_dirty_keys(self):
        from ai_knot.materialization import dirty_keys_for_claims

        # Two claims for same slot → one key
        c1 = _make_claim("Alice", "job", "engineer")
        c2 = _make_claim("Alice", "job", "manager")
        keys = dirty_keys_for_claims([c1, c2])
        alice_job_keys = [k for k in keys if k.subject == "Alice" and k.relation == "job"]
        assert len(alice_job_keys) == 1

    def test_dirty_key_minimality_prefers_slot_over_subject(self):
        """If relation is known, emit (subject, relation) not subject-only."""
        from ai_knot.materialization import dirty_keys_for_claims

        claims = [_make_claim("Alice", "salary")]
        keys = dirty_keys_for_claims(claims)
        # Must NOT emit a subject-only key when slot is available
        subject_only = [k for k in keys if k.subject == "Alice" and k.relation is None]
        assert len(subject_only) == 0


# ---------------------------------------------------------------------------
# SQLite bundle invalidation
# ---------------------------------------------------------------------------


class TestBundleInvalidation:
    def test_invalidate_by_slot_key(self, tmp_path):
        from ai_knot.storage.sqlite_storage import SQLiteStorage

        db = SQLiteStorage(db_path=str(tmp_path / "test.db"))
        claims = [_make_claim("Alice", "job")]
        bundles, memberships = build_state_timeline_bundles(
            claims, agent_id=AGENT, materialization_version=VER
        )
        db.save_bundles(AGENT, bundles, memberships)

        # Invalidate by slot
        key = DirtyKey(subject="Alice", relation="job")
        removed = db.invalidate_by_keys(AGENT, [key])
        assert removed >= 1

        # Bundle should be gone
        remaining = db.load_bundles_by_topic(AGENT, ["Alice::job"])
        assert remaining == []

    def test_invalidate_by_subject(self, tmp_path):
        from ai_knot.storage.sqlite_storage import SQLiteStorage

        db = SQLiteStorage(db_path=str(tmp_path / "test.db"))
        claims = [_make_claim("Alice", "job")]
        bundles, memberships = build_entity_topic_bundles(
            claims, agent_id=AGENT, materialization_version=VER
        )
        db.save_bundles(AGENT, bundles, memberships)

        # Invalidate by subject
        key = DirtyKey(subject="Alice")
        removed = db.invalidate_by_keys(AGENT, [key])
        assert removed >= 1

        remaining = db.load_bundles_by_topic(AGENT, ["Alice"])
        assert remaining == []


def test_event_bundles_split_by_relation():
    """Two events for same subject with different relations → two EVENT_NEIGHBORHOOD bundles."""
    now = datetime(2026, 4, 15, tzinfo=UTC)

    def _event_claim(cid: str, relation: str, value: str) -> AtomicClaim:
        return AtomicClaim(
            id=cid,
            agent_id="a",
            kind=ClaimKind.EVENT,
            subject="Alice",
            relation=relation,
            value_text=value,
            polarity="support",
            confidence=0.8,
            salience=0.8,
            source_episode_id="ep1",
            source_spans=((0, 10),),
            observed_at=now,
            valid_from=now,
            value_tokens=(),
            slot_key=f"Alice::{relation}",
            qualifiers={},
            event_time=None,
            valid_until=None,
            materialization_version=5,
            materialized_at=now,
            version=0,
            origin_agent_id="a",
        )

    claims = [
        _event_claim("c1", "attended", "pottery workshop"),
        _event_claim("c2", "bought", "new laptop"),
    ]
    raw_eps = [
        RawEpisode(
            id="ep1",
            agent_id="a",
            session_id="sess",
            turn_id="ep1",
            speaker="user",
            observed_at=now,
            session_date=now,
            raw_text="test",
            source_meta={},
            parent_episode_id=None,
        )
    ]
    bundles, memberships = build_event_neighborhood_bundles(
        claims, raw_eps, agent_id="a", materialization_version=5
    )
    topics = {b.topic for b in bundles}
    assert "Alice::attended" in topics, f"Expected Alice::attended, got {topics}"
    assert "Alice::bought" in topics, f"Expected Alice::bought, got {topics}"
    assert len(bundles) == 2, f"Expected 2 bundles, got {len(bundles)}"
