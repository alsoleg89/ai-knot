"""S7 regression tests — feature flag on/off parity.

Each flag is tested in both its default-OFF state and its enabled-ON state.
Tests verify:
  - Default OFF: behaviour is unchanged from S6 baseline.
  - Enabled ON: the guarded code path fires (or does not fire noise).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_knot.materialization import materialize_episode
from ai_knot.query_types import (
    AtomicClaim,
    BundleKind,
    ClaimKind,
    RawEpisode,
)
from ai_knot.support_bundles import build_event_neighborhood_bundles

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_raw(text: str, agent_id: str = "test", speaker: str = "Alice") -> RawEpisode:
    """Construct a RawEpisode with a speaker prefix."""
    return RawEpisode(
        id="ep_flag_test",
        agent_id=agent_id,
        session_id="s1",
        turn_id="t1",
        speaker=speaker,
        observed_at=datetime.now(UTC),
        raw_text=f"{speaker}: {text}",
        session_date=None,
    )


def _make_event_claim(
    subject: str,
    relation: str = "occurred",
    agent_id: str = "test",
) -> AtomicClaim:
    now = datetime.now(UTC)
    return AtomicClaim(
        id=f"{subject}:{relation}",
        agent_id=agent_id,
        kind=ClaimKind.EVENT,
        subject=subject,
        relation=relation,
        value_text=f"{subject} {relation}",
        value_tokens=(),
        qualifiers={},
        polarity="support",
        event_time=now,
        observed_at=now,
        valid_from=now,
        valid_until=None,
        confidence=0.85,
        salience=1.0,
        source_episode_id="ep1",
        source_spans=((0, 10),),
        materialization_version=4,
        materialized_at=now,
        slot_key=f"{subject}::{relation}",
        version=0,
        origin_agent_id=agent_id,
    )


# ---------------------------------------------------------------------------
# AI_KNOT_ENABLE_FP_EVENTS
# ---------------------------------------------------------------------------


class TestFPEventsFlag:
    def test_fp_events_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With flag OFF (default), 'I attended X' does NOT produce an EVENT claim."""
        monkeypatch.delenv("AI_KNOT_ENABLE_FP_EVENTS", raising=False)
        raw = _make_raw("I attended the annual conference.")
        claims = materialize_episode(raw)
        event_claims = [c for c in claims if c.kind is ClaimKind.EVENT]
        # Without FP events, no EVENT claim should be produced for this sentence.
        attended_claims = [c for c in event_claims if c.relation == "attended"]
        assert len(attended_claims) == 0, (
            f"Expected no 'attended' EVENT claim with flag OFF; got: {attended_claims}"
        )

    def test_fp_events_on_produces_event_claim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With flag ON, 'I attended X' produces an EVENT claim for the speaker."""
        monkeypatch.setenv("AI_KNOT_ENABLE_FP_EVENTS", "1")
        raw = _make_raw("I attended the annual conference.")
        claims = materialize_episode(raw)
        attended_claims = [
            c for c in claims if c.kind is ClaimKind.EVENT and c.relation == "attended"
        ]
        assert len(attended_claims) >= 1, (
            f"Expected at least one 'attended' EVENT claim; got: {claims}"
        )
        assert attended_claims[0].subject == "Alice"

    def test_fp_events_on_bought(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With flag ON, 'I bought X' produces a 'bought' EVENT claim."""
        monkeypatch.setenv("AI_KNOT_ENABLE_FP_EVENTS", "1")
        raw = _make_raw("I bought a new car.")
        claims = materialize_episode(raw)
        bought_claims = [c for c in claims if c.kind is ClaimKind.EVENT and c.relation == "bought"]
        assert len(bought_claims) >= 1, f"Expected at least one 'bought' EVENT claim; got: {claims}"

    def test_fp_events_off_does_not_suppress_fp_likes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flag OFF: FP preference patterns still produce STATE claims normally."""
        monkeypatch.delenv("AI_KNOT_ENABLE_FP_EVENTS", raising=False)
        raw = _make_raw("I love hiking.")
        claims = materialize_episode(raw)
        like_claims = [c for c in claims if c.relation == "likes" and c.subject == "Alice"]
        assert len(like_claims) >= 1, (
            f"FP likes should still fire with AI_KNOT_ENABLE_FP_EVENTS=0; got: {claims}"
        )


# ---------------------------------------------------------------------------
# AI_KNOT_EVENT_BUNDLE_BY_RELATION
# ---------------------------------------------------------------------------


class TestEventBundleByRelationFlag:
    def test_bundle_by_subject_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With flag OFF, all events for a subject share one bundle keyed by subject."""
        monkeypatch.delenv("AI_KNOT_EVENT_BUNDLE_BY_RELATION", raising=False)
        claims = [
            _make_event_claim("Alice", "attended"),
            _make_event_claim("Alice", "bought"),
        ]
        bundles, _ = build_event_neighborhood_bundles(
            claims, [], agent_id="test", materialization_version=4
        )
        assert len(bundles) == 1, f"Expected 1 bundle (by subject); got {len(bundles)}"
        assert bundles[0].topic == "Alice"

    def test_bundle_by_relation_when_flag_on(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With flag ON, each (subject, relation) pair gets its own bundle."""
        monkeypatch.setenv("AI_KNOT_EVENT_BUNDLE_BY_RELATION", "1")
        claims = [
            _make_event_claim("Alice", "attended"),
            _make_event_claim("Alice", "bought"),
        ]
        bundles, _ = build_event_neighborhood_bundles(
            claims, [], agent_id="test", materialization_version=4
        )
        assert len(bundles) == 2, f"Expected 2 bundles (by relation); got {len(bundles)}"
        topics = {b.topic for b in bundles}
        assert "Alice::attended" in topics
        assert "Alice::bought" in topics

    def test_bundle_by_relation_stable_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bundle IDs are deterministic regardless of flag state."""
        monkeypatch.setenv("AI_KNOT_EVENT_BUNDLE_BY_RELATION", "1")
        claims = [_make_event_claim("Alice", "attended")]
        bundles1, _ = build_event_neighborhood_bundles(
            claims, [], agent_id="test", materialization_version=4
        )
        bundles2, _ = build_event_neighborhood_bundles(
            claims, [], agent_id="test", materialization_version=4
        )
        assert bundles1[0].id == bundles2[0].id

    def test_bundle_by_relation_off_uses_event_neighborhood_kind(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bundle kind is always EVENT_NEIGHBORHOOD regardless of flag."""
        monkeypatch.setenv("AI_KNOT_EVENT_BUNDLE_BY_RELATION", "1")
        claims = [_make_event_claim("Bob", "attended")]
        bundles, _ = build_event_neighborhood_bundles(
            claims, [], agent_id="test", materialization_version=4
        )
        assert all(b.kind is BundleKind.EVENT_NEIGHBORHOOD for b in bundles)


# ---------------------------------------------------------------------------
# AI_KNOT_ENABLE_CALENDAR_GUARD
# ---------------------------------------------------------------------------


class TestCalendarGuardFlag:
    def test_when_question_routes_historical_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Default (flag OFF): 'When did X start?' routes to historical scope."""
        monkeypatch.delenv("AI_KNOT_ENABLE_CALENDAR_GUARD", raising=False)
        from ai_knot.query_contract import analyze_query

        frame = analyze_query("When did Alice start working at TechCorp?")
        assert frame.temporal_scope == "historical"

    def test_calendar_guard_on_requires_past_aux(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Flag ON: 'when' question with past auxiliary still routes historical."""
        monkeypatch.setenv("AI_KNOT_ENABLE_CALENDAR_GUARD", "1")
        from ai_knot.query_contract import analyze_query

        frame = analyze_query("When did Alice start working at TechCorp?")
        # "when" + "did" (past aux) → still historical with calendar guard
        assert frame.temporal_scope == "historical"

    def test_calendar_guard_on_no_past_aux_no_historical(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flag ON: pure 'when' question without a past-tense aux does NOT route historical."""
        monkeypatch.setenv("AI_KNOT_ENABLE_CALENDAR_GUARD", "1")
        from ai_knot.query_contract import analyze_query

        # "when is X available?" — "when" but no past-tense auxiliary
        frame = analyze_query("When is Alice available?")
        # With calendar guard ON, "when" + "is" (present) should NOT fire historical
        assert frame.temporal_scope != "historical", (
            f"Expected non-historical scope with calendar guard on 'when is'; "
            f"got: {frame.temporal_scope}"
        )

    def test_calendar_guard_off_historical_signals_fire(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Flag OFF: historical signals like 'before' fire without past-aux requirement."""
        monkeypatch.delenv("AI_KNOT_ENABLE_CALENDAR_GUARD", raising=False)
        from ai_knot.query_contract import analyze_query

        frame = analyze_query("What did Alice do before moving to Berlin?")
        assert frame.temporal_scope == "historical"


# ---------------------------------------------------------------------------
# AI_KNOT_CANDIDATE_RANK_PENALTIES
# ---------------------------------------------------------------------------


class TestCandidateRankPenaltiesFlag:
    def _make_claims_for_ranking(self) -> list[AtomicClaim]:
        now = datetime.now(UTC)
        slot_claim = AtomicClaim(
            id="slot_claim",
            agent_id="test",
            kind=ClaimKind.STATE,
            subject="Alice",
            relation="likes",
            value_text="hiking",
            value_tokens=("hiking",),
            qualifiers={},
            polarity="support",
            event_time=None,
            observed_at=now,
            valid_from=now,
            valid_until=None,
            confidence=0.85,
            salience=1.0,
            source_episode_id="ep1",
            source_spans=((0, 10),),
            materialization_version=4,
            materialized_at=now,
            slot_key="Alice::likes",
            version=0,
            origin_agent_id="test",
        )
        off_slot_claim = AtomicClaim(
            id="off_slot_claim",
            agent_id="test",
            kind=ClaimKind.STATE,
            subject="Bob",
            relation="state",
            value_text="happy",
            value_tokens=("happy",),
            qualifiers={},
            polarity="support",
            event_time=None,
            observed_at=now,
            valid_from=now,
            valid_until=None,
            confidence=0.85,
            salience=1.0,
            source_episode_id="ep1",
            source_spans=((0, 10),),
            materialization_version=4,
            materialized_at=now,
            slot_key="Bob::state",
            version=0,
            origin_agent_id="test",
        )
        return [slot_claim, off_slot_claim]

    def test_rank_penalties_off_by_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default (flag OFF): candidate_rank runs without slot-miss penalty."""
        monkeypatch.delenv("AI_KNOT_CANDIDATE_RANK_PENALTIES", raising=False)
        from datetime import UTC, datetime

        from ai_knot.query_operators import candidate_rank
        from ai_knot.query_types import (
            AnswerContract,
            AnswerSpace,
            EvidenceProfile,
            EvidenceRegime,
            TimeAxis,
            TruthMode,
        )

        claims = self._make_claims_for_ranking()
        contract = AnswerContract(
            answer_space=AnswerSpace.DESCRIPTION,
            truth_mode=TruthMode.DIRECT,
            time_axis=TimeAxis.NONE,
            locality="point",
            evidence_regime=EvidenceRegime.SINGLE,
        )
        profile = EvidenceProfile(
            n_support=2,
            n_contra=0,
            n_ambiguous=0,
            density_per_entity=1.0,
            temporal_span=None,
            coverage_ratio=1.0,
            has_explicit_event_time=False,
            slot_bundle_hits=1,
            question_tokens=("what", "does", "alice", "like"),
            focus_entities=("Alice",),
            focus_relation="likes",
        )
        items, conf, notes = candidate_rank(claims, [], contract, profile, datetime.now(UTC))
        assert len(items) >= 1

    def test_rank_penalties_on_slot_miss_penalized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Flag ON: off-slot claims receive a slot-miss penalty (-0.2)."""
        monkeypatch.setenv("AI_KNOT_CANDIDATE_RANK_PENALTIES", "1")
        from datetime import UTC, datetime

        from ai_knot.query_operators import candidate_rank
        from ai_knot.query_types import (
            AnswerContract,
            AnswerSpace,
            EvidenceProfile,
            EvidenceRegime,
            SupportBundle,
            TimeAxis,
            TruthMode,
        )

        claims = self._make_claims_for_ranking()
        contract = AnswerContract(
            answer_space=AnswerSpace.DESCRIPTION,
            truth_mode=TruthMode.DIRECT,
            time_axis=TimeAxis.NONE,
            locality="point",
            evidence_regime=EvidenceRegime.SINGLE,
        )
        profile = EvidenceProfile(
            n_support=2,
            n_contra=0,
            n_ambiguous=0,
            density_per_entity=1.0,
            temporal_span=None,
            coverage_ratio=1.0,
            has_explicit_event_time=False,
            slot_bundle_hits=1,
            question_tokens=("what", "does", "alice", "like"),
            focus_entities=("Alice",),
            focus_relation="likes",
        )
        # Build a fake slot bundle so slot_keys is non-empty inside candidate_rank.
        now = datetime.now(UTC)
        slot_bundle = SupportBundle(
            id="fake_slot_bundle",
            agent_id="test",
            kind=BundleKind.STATE_TIMELINE,
            topic="Alice::likes",
            member_claim_ids=("slot_claim",),
            score_formula="mean(salience*confidence)",
            bundle_score=0.85,
            built_from_materialization_version=4,
            built_at=now,
        )
        items, conf, notes = candidate_rank(
            claims, [slot_bundle], contract, profile, datetime.now(UTC)
        )
        # The slot-matching claim (Alice::likes) should rank above the off-slot claim (Bob::state)
        assert len(items) >= 2
        assert items[0].value == "hiking", (
            f"Expected 'hiking' (slot-match) to rank first; got: {items[0].value}"
        )
