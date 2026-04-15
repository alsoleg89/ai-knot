"""Tests for query_operators — pure functions, no storage access."""

from __future__ import annotations

import uuid as _uuid
from datetime import UTC, datetime

import pytest

from ai_knot.query_operators import (
    OPERATORS,
    bounded_hypothesis_test,
    candidate_rank,
    choose_strategy,
    exact_state,
    narrative_cluster_render,
    set_collect,
    time_resolve,
)
from ai_knot.query_types import (
    AnswerContract,
    AnswerSpace,
    AtomicClaim,
    BundleKind,
    ClaimKind,
    EvidenceProfile,
    EvidenceRegime,
    QueryFrame,
    SupportBundle,
    TimeAxis,
    TruthMode,
    make_bundle_id,
    make_episode_id,
)

NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def _ep_id() -> str:
    return make_episode_id("agent", "sess", "turn")


def _claim(
    subject: str = "Alice",
    relation: str = "job",
    value: str = "doctor",
    kind: ClaimKind = ClaimKind.STATE,
    polarity: str = "support",
    confidence: float = 0.9,
    salience: float = 1.0,
    event_time: datetime | None = None,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
    qualifiers: dict | None = None,
) -> AtomicClaim:
    ep = _ep_id()
    return AtomicClaim(
        id=str(_uuid.uuid4()),
        agent_id="agent",
        kind=kind,
        subject=subject,
        relation=relation,
        value_text=value,
        value_tokens=(value,),
        qualifiers=qualifiers or {},
        polarity=polarity,
        event_time=event_time,
        observed_at=NOW,
        valid_from=valid_from or NOW,
        valid_until=valid_until,
        confidence=confidence,
        salience=salience,
        source_episode_id=ep,
        source_spans=((0, len(value)),),
        materialization_version=1,
        materialized_at=NOW,
        slot_key=f"{subject}::{relation}",
        version=1,
        origin_agent_id="agent",
    )


def _contract(
    answer_space: AnswerSpace = AnswerSpace.DESCRIPTION,
    time_axis: TimeAxis = TimeAxis.NONE,
    truth_mode: TruthMode = TruthMode.DIRECT,
    evidence_regime: EvidenceRegime = EvidenceRegime.SINGLE,
    uncertainty_threshold: float = 0.25,
) -> AnswerContract:
    return AnswerContract(
        answer_space=answer_space,
        truth_mode=truth_mode,
        time_axis=time_axis,
        locality="point",
        evidence_regime=evidence_regime,
        uncertainty_threshold=uncertainty_threshold,
    )


def _frame(
    answer_space: AnswerSpace = AnswerSpace.DESCRIPTION,
    focus_entities: tuple[str, ...] = ("Alice",),
    evidence_regime: EvidenceRegime = EvidenceRegime.SINGLE,
) -> QueryFrame:
    return QueryFrame(
        focus_entities=focus_entities,
        target_kind="state",
        answer_space=answer_space,
        temporal_scope="none",
        epistemic_mode=TruthMode.DIRECT,
        locality="point",
        evidence_regime=evidence_regime,
        focus_relation=None,
    )


def _profile(
    n_support: int = 1,
    n_contra: int = 0,
    n_ambiguous: int = 0,
    density: float = 1.0,
    coverage: float = 1.0,
    temporal_span=None,
    has_explicit_event_time: bool = False,
) -> EvidenceProfile:
    return EvidenceProfile(
        n_support=n_support,
        n_contra=n_contra,
        n_ambiguous=n_ambiguous,
        density_per_entity=density,
        temporal_span=temporal_span,
        coverage_ratio=coverage,
        has_explicit_event_time=has_explicit_event_time,
    )


def _bundle(topic: str = "Alice") -> SupportBundle:
    return SupportBundle(
        id=make_bundle_id(),
        agent_id="agent",
        kind=BundleKind.ENTITY_TOPIC,
        topic=topic,
        member_claim_ids=(),
        score_formula="test",
        bundle_score=0.5,
        built_from_materialization_version=1,
        built_at=NOW,
    )


# ---------------------------------------------------------------------------
# choose_strategy
# ---------------------------------------------------------------------------


class TestChooseStrategy:
    def test_set_always_set_collect(self):
        frame = _frame(answer_space=AnswerSpace.SET, evidence_regime=EvidenceRegime.AGGREGATE)
        contract = _contract(answer_space=AnswerSpace.SET, evidence_regime=EvidenceRegime.AGGREGATE)
        profile = _profile(n_support=5)
        assert choose_strategy(frame, contract, profile) == "set_collect"

    def test_temporal_event_time_resolve(self):
        frame = _frame()
        contract = _contract(time_axis=TimeAxis.EVENT)
        profile = _profile(n_support=1)
        assert choose_strategy(frame, contract, profile) == "time_resolve"

    def test_temporal_interval_time_resolve(self):
        frame = _frame()
        contract = _contract(time_axis=TimeAxis.INTERVAL)
        profile = _profile(n_support=2)
        assert choose_strategy(frame, contract, profile) == "time_resolve"

    def test_bool_with_direct_evidence_exact_state(self):
        """BOOL with direct single evidence → exact_state, not hypothesis."""
        frame = _frame(answer_space=AnswerSpace.BOOL, evidence_regime=EvidenceRegime.SINGLE)
        contract = _contract(
            answer_space=AnswerSpace.BOOL,
            evidence_regime=EvidenceRegime.SINGLE,
        )
        profile = _profile(n_support=1, n_contra=0)
        strategy = choose_strategy(frame, contract, profile)
        assert strategy == "exact_state"

    def test_bool_without_direct_evidence_hypothesis(self):
        """BOOL with distributed/conflicting evidence → bounded_hypothesis_test."""
        frame = _frame(
            answer_space=AnswerSpace.BOOL,
            evidence_regime=EvidenceRegime.SUPPORT_VS_CONTRA,
        )
        contract = _contract(
            answer_space=AnswerSpace.BOOL,
            evidence_regime=EvidenceRegime.SUPPORT_VS_CONTRA,
        )
        # No direct SINGLE evidence
        profile = _profile(n_support=2, n_contra=1)
        strategy = choose_strategy(frame, contract, profile)
        assert strategy == "bounded_hypothesis_test"

    def test_multi_support_candidate_rank(self):
        frame = _frame()
        contract = _contract(evidence_regime=EvidenceRegime.AGGREGATE)
        profile = _profile(n_support=3)
        assert choose_strategy(frame, contract, profile) == "candidate_rank"

    def test_no_support_narrative_render(self):
        frame = _frame()
        contract = _contract()
        profile = _profile(n_support=0)
        assert choose_strategy(frame, contract, profile) == "narrative_cluster_render"


# ---------------------------------------------------------------------------
# exact_state
# ---------------------------------------------------------------------------


class TestExactState:
    def test_returns_single_best_item(self):
        claims = [
            _claim(value="doctor", confidence=0.9, salience=1.0),
            _claim(value="nurse", confidence=0.6, salience=1.0),
        ]
        items, conf, notes = exact_state(claims, [], _contract(), _profile(), NOW)
        assert len(items) == 1
        assert items[0].value == "doctor"
        assert conf > 0

    def test_no_support_returns_empty(self):
        claims = [_claim(polarity="contra")]
        items, conf, notes = exact_state(claims, [], _contract(), _profile(n_support=0), NOW)
        assert items == []
        assert conf == 0.0

    def test_prefers_state_kind_over_generic(self):
        state_claim = _claim(value="doctor", kind=ClaimKind.STATE, confidence=0.7)
        generic_claim = _claim(value="manager", kind=ClaimKind.EVENT, confidence=0.6)
        items, _, _ = exact_state([generic_claim, state_claim], [], _contract(), _profile(), NOW)
        assert items[0].value == "doctor"


# ---------------------------------------------------------------------------
# set_collect
# ---------------------------------------------------------------------------


class TestSetCollect:
    def test_collects_all_distinct_values(self):
        claims = [
            _claim(relation="hobby", value="reading"),
            _claim(relation="hobby", value="cycling"),
            _claim(relation="hobby", value="chess"),
        ]
        contract = _contract(answer_space=AnswerSpace.SET, evidence_regime=EvidenceRegime.AGGREGATE)
        items, _, _ = set_collect(claims, [], contract, _profile(n_support=3), NOW)
        values = {i.value for i in items}
        assert {"reading", "cycling", "chess"} == values

    def test_deduplicates_same_value(self):
        claims = [
            _claim(relation="hobby", value="reading", confidence=0.9),
            _claim(relation="hobby", value="reading", confidence=0.7),
        ]
        contract = _contract(answer_space=AnswerSpace.SET, evidence_regime=EvidenceRegime.AGGREGATE)
        items, _, _ = set_collect(claims, [], contract, _profile(n_support=2), NOW)
        assert len(items) == 1
        assert items[0].value == "reading"
        assert items[0].confidence == pytest.approx(0.9, abs=0.1)

    def test_excludes_contra_claims(self):
        claims = [
            _claim(relation="hobby", value="reading", polarity="support"),
            _claim(relation="hobby", value="gaming", polarity="contra"),
        ]
        contract = _contract(answer_space=AnswerSpace.SET, evidence_regime=EvidenceRegime.AGGREGATE)
        items, _, _ = set_collect(claims, [], contract, _profile(n_support=1), NOW)
        values = {i.value for i in items}
        assert "gaming" not in values
        assert "reading" in values

    def test_does_not_collapse_different_values(self):
        """Different values under same slot_key must remain separate items."""
        claims = [
            _claim(subject="Alice", relation="pet", value="cat"),
            _claim(subject="Alice", relation="pet", value="dog"),
        ]
        contract = _contract(answer_space=AnswerSpace.SET, evidence_regime=EvidenceRegime.AGGREGATE)
        items, _, _ = set_collect(claims, [], contract, _profile(n_support=2), NOW)
        assert len(items) == 2


# ---------------------------------------------------------------------------
# time_resolve
# ---------------------------------------------------------------------------


class TestTimeResolve:
    def test_prefers_event_time(self):
        event_dt = datetime(2023, 5, 7, tzinfo=UTC)
        claims = [_claim(event_time=event_dt, qualifiers={"date_token": "2023-05-07"})]
        contract = _contract(time_axis=TimeAxis.EVENT)
        items, _, _ = time_resolve(claims, [], contract, _profile(), NOW)
        assert items
        assert "2023-05-07" in items[0].value

    def test_interval_returns_range(self):
        t1 = datetime(2023, 1, 1, tzinfo=UTC)
        t2 = datetime(2023, 6, 1, tzinfo=UTC)
        claims = [
            _claim(event_time=t1, valid_from=t1),
            _claim(event_time=t2, valid_from=t2),
        ]
        contract = _contract(time_axis=TimeAxis.INTERVAL)
        items, conf, _ = time_resolve(claims, [], contract, _profile(n_support=2), NOW)
        assert len(items) == 1
        assert "–" in items[0].value  # contains separator

    def test_no_claims_returns_empty(self):
        items, conf, _ = time_resolve(
            [], [], _contract(time_axis=TimeAxis.EVENT), _profile(n_support=0), NOW
        )
        assert items == []
        assert conf == 0.0


# ---------------------------------------------------------------------------
# candidate_rank
# ---------------------------------------------------------------------------


class TestCandidateRank:
    def test_sorts_by_confidence_salience(self):
        claims = [
            _claim(value="low", confidence=0.3, salience=0.5),
            _claim(value="high", confidence=0.9, salience=0.9),
            _claim(value="mid", confidence=0.6, salience=0.7),
        ]
        items, _, _ = candidate_rank(claims, [], _contract(), _profile(n_support=3), NOW)
        assert items[0].value == "high"

    def test_no_support_returns_empty(self):
        items, conf, _ = candidate_rank([], [], _contract(), _profile(n_support=0), NOW)
        assert items == []

    def test_returns_at_most_10_items(self):
        claims = [_claim(value=f"val{i}", confidence=0.5) for i in range(20)]
        items, _, _ = candidate_rank(claims, [], _contract(), _profile(n_support=20), NOW)
        assert len(items) <= 10


# ---------------------------------------------------------------------------
# bounded_hypothesis_test
# ---------------------------------------------------------------------------


class TestBoundedHypothesisTest:
    def test_yes_when_support_dominates(self):
        profile = _profile(n_support=5, n_contra=0, n_ambiguous=0)
        items, conf, _ = bounded_hypothesis_test(
            [], [], _contract(answer_space=AnswerSpace.BOOL), profile, NOW
        )
        assert items[0].value == "yes"
        assert conf > 0.5

    def test_no_when_contra_dominates(self):
        profile = _profile(n_support=0, n_contra=5, n_ambiguous=0)
        items, conf, _ = bounded_hypothesis_test(
            [], [], _contract(answer_space=AnswerSpace.BOOL), profile, NOW
        )
        assert items[0].value == "no"

    def test_uncertain_when_score_below_threshold(self):
        # score = 1 - 0 - 0.5*0 = 1 — above threshold 0.25; need tie
        profile = _profile(n_support=1, n_contra=1, n_ambiguous=0)
        contract = _contract(answer_space=AnswerSpace.BOOL, uncertainty_threshold=5.0)
        items, _, _ = bounded_hypothesis_test([], [], contract, profile, NOW)
        assert items[0].value == "uncertain"


# ---------------------------------------------------------------------------
# narrative_cluster_render
# ---------------------------------------------------------------------------


class TestNarrativeClusterRender:
    def test_joins_claims_deterministically(self):
        claims = [
            _claim(value="Alice works at Acme"),
            _claim(value="Alice lives in Paris"),
        ]
        items, conf, _ = narrative_cluster_render(claims, [], _contract(), _profile(), NOW)
        assert items
        assert "Alice works at Acme" in items[0].value or "Alice lives in Paris" in items[0].value

    def test_renderer_is_called_if_provided(self):
        claims = [_claim(value="some fact")]
        called_with: list[str] = []

        def renderer(text: str) -> str:
            called_with.append(text)
            return "rendered: " + text

        items, _, _ = narrative_cluster_render(
            claims, [], _contract(), _profile(), NOW, renderer=renderer
        )
        assert called_with
        assert "rendered:" in items[0].value

    def test_renderer_failure_falls_back_to_deterministic(self):
        claims = [_claim(value="fact one")]

        def bad_renderer(text: str) -> str:
            raise RuntimeError("LLM exploded")

        items, _, notes = narrative_cluster_render(
            claims, [], _contract(), _profile(), NOW, renderer=bad_renderer
        )
        assert items
        assert any("renderer failed" in n for n in notes)

    def test_no_support_returns_empty(self):
        claims = [_claim(polarity="contra", value="x")]
        items, conf, _ = narrative_cluster_render(
            claims, [], _contract(), _profile(n_support=0), NOW
        )
        assert items == []


# ---------------------------------------------------------------------------
# OPERATORS registry
# ---------------------------------------------------------------------------


def test_operators_registry_complete():
    expected = {
        "exact_state",
        "set_collect",
        "time_resolve",
        "candidate_rank",
        "bounded_hypothesis_test",
        "narrative_cluster_render",
    }
    assert set(OPERATORS.keys()) == expected


# ---------------------------------------------------------------------------
# Relevance-aware operator fixes
# ---------------------------------------------------------------------------


class TestExactStateRelevance:
    def test_prefers_slot_matched_claim_over_unrelated(self):
        """exact_state must prefer the slot-matched claim over a more recent but irrelevant one."""
        relevant = _claim(
            subject="Evan",
            relation="drive",
            value="a pickup truck",
            confidence=0.8,
            salience=1.0,
        )
        # Irrelevant claim — different slot, slightly higher confidence.
        irrelevant = _claim(
            subject="Evan",
            relation="state",
            value="happy",
            confidence=0.85,
            salience=1.0,
        )
        profile = EvidenceProfile(
            n_support=2,
            n_contra=0,
            n_ambiguous=0,
            density_per_entity=2.0,
            temporal_span=None,
            coverage_ratio=1.0,
            has_explicit_event_time=False,
            slot_bundle_hits=1,
            focus_entities=("Evan",),
            focus_relation="drive",
            question_tokens=("what", "car", "evan", "drive"),
        )
        items, _, _ = exact_state([relevant, irrelevant], [], _contract(), profile, NOW)
        assert items[0].value == "a pickup truck", (
            f"Expected slot-matched claim 'a pickup truck', got {items[0].value!r}"
        )


class TestTimeResolveNoSessionDate:
    def test_returns_empty_when_no_explicit_event_time(self):
        """time_resolve must return empty when no claim has an explicit date_token."""
        from ai_knot.query_types import TimeAxis

        # Claim with valid_from (session date) but no explicit event_time/date_token.
        claim_no_date = _claim(
            kind=ClaimKind.STATE,
            event_time=None,
            valid_from=datetime(2024, 1, 1, tzinfo=UTC),
            qualifiers={},  # no date_token
        )
        contract = _contract(time_axis=TimeAxis.EVENT)
        profile = EvidenceProfile(
            n_support=1,
            n_contra=0,
            n_ambiguous=0,
            density_per_entity=1.0,
            temporal_span=None,
            coverage_ratio=1.0,
            has_explicit_event_time=False,
            explicit_time_hits=0,
        )
        items, conf, notes = time_resolve([claim_no_date], [], contract, profile, NOW)
        assert items == [], f"Expected empty items when no explicit event time, got {items}"
        assert conf == 0.0


class TestBoundedHypothesisRelevance:
    def test_returns_uncertain_when_no_subject_match(self):
        """bounded_hypothesis_test must return 'uncertain' for unrelated subject claims."""
        # Claims about "Dave" when question is about "Caroline".
        unrelated_claims = [
            _claim(subject="Dave", relation="state", value="happy", polarity="support"),
            _claim(subject="Dave", relation="job", value="carpenter", polarity="support"),
        ]
        profile = EvidenceProfile(
            n_support=2,
            n_contra=0,
            n_ambiguous=0,
            density_per_entity=2.0,
            temporal_span=None,
            coverage_ratio=1.0,
            has_explicit_event_time=False,
            focus_entities=("Caroline",),
            focus_relation=None,
        )
        items, conf, notes = bounded_hypothesis_test(
            unrelated_claims, [], _contract(answer_space=AnswerSpace.BOOL), profile, NOW
        )
        assert items[0].value == "uncertain", (
            f"Expected 'uncertain' for unrelated subject claims, got {items[0].value!r}"
        )

    def test_passes_when_subject_matches(self):
        """bounded_hypothesis_test succeeds when claims match focus entity."""
        matching_claims = [
            _claim(subject="Bob", relation="job", value="carpenter", polarity="support"),
        ]
        profile = EvidenceProfile(
            n_support=1,
            n_contra=0,
            n_ambiguous=0,
            density_per_entity=1.0,
            temporal_span=None,
            coverage_ratio=1.0,
            has_explicit_event_time=False,
            focus_entities=("Bob",),
            focus_relation=None,
        )
        items, conf, _ = bounded_hypothesis_test(
            matching_claims, [], _contract(answer_space=AnswerSpace.BOOL), profile, NOW
        )
        assert items[0].value == "yes"


class TestChooseStrategyFixes:
    def test_description_with_fallback_and_no_slot_bundle_goes_to_candidate_rank(self):
        """choose_strategy must avoid exact_state for fallback-only description queries."""
        frame = _frame(answer_space=AnswerSpace.DESCRIPTION, evidence_regime=EvidenceRegime.SINGLE)
        contract = _contract(
            answer_space=AnswerSpace.DESCRIPTION, evidence_regime=EvidenceRegime.SINGLE
        )
        profile = EvidenceProfile(
            n_support=5,
            n_contra=0,
            n_ambiguous=0,
            density_per_entity=5.0,
            temporal_span=None,
            coverage_ratio=1.0,
            has_explicit_event_time=False,
            slot_bundle_hits=0,
            fallback_used=True,
            focus_entities=("Alice",),
        )
        strategy = choose_strategy(frame, contract, profile)
        assert strategy != "exact_state", (
            f"description+fallback+no_slot should not route to exact_state, got {strategy!r}"
        )

    def test_description_with_slot_bundle_can_use_exact_state(self):
        """choose_strategy may use exact_state when a slot bundle is present."""
        frame = _frame(answer_space=AnswerSpace.DESCRIPTION, evidence_regime=EvidenceRegime.SINGLE)
        contract = _contract(
            answer_space=AnswerSpace.DESCRIPTION, evidence_regime=EvidenceRegime.SINGLE
        )
        profile = EvidenceProfile(
            n_support=1,
            n_contra=0,
            n_ambiguous=0,
            density_per_entity=1.0,
            temporal_span=None,
            coverage_ratio=1.0,
            has_explicit_event_time=False,
            slot_bundle_hits=1,
            fallback_used=False,
        )
        strategy = choose_strategy(frame, contract, profile)
        assert strategy == "exact_state"
