"""End-to-end tests for the query runtime pipeline."""

from __future__ import annotations

from datetime import UTC, datetime

from ai_knot.knowledge import KnowledgeBase
from ai_knot.query_types import QueryAnswer
from ai_knot.storage.sqlite_storage import SQLiteStorage


def _make_kb(tmp_path, agent_id="agent"):
    db = str(tmp_path / f"{agent_id}.db")
    storage = SQLiteStorage(db_path=db)
    return KnowledgeBase(agent_id=agent_id, storage=storage)


NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Basic query pipeline
# ---------------------------------------------------------------------------


class TestQueryPipeline:
    def test_query_empty_db_returns_answer(self, tmp_path):
        kb = _make_kb(tmp_path)
        answer = kb.query("What is Alice's job?", now=NOW)
        assert isinstance(answer, QueryAnswer)
        assert isinstance(answer.text, str)
        assert answer.trace is not None

    def test_query_with_episodes_has_trace_fields(self, tmp_path):
        kb = _make_kb(tmp_path)
        kb.ingest_episode(
            session_id="sess-0",
            turn_id="turn-0",
            speaker="user",
            observed_at=NOW,
            raw_text="Alice works as a software engineer at TechCorp.",
        )
        answer = kb.query("What does Alice do?", now=NOW)
        trace = answer.trace
        assert trace.question == "What does Alice do?"
        assert trace.strategy in {
            "exact_state",
            "set_collect",
            "time_resolve",
            "candidate_rank",
            "bounded_hypothesis_test",
            "narrative_cluster_render",
        }
        assert trace.latency_ms > 0

    def test_query_trace_has_all_required_fields(self, tmp_path):
        kb = _make_kb(tmp_path)
        kb.ingest_episode(
            session_id="sess-0",
            turn_id="turn-0",
            speaker="user",
            observed_at=NOW,
            raw_text="Bob is a doctor.",
        )
        answer = kb.query("Is Bob a doctor?", now=NOW)
        trace = answer.trace
        # All 10 trace fields
        assert hasattr(trace, "question")
        assert hasattr(trace, "frame")
        assert hasattr(trace, "contract")
        assert hasattr(trace, "retrieved_bundle_ids")
        assert hasattr(trace, "expanded_claim_ids")
        assert hasattr(trace, "evidence_profile")
        assert hasattr(trace, "strategy")
        assert hasattr(trace, "decision_notes")
        assert hasattr(trace, "latency_ms")

    def test_query_json_is_serializable(self, tmp_path):
        kb = _make_kb(tmp_path)
        kb.ingest_episode(
            session_id="sess-0",
            turn_id="turn-0",
            speaker="user",
            observed_at=NOW,
            raw_text="Alice likes reading and cycling.",
        )
        result = kb.query_json("What does Alice like?", now=NOW)
        assert isinstance(result, dict)
        assert "text" in result
        assert "confidence" in result

    def test_set_question_strategy_is_set_collect(self, tmp_path):
        kb = _make_kb(tmp_path)
        for i, hobby in enumerate(["reading", "cycling", "chess"]):
            kb.ingest_episode(
                session_id="sess-0",
                turn_id=f"turn-{i}",
                speaker="user",
                observed_at=NOW,
                raw_text=f"Alice enjoys {hobby} in her free time.",
            )
        answer = kb.query("List all hobbies Alice has.", now=NOW)
        # SET question → set_collect strategy
        assert answer.trace.strategy == "set_collect"

    def test_explain_query_returns_trace(self, tmp_path):
        kb = _make_kb(tmp_path)
        trace = kb.explain_query("What is Bob's job?", now=NOW)
        assert trace is not None
        assert hasattr(trace, "strategy")
        assert hasattr(trace, "frame")

    def test_confidence_in_range(self, tmp_path):
        kb = _make_kb(tmp_path)
        kb.ingest_episode(
            session_id="sess-0",
            turn_id="turn-0",
            speaker="user",
            observed_at=NOW,
            raw_text="Alice is 30 years old.",
        )
        answer = kb.query("How old is Alice?", now=NOW)
        assert 0.0 <= answer.confidence <= 1.0


# ---------------------------------------------------------------------------
# Ingest + rebuild + query cycle
# ---------------------------------------------------------------------------


class TestRebuildAndQuery:
    def test_rebuild_then_query_stable(self, tmp_path):
        kb = _make_kb(tmp_path)
        kb.ingest_episode(
            session_id="sess-0",
            turn_id="turn-0",
            speaker="user",
            observed_at=NOW,
            raw_text="Alice works at TechCorp as an engineer.",
            materialize=False,
        )
        kb.rebuild_materialized(force=True)
        answer = kb.query("Where does Alice work?", now=NOW)
        assert isinstance(answer.text, str)

    def test_double_rebuild_idempotent_query(self, tmp_path):
        kb = _make_kb(tmp_path)
        for i in range(5):
            kb.ingest_episode(
                session_id="sess-0",
                turn_id=f"turn-{i}",
                speaker="user",
                observed_at=NOW,
                raw_text=f"Alice has skill {['python', 'java', 'go', 'rust', 'c++'][i]}.",
                materialize=False,
            )
        kb.rebuild_materialized(force=True)
        answer1 = kb.query("What skills does Alice have?", now=NOW)
        kb.rebuild_materialized(force=True)
        answer2 = kb.query("What skills does Alice have?", now=NOW)
        # Strategy must be identical (same data)
        assert answer1.trace.strategy == answer2.trace.strategy


# ---------------------------------------------------------------------------
# Legacy recall compatibility
# ---------------------------------------------------------------------------


class TestLegacyRecallCompat:
    def test_legacy_recall_still_works(self, tmp_path):
        """kb.recall() must still return text after new methods are added."""
        kb = _make_kb(tmp_path)
        kb.add("Alice works as a nurse at City Hospital.")
        result = kb.recall("What does Alice do?", top_k=3)
        assert isinstance(result, str)

    def test_legacy_add_still_works(self, tmp_path):
        kb = _make_kb(tmp_path)
        fact = kb.add("Bob is a software engineer.", importance=0.9)
        assert fact.id
        assert fact.content == "Bob is a software engineer."


# ---------------------------------------------------------------------------
# Query stability
# ---------------------------------------------------------------------------


class TestQueryStability:
    def test_same_input_same_strategy(self, tmp_path):
        """Same question on same data → same strategy (deterministic)."""
        kb = _make_kb(tmp_path)
        kb.ingest_episode(
            session_id="sess-0",
            turn_id="turn-0",
            speaker="user",
            observed_at=NOW,
            raw_text="Carol is a teacher at Elm School.",
        )
        answer1 = kb.query("What is Carol's job?", now=NOW)
        answer2 = kb.query("What is Carol's job?", now=NOW)
        assert answer1.trace.strategy == answer2.trace.strategy

    def test_batch_ingest_order_irrelevant_to_strategy(self, tmp_path):
        """Permuting ingest order must not change the query strategy."""
        db_path = str(tmp_path / "order_test.db")

        def _make_and_query(order: list[int]) -> str:
            import os

            if os.path.exists(db_path):
                os.remove(db_path)
            storage = SQLiteStorage(db_path=db_path)
            kb = KnowledgeBase(agent_id="agent", storage=storage)
            texts = [
                "Alice works at Acme.",
                "Alice is a manager.",
                "Alice lives in Paris.",
            ]
            for i in order:
                kb.ingest_episode(
                    session_id="sess-0",
                    turn_id=f"turn-{i}",
                    speaker="user",
                    observed_at=NOW,
                    raw_text=texts[i],
                )
            return kb.query("What is Alice's job?", now=NOW).trace.strategy

        s1 = _make_and_query([0, 1, 2])
        s2 = _make_and_query([2, 0, 1])
        # Same facts, different order → same strategy (deterministic materializer)
        assert s1 == s2


# ---------------------------------------------------------------------------
# Evidence profile: temporal anchor signal
# ---------------------------------------------------------------------------


def _make_claim(
    cid: str,
    kind: object,
    subject: str,
    relation: str,
    value_text: str,
    source_episode_id: str,
    now: object,
    qualifiers: dict,
    polarity: str = "support",
) -> object:
    """Helper: build a minimal AtomicClaim with all required fields."""
    from ai_knot.query_types import AtomicClaim

    return AtomicClaim(
        id=cid,
        agent_id="a",
        kind=kind,  # type: ignore[arg-type]
        subject=subject,
        relation=relation,
        value_text=value_text,
        value_tokens=(),
        qualifiers=qualifiers,
        polarity=polarity,
        event_time=None,
        observed_at=now,  # type: ignore[arg-type]
        valid_from=now,  # type: ignore[arg-type]
        valid_until=None,
        confidence=0.9,
        salience=0.9,
        source_episode_id=source_episode_id,
        source_spans=((0, 10),),
        materialization_version=1,
        materialized_at=now,  # type: ignore[arg-type]
        slot_key=f"{subject}::{relation}",
        version=1,
        origin_agent_id="a",
    )


def test_evidence_profile_has_temporal_anchor():
    """_build_evidence_profile sets has_temporal_anchor for session-anchored claims."""
    from datetime import UTC, datetime

    from ai_knot.query_runtime import _build_evidence_profile
    from ai_knot.query_types import (
        AnswerContract,
        AnswerSpace,
        ClaimKind,
        EvidenceRegime,
        QueryFrame,
        TimeAxis,
        TruthMode,
    )

    now = datetime(2026, 4, 15, tzinfo=UTC)

    claim_anchored = _make_claim(
        "c1",
        ClaimKind.EVENT,
        "Alice",
        "attended",
        "workshop",
        "ep1",
        now,
        {"time_anchor": "session_date", "relative_time": "yesterday"},
    )
    claim_plain = _make_claim(
        "c2",
        ClaimKind.STATE,
        "Alice",
        "likes",
        "coffee",
        "ep2",
        now,
        {},
    )

    frame = QueryFrame(
        focus_entities=("Alice",),
        target_kind="event",
        answer_space=AnswerSpace.SCALAR,
        temporal_scope="historical",
        epistemic_mode=TruthMode.DIRECT,
        locality="point",
        evidence_regime=EvidenceRegime.SINGLE,
        focus_relation="attended",
    )
    contract = AnswerContract(
        answer_space=AnswerSpace.SCALAR,
        truth_mode=TruthMode.DIRECT,
        time_axis=TimeAxis.EVENT,
        locality="point",
        evidence_regime=EvidenceRegime.SINGLE,
    )

    profile = _build_evidence_profile(
        [claim_anchored, claim_plain],  # type: ignore[arg-type]
        [],
        contract,
        frame,
    )
    assert profile.has_temporal_anchor is True, "Expected has_temporal_anchor=True"

    profile_no_anchor = _build_evidence_profile(
        [claim_plain],  # type: ignore[arg-type]
        [],
        contract,
        frame,
    )
    assert profile_no_anchor.has_temporal_anchor is False


def test_evidence_ids_raw_search_first():
    """_collect_evidence_episode_ids puts raw-search episodes first for topical relevance."""
    from datetime import UTC, datetime

    from ai_knot.query_runtime import _collect_evidence_episode_ids
    from ai_knot.query_types import AnswerItem, ClaimKind

    now = datetime(2026, 4, 15, tzinfo=UTC)

    items = [
        AnswerItem(
            value="v1",
            confidence=0.9,
            source_claim_ids=("c1",),
            source_episode_ids=("ep_answer_1",),
        ),
    ]
    claims = [
        _make_claim(
            "c2",
            ClaimKind.STATE,
            "Alice",
            "likes",
            "x",
            "ep_claim_only",  # only in claims, not in raw_search_ids
            now,
            {},
        )
    ]
    raw_search_ids = ["ep_raw_1", "ep_raw_2"]

    result = _collect_evidence_episode_ids(
        items,  # type: ignore[arg-type]
        claims,  # type: ignore[arg-type]
        episode_search_ids=raw_search_ids,
        cap=5,
    )
    # Raw-search episodes come first (topically relevant via entity+query text).
    assert result[0] == "ep_raw_1", f"Raw-search episode should be first, got {result}"
    assert "ep_claim_only" in result
    assert result.index("ep_raw_1") < result.index("ep_claim_only")


def test_expand_centers_first_orders_centers_before_neighbors():
    """Centers (entity-scoped) must come before prev/next neighbors (unfiltered)."""
    from types import SimpleNamespace

    from ai_knot.query_runtime import _expand_centers_first

    # Two ranked hits, each with prev+next neighbors. The neighbors would
    # previously (interleaved order) show up before the second center and
    # dominate a small render window.
    eps = [
        SimpleNamespace(id="center_1", prev_id="neighbor_1a", next_id="neighbor_1b"),
        SimpleNamespace(id="center_2", prev_id="neighbor_2a", next_id="neighbor_2b"),
    ]

    result = _expand_centers_first(eps, cap=6)
    assert result[:2] == ["center_1", "center_2"], f"Centers must lead: {result}"
    assert set(result) == {
        "center_1",
        "center_2",
        "neighbor_1a",
        "neighbor_1b",
        "neighbor_2a",
        "neighbor_2b",
    }


def test_expand_centers_first_respects_cap_and_dedups():
    """Cap is respected across centers + neighbors; duplicates are collapsed."""
    from types import SimpleNamespace

    from ai_knot.query_runtime import _expand_centers_first

    # center_2 is also listed as neighbor of center_1; must appear only once.
    eps = [
        SimpleNamespace(id="center_1", prev_id=None, next_id="center_2"),
        SimpleNamespace(id="center_2", prev_id=None, next_id="after"),
        SimpleNamespace(id="center_3", prev_id=None, next_id=None),
    ]

    # Cap at 2 — only two centers fit, neighbors are skipped entirely.
    result = _expand_centers_first(eps, cap=2)
    assert result == ["center_1", "center_2"]

    # Cap at 5 — all 3 centers + neighbor "after"; center_2 not duplicated.
    full = _expand_centers_first(eps, cap=5)
    assert full == ["center_1", "center_2", "center_3", "after"]
    assert len(full) == len(set(full))


def test_set_caps_widened_vs_scalar():
    """_caps_for_contract widens funnel for SET, leaves scalar unchanged."""
    from ai_knot.query_runtime import _PROFILE_CAPS, _caps_for_contract
    from ai_knot.query_types import (
        AnswerContract,
        AnswerSpace,
        EvidenceRegime,
        TimeAxis,
        TruthMode,
    )

    base = _PROFILE_CAPS["balanced"]

    scalar = AnswerContract(
        answer_space=AnswerSpace.SCALAR,
        truth_mode=TruthMode.DIRECT,
        time_axis=TimeAxis.NONE,
        locality="point",
        evidence_regime=EvidenceRegime.SINGLE,
    )
    set_ctx = AnswerContract(
        answer_space=AnswerSpace.SET,
        truth_mode=TruthMode.DIRECT,
        time_axis=TimeAxis.NONE,
        locality="point",
        evidence_regime=EvidenceRegime.AGGREGATE,
    )

    assert _caps_for_contract(base, scalar) is base, "Scalar contract must return base unchanged"

    widened = _caps_for_contract(base, set_ctx)
    assert widened.render_top_k > base.render_top_k, "SET must widen render_top_k"
    assert widened.char_budget > base.char_budget, "SET must widen char_budget"
    assert widened.collect_cap > base.collect_cap, "SET must widen collect_cap"
    assert widened.per_turn_max == base.per_turn_max, "per_turn_max must be unchanged"
