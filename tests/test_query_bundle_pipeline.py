"""Regression tests for query-pipeline fixes P1-P5.

Each test corresponds to a verified bug. Failure = regression reintroduced.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ai_knot.knowledge import KnowledgeBase
from ai_knot.query_contract import _extract_focus_entities, analyze_query
from ai_knot.query_types import BundleKind
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.support_retrieval import bundle_kinds_for_contract

NOW = datetime(2024, 1, 1, tzinfo=UTC)


def _kb(tmp_path: object) -> KnowledgeBase:
    return KnowledgeBase(
        agent_id="a",
        storage=SQLiteStorage(db_path=str(tmp_path / "t.db")),  # type: ignore[operator]
    )


# ---------------------------------------------------------------------------
# P1a — eager bundle build on ingest_episode
# ---------------------------------------------------------------------------


def test_p1a_ingest_materializes_bundles(tmp_path: object) -> None:
    """ingest_episode must build and persist bundles immediately."""
    kb = _kb(tmp_path)
    kb.ingest_episode(
        session_id="s",
        turn_id="t0",
        speaker="user",
        observed_at=NOW,
        # "Alice is a software engineer." matches _STATE_RE → produces 1 claim
        raw_text="Alice is a software engineer.",
    )
    bundles = kb._storage.load_bundles_by_topic("a", ["Alice"], None)
    assert bundles, "ingest_episode must build bundles eagerly"


# ---------------------------------------------------------------------------
# P1b — rebuild_materialized actually rebuilds (not just clears) bundles
# ---------------------------------------------------------------------------


def test_p1b_rebuild_materializes_bundles(tmp_path: object) -> None:
    """rebuild_materialized must report n_bundles_cleared > 0 after build."""
    kb = _kb(tmp_path)
    kb.ingest_episode(
        session_id="s",
        turn_id="t0",
        speaker="user",
        observed_at=NOW,
        # "Bob is a doctor." matches _STATE_RE → produces 1 claim
        raw_text="Bob is a doctor.",
    )
    report = kb.rebuild_materialized(force=True)
    assert report.n_bundles_cleared > 0, (
        "rebuild_materialized must rebuild bundles; currently clears and reports 0"
    )


# ---------------------------------------------------------------------------
# P2 — expand_claims uses in-memory member_claim_ids from synthetic bundles
# ---------------------------------------------------------------------------


def test_p2_evidence_profile_has_claims(tmp_path: object) -> None:
    """After ingest, query evidence profile must have n_support > 0."""
    kb = _kb(tmp_path)
    kb.ingest_episode(
        session_id="s",
        turn_id="t0",
        speaker="user",
        observed_at=NOW,
        # "Carol is a chess player." matches _STATE_RE → 1 claim, polarity="support"
        raw_text="Carol is a chess player.",
    )
    ans = kb.query("What does Carol play?", now=NOW)
    assert ans.trace is not None
    ep = ans.trace.evidence_profile
    # n_support counts claims with polarity="support" reaching the evidence layer.
    assert ep.n_support > 0, (
        "expand_claims must carry claim IDs to the evidence profile "
        "(was broken: synthetic fallback bundles had member_claim_ids ignored)"
    )


# ---------------------------------------------------------------------------
# P3 — single-token names and possessives extracted as focus entities
# ---------------------------------------------------------------------------


def test_p3_single_token_focus_entities() -> None:
    """Single-word proper names must be extracted as focus entities."""
    assert "Alice" in _extract_focus_entities("What does Alice do?")
    assert "Bob" in _extract_focus_entities("Bob's job is what?")


def test_p3_question_words_not_extracted() -> None:
    """Question-opener words (What, When, …) must not be returned as entities."""
    entities = _extract_focus_entities("What does Alice do?")
    assert "What" not in entities
    entities2 = _extract_focus_entities("When did Carol start?")
    assert "When" not in entities2


def test_p3_possessive_stripped() -> None:
    """'s possessive suffix must be stripped from extracted names."""
    entities = _extract_focus_entities("What is Alice's job?")
    assert "Alice" in entities
    assert "Alice's" not in entities


# ---------------------------------------------------------------------------
# P4 — temporal routing wins over SET; enum comparison uses `is`
# ---------------------------------------------------------------------------


def test_p4_temporal_routing_uses_event_neighborhood() -> None:
    """time_axis=EVENT must route to EVENT_NEIGHBORHOOD bundles."""
    from ai_knot.query_contract import derive_answer_contract

    frame = analyze_query("When did Alice start at Acme?")
    contract = derive_answer_contract(frame)
    kinds = bundle_kinds_for_contract(contract)
    assert kinds is not None
    assert BundleKind.EVENT_NEIGHBORHOOD in kinds, (
        "temporal query must prefer EVENT_NEIGHBORHOOD over SET bundles"
    )


def test_p4_set_routing_is_not_always_truthy() -> None:
    """SET routing must not fire for a non-set question (enum `is` comparison)."""
    from ai_knot.query_contract import derive_answer_contract
    from ai_knot.query_types import AnswerSpace

    # A scalar question — should NOT route to entity-topic-only SET bundles.
    frame = analyze_query("How many siblings does Alice have?")
    assert frame.answer_space is AnswerSpace.SCALAR
    contract = derive_answer_contract(frame)
    kinds = bundle_kinds_for_contract(contract)
    # Scalar → no forced SET routing; kinds must be None or not exclusively SET kinds.
    if kinds is not None:
        assert BundleKind.EVENT_NEIGHBORHOOD in kinds or kinds != [
            BundleKind.ENTITY_TOPIC,
            BundleKind.STATE_TIMELINE,
        ], "scalar question must not be routed as SET"


# ---------------------------------------------------------------------------
# P5 — MCP tool_query falls back on NotImplementedError
# ---------------------------------------------------------------------------


def test_p5_mcp_tool_falls_back_on_not_implemented(tmp_path: object, monkeypatch: object) -> None:
    """tool_query must treat NotImplementedError like RuntimeError and fall back."""
    from ai_knot import _mcp_tools

    kb = _kb(tmp_path)
    kb.add("legacy fact: Eve is a pilot.")

    def _boom(*a: object, **kw: object) -> object:
        raise NotImplementedError("Postgres v2 plane not yet implemented")

    monkeypatch.setattr(kb, "query", _boom)
    out = _mcp_tools.tool_query(kb, "Who is Eve?", top_k=5)
    assert isinstance(out, str)
    assert out, "fallback recall must return a non-empty string"
