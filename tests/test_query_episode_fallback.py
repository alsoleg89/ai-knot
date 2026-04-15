"""Regression tests for episode-level fallback when bundle plane is empty."""

from __future__ import annotations

from datetime import UTC, datetime

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.sqlite_storage import SQLiteStorage

NOW = datetime(2024, 1, 1, tzinfo=UTC)


def _kb(tmp_path: object) -> KnowledgeBase:
    return KnowledgeBase(
        agent_id="a",
        storage=SQLiteStorage(db_path=str(tmp_path / "t.db")),  # type: ignore[operator]
    )


def test_episode_fallback_when_bundle_plane_empty(tmp_path: object) -> None:
    """With materialize=False the bundle plane is empty; fallback must kick in."""
    kb = _kb(tmp_path)
    kb.ingest_episode(
        session_id="s",
        turn_id="t",
        speaker="Quentin",
        observed_at=NOW,
        raw_text="Quentin witnessed Novalux on 2024-06-15.",
        session_date=datetime(2024, 6, 15, tzinfo=UTC),
        materialize=False,
    )
    ans = kb.query("When did Quentin witness Novalux?")
    assert ans.trace.evidence_profile.episode_fallback_used is True
    assert "Quentin witnessed Novalux" in ans.evidence_text


def test_raw_search_runs_even_when_bundles_exist(tmp_path: object) -> None:
    """Raw-episode search now always enriches evidence_text when focus_entities are known.

    Previously the search was skipped when bundles produced answer_items, causing
    evidence_text to contain only short claim-fragment source episodes (wrong context
    for open-ended questions).  The regression: cat4 dropped from 84 % to 10 %.
    """
    kb = _kb(tmp_path)
    kb.ingest_episode(
        session_id="s",
        turn_id="t",
        speaker="Petra",
        observed_at=NOW,
        raw_text="Petra: I drive a Vortex.",
    )
    ans = kb.query("What does Petra drive?")
    # Raw-search fires whenever focus_entities are present; evidence_text must
    # contain the raw episode text regardless of whether answer_items are populated.
    assert "Vortex" in ans.evidence_text


def test_raw_search_enriches_evidence_when_answer_items_nonempty(tmp_path: object) -> None:
    """evidence_text must contain raw episode text even when operator produced answer_items.

    Regression: candidate_rank returned answer_items from weak atomic_claims; the
    old guard `not answer_items` prevented raw-search from running, so evidence_text
    was built from wrong (claim-source) episodes and the LLM lacked actual context.
    """
    kb = _kb(tmp_path)
    kb.ingest_episode(
        session_id="s",
        turn_id="t-noise",
        speaker="Melanie",
        observed_at=NOW,
        raw_text="Melanie: I like sunsets over water.",
    )
    kb.ingest_episode(
        session_id="s",
        turn_id="t-answer",
        speaker="Melanie",
        observed_at=NOW,
        session_date=datetime(2023, 5, 7, tzinfo=UTC),
        raw_text="Melanie: Yeah, I painted that lake sunrise last year! It's special to me.",
    )
    ans = kb.query("When did Melanie paint a sunrise?")
    # The raw painting episode must appear in evidence_text irrespective of
    # whether answer_items came back from atomic_claims materialization.
    assert "painted that lake sunrise last year" in ans.evidence_text


def test_episode_fallback_triggers_when_claims_exist_but_answer_is_empty(tmp_path: object) -> None:
    """Raw-episode fallback must still run when claims exist but answer_items are empty."""
    kb = _kb(tmp_path)
    kb.ingest_episode(
        session_id="s",
        turn_id="t-like",
        speaker="Melanie",
        observed_at=NOW,
        raw_text="Melanie: I like painting.",
    )
    kb.ingest_episode(
        session_id="s",
        turn_id="t-sunrise",
        speaker="Melanie",
        observed_at=NOW,
        session_date=datetime(2023, 5, 7, tzinfo=UTC),
        raw_text="Melanie: Yeah, I painted that lake sunrise last year! It's special to me.",
    )

    ans = kb.query("When did Melanie paint a sunrise?")

    assert ans.text == "No answer found."
    assert ans.trace.evidence_profile.episode_fallback_used is True
    assert "painted that lake sunrise last year" in ans.evidence_text
    assert "2023-05-07" in ans.evidence_text


def test_episode_search_does_not_drop_older_exact_match(tmp_path: object) -> None:
    """Older exact raw-episode matches must survive newer fuzzy entity matches."""
    storage = SQLiteStorage(db_path=str(tmp_path / "t.db"))  # type: ignore[operator]
    kb = KnowledgeBase(agent_id="a", storage=storage)

    for i in range(205):
        kb.ingest_episode(
            session_id="s",
            turn_id=f"recent-{i}",
            speaker="Melanie",
            observed_at=NOW,
            session_date=datetime(2024, 1, 1, tzinfo=UTC),
            raw_text=(
                "Melanie: Painting landscapes and still life is my favorite! "
                f"Here's painting number {i}."
            ),
            materialize=False,
        )

    kb.ingest_episode(
        session_id="s",
        turn_id="older-exact",
        speaker="Melanie",
        observed_at=NOW,
        session_date=datetime(2023, 5, 7, tzinfo=UTC),
        raw_text="Melanie: Yeah, I painted that lake sunrise last year! It's special to me.",
        materialize=False,
    )

    hits = storage.search_episodes_by_entities(
        "a",
        ["Melanie"],
        query="When did Melanie paint a sunrise?",
        top_k=5,
    )

    assert hits
    assert "painted that lake sunrise last year" in hits[0].raw_text
