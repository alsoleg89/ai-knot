"""Regression tests for _window_text construction in search_episodes_by_entities.

Bug (pre-fix): window was built as ``center + prev + center + next``, duplicating
the center text.  This inflated center token TF in the window and corrupted
avg_doc_len — both affect BM25 scores.

Fix: window = ``prev + center + next`` (each turn appears exactly once).

Key facts about how the function works:
- The LIKE '%entity%' filter determines CANDIDATES (only raw_text matches are returned).
- _window_text is used for BM25 SCORING of those candidates.
- Final score = max(bm25(window_tokens), bm25(center_tokens)).
  Centre benefits from short-doc length-norm; window provides cross-turn IDF.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ai_knot.query_types import RawEpisode
from ai_knot.storage.sqlite_storage import SQLiteStorage


def _ep(ep_id: str, session_id: str, turn_i: int, text: str, agent_id: str = "a") -> RawEpisode:
    return RawEpisode(
        id=ep_id,
        agent_id=agent_id,
        session_id=session_id,
        turn_id=f"{session_id}-{turn_i}",
        speaker="user",
        observed_at=datetime(2024, 1, 1, turn_i + 1, tzinfo=UTC),
        session_date=None,
        raw_text=text,
        source_meta={},
        parent_episode_id=None,
    )


def test_cross_turn_context_boosts_center(tmp_path: object) -> None:
    """Cross-turn window context allows center to rank above a standalone match.

    When two episodes both match the entity LIKE filter, the one whose window
    contains additional query tokens from neighbouring turns must score higher.
    This is the intended purpose of the window text: cross-turn BM25 scoring.

    With double-center the centre's tokens appeared twice in the window, inflating
    its TF and corrupting avg_doc_len.  After the fix, each turn appears once,
    so the IDF and avg_doc_len computations are correct.
    """
    storage = SQLiteStorage(str(tmp_path) + "/test.db", embed_url="")  # type: ignore[operator]

    # Session A: prev has APPOINTMENT, center matches DOCTOR entity (also in ep2).
    # Center's window (prev+center+next) gives it access to APPOINTMENT context.
    storage.save_episodes(
        "a",
        [
            _ep("a0", "s", 0, "APPOINTMENT scheduled today"),  # prev
            _ep("a1", "s", 1, "DOCTOR visit confirmed"),  # center — matches DOCTOR
            _ep("a2", "s", 2, "everything went fine"),  # next
        ],
    )
    # ep2: standalone DOCTOR episode — same entity but no window context about APPOINTMENT.
    storage.save_episodes(
        "a",
        [_ep("b0", "s2", 0, "DOCTOR work stuff random noise")],
    )

    hits = storage.search_episodes_by_entities("a", ["DOCTOR"], query="DOCTOR APPOINTMENT", top_k=3)

    assert len(hits) >= 2, f"expected a1 and b0 in results, got {[h.id for h in hits]}"
    ids = [h.id for h in hits]
    # a1's window contains APPOINTMENT (from prev), so it must rank above b0 for
    # the query "DOCTOR APPOINTMENT".
    assert ids[0] == "a1", (
        f"expected a1 (window contains APPOINTMENT) to rank 1st, got {ids[0]}; full ranking: {ids}"
    )


def test_standalone_not_disadvantaged_vs_window_center(tmp_path: object) -> None:
    """A standalone episode must not be outscored by an identical episode that has
    noisy window context.

    With double-center: the center's own tokens appeared in the window TWICE,
    so bm25(window) could exceed bm25(center) for center-specific terms, giving
    the windowed episode an unfair advantage over an identical standalone.

    After the fix, window = prev + center + next; if prev and next contain only
    noise tokens, bm25(window) ≤ bm25(center) for terms that only appear in center.
    The identical standalone must rank at least as high (tiebreak goes to recency).
    """
    storage = SQLiteStorage(str(tmp_path) + "/test.db", embed_url="")  # type: ignore[operator]

    # ep_center: in a 3-episode session; prev/next have unrelated noise.
    storage.save_episodes(
        "a",
        [
            _ep("a0", "s", 0, "NOISE one two three"),
            _ep("a1", "s", 1, "QUERYTERM signal data"),  # center
            _ep("a2", "s", 2, "NOISE four five six"),
        ],
    )
    # ep_standalone: same raw_text as center, but no window context.
    # Give it a LATER timestamp so recency tiebreak does NOT hide BM25 effects.
    storage.save_episodes(
        "a",
        [
            RawEpisode(
                id="b0",
                agent_id="a",
                session_id="s2",
                turn_id="s2-0",
                speaker="user",
                observed_at=datetime(2024, 6, 1, tzinfo=UTC),  # later than a1
                session_date=None,
                raw_text="QUERYTERM signal data",
                source_meta={},
                parent_episode_id=None,
            )
        ],
    )

    hits = storage.search_episodes_by_entities(
        "a", ["QUERYTERM"], query="QUERYTERM signal", top_k=3
    )

    ids = [h.id for h in hits]
    assert "b0" in ids, "standalone b0 must be in results"
    assert "a1" in ids, "window-center a1 must be in results"

    b0_rank = ids.index("b0")
    a1_rank = ids.index("a1")
    # b0 is more recent (Jun 2024 > Jan 2024) AND not disadvantaged by noisy window.
    # With double-center, a1's window = "QUERYTERM signal data NOISE one two three
    #   QUERYTERM signal data NOISE four five six" — QUERYTERM appears 2× → window
    #   score inflated → a1 could beat b0 despite b0 being more recent and identical.
    # After fix: a1's window score ≤ a1's center score; b0 is more recent so b0 wins.
    assert b0_rank <= a1_rank, (
        f"b0 (rank {b0_rank}, more recent) must rank <= a1 (rank {a1_rank}); "
        "double-center would inflate a1's window score and unfairly beat b0"
    )
