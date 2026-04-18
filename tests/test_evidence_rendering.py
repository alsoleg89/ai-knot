"""Tests for _render_evidence_context — session-grouped and flat evidence rendering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from ai_knot.query_runtime import _render_evidence_context, _render_evidence_flat


@dataclass
class FakeEp:
    id: str
    raw_text: str
    session_id: str
    session_date: datetime | None = None
    speaker: str = ""
    prev_id: str | None = None
    next_id: str | None = None


def _make_storage(eps: list[FakeEp]) -> object:
    """Return a fake storage object whose get_episode does (agent_id, eid) -> FakeEp | None."""
    ep_map = {ep.id: ep for ep in eps}

    class FakeStorage:
        def get_episode(self, _agent_id: str, eid: str) -> FakeEp | None:
            return ep_map.get(eid)

    return FakeStorage()


def _dt(date_str: str) -> datetime:
    year, month, day = date_str.split("-")
    return datetime(int(year), int(month), int(day), tzinfo=UTC)


class TestSessionGrouping:
    """test_session_grouping: two sessions appear in chronological order with headers."""

    def test_two_sessions_produce_two_headers(self) -> None:
        eps = [
            FakeEp(
                id="ep-a1",
                raw_text="Hello from session A",
                session_id="sess-A",
                session_date=_dt("2024-01-10"),
                speaker="Alice",
            ),
            FakeEp(
                id="ep-b1",
                raw_text="Hello from session B",
                session_id="sess-B",
                session_date=_dt("2024-03-05"),
                speaker="Bob",
            ),
        ]
        storage = _make_storage(eps)
        result = _render_evidence_context(storage, "agent1", ["ep-a1", "ep-b1"])

        assert "## Session 2024-01-10" in result
        assert "## Session 2024-03-05" in result

    def test_sessions_appear_in_retrieval_relevance_order(self) -> None:
        # ep-b1 (later date) appears first in episode_ids — should be first in output
        eps = [
            FakeEp(
                id="ep-b1",
                raw_text="Later session turn",
                session_id="sess-B",
                session_date=_dt("2024-06-01"),
            ),
            FakeEp(
                id="ep-a1",
                raw_text="Earlier session turn",
                session_id="sess-A",
                session_date=_dt("2024-01-15"),
            ),
        ]
        storage = _make_storage(eps)
        result = _render_evidence_context(storage, "agent1", ["ep-b1", "ep-a1"])

        idx_a = result.index("## Session 2024-01-15")
        idx_b = result.index("## Session 2024-06-01")
        assert idx_b < idx_a, "First-retrieved session must appear first (relevance order)"

    def test_episode_text_under_correct_session(self) -> None:
        eps = [
            FakeEp(
                id="ep-a1",
                raw_text="Alice says something",
                session_id="sess-A",
                session_date=_dt("2024-01-10"),
                speaker="Alice",
            ),
            FakeEp(
                id="ep-b1",
                raw_text="Bob says something else",
                session_id="sess-B",
                session_date=_dt("2024-03-05"),
                speaker="Bob",
            ),
        ]
        storage = _make_storage(eps)
        result = _render_evidence_context(storage, "agent1", ["ep-a1", "ep-b1"])

        lines = result.splitlines()
        # Find header positions
        idx_a = next(i for i, ln in enumerate(lines) if "2024-01-10" in ln and ln.startswith("##"))
        idx_b = next(i for i, ln in enumerate(lines) if "2024-03-05" in ln and ln.startswith("##"))

        # Alice's text must appear after header A and before header B
        alice_line = next(i for i, ln in enumerate(lines) if "Alice says something" in ln)
        bob_line = next(i for i, ln in enumerate(lines) if "Bob says something else" in ln)

        assert idx_a < alice_line < idx_b, "Alice's text must be under session A header"
        assert bob_line > idx_b, "Bob's text must be under session B header"


class TestSingleSessionHeader:
    """test_single_session_no_extra_header: single session still renders with a header."""

    def test_single_session_has_header(self) -> None:
        eps = [
            FakeEp(
                id="ep-x1",
                raw_text="Only session turn",
                session_id="sess-X",
                session_date=_dt("2024-05-20"),
                speaker="Charlie",
            ),
        ]
        storage = _make_storage(eps)
        result = _render_evidence_context(storage, "agent1", ["ep-x1"])

        assert "## Session 2024-05-20" in result
        assert "Only session turn" in result

    def test_single_session_exactly_one_header(self) -> None:
        eps = [
            FakeEp(
                id="ep-x1",
                raw_text="Turn one",
                session_id="sess-X",
                session_date=_dt("2024-05-20"),
            ),
            FakeEp(
                id="ep-x2",
                raw_text="Turn two",
                session_id="sess-X",
                session_date=_dt("2024-05-20"),
            ),
        ]
        storage = _make_storage(eps)
        result = _render_evidence_context(storage, "agent1", ["ep-x1", "ep-x2"])

        header_count = result.count("## Session")
        assert header_count == 1, f"Expected 1 header, got {header_count}"


class TestEdgeCases:
    def test_empty_episode_ids_returns_empty_string(self) -> None:
        storage = _make_storage([])
        result = _render_evidence_context(storage, "agent1", [])
        assert result == ""

    def test_missing_episode_skipped(self) -> None:
        eps = [
            FakeEp(
                id="ep-real",
                raw_text="Real turn",
                session_id="sess-A",
                session_date=_dt("2024-02-01"),
            ),
        ]
        storage = _make_storage(eps)
        # "ep-ghost" does not exist in storage
        result = _render_evidence_context(storage, "agent1", ["ep-ghost", "ep-real"])
        assert "Real turn" in result
        assert "## Session 2024-02-01" in result

    def test_date_prefix_on_each_turn(self) -> None:
        eps = [
            FakeEp(
                id="ep-d1",
                raw_text="Dated turn text",
                session_id="sess-D",
                session_date=_dt("2024-07-04"),
                speaker="Dave",
            ),
        ]
        storage = _make_storage(eps)
        result = _render_evidence_context(storage, "agent1", ["ep-d1"])

        # Each turn line should carry a [YYYY-MM-DD] prefix
        assert "[2024-07-04]" in result

    def test_speaker_prefix_added_when_missing(self) -> None:
        eps = [
            FakeEp(
                id="ep-s1",
                raw_text="plain text without speaker",
                session_id="sess-S",
                session_date=_dt("2024-08-01"),
                speaker="Eve",
            ),
        ]
        storage = _make_storage(eps)
        result = _render_evidence_context(storage, "agent1", ["ep-s1"])
        assert "Eve: plain text without speaker" in result

    def test_speaker_prefix_not_doubled(self) -> None:
        eps = [
            FakeEp(
                id="ep-s2",
                raw_text="Frank: already has prefix",
                session_id="sess-S",
                session_date=_dt("2024-08-02"),
                speaker="Frank",
            ),
        ]
        storage = _make_storage(eps)
        result = _render_evidence_context(storage, "agent1", ["ep-s2"])
        # Should NOT produce "Frank: Frank: already has prefix"
        assert "Frank: Frank:" not in result
        assert "Frank: already has prefix" in result

    def test_no_duplicate_episodes_via_window(self) -> None:
        """Episodes referenced as prev/next of another must not appear twice."""
        eps = [
            FakeEp(
                id="ep-1",
                raw_text="Turn 1",
                session_id="sess-A",
                session_date=_dt("2024-03-01"),
                next_id="ep-2",
            ),
            FakeEp(
                id="ep-2",
                raw_text="Turn 2",
                session_id="sess-A",
                session_date=_dt("2024-03-01"),
                prev_id="ep-1",
            ),
        ]
        storage = _make_storage(eps)
        result = _render_evidence_context(storage, "agent1", ["ep-1", "ep-2"])
        # "Turn 2" should appear exactly once
        assert result.count("Turn 2") == 1

    def test_session_without_date_uses_id_prefix(self) -> None:
        eps = [
            FakeEp(
                id="some-session-id-xyz",
                raw_text="No date episode",
                session_id="some-session-id-xyz",
                session_date=None,
            ),
        ]
        storage = _make_storage(eps)
        result = _render_evidence_context(storage, "agent1", ["some-session-id-xyz"])
        assert "## Session some-session-" in result


class TestFlatRendering:
    def test_flat_mode_has_no_session_headers(self) -> None:
        eps = [
            FakeEp(
                id="ep-a1",
                raw_text="Turn A",
                session_id="sess-A",
                session_date=_dt("2024-01-10"),
            ),
            FakeEp(
                id="ep-b1",
                raw_text="Turn B",
                session_id="sess-B",
                session_date=_dt("2024-03-05"),
            ),
        ]
        storage = _make_storage(eps)
        result = _render_evidence_flat(storage, "agent1", ["ep-a1", "ep-b1"])
        assert "## Session" not in result
        assert "Turn A" in result
        assert "Turn B" in result

    def test_flat_mode_preserves_input_order(self) -> None:
        eps = [
            FakeEp(
                id="ep-later",
                raw_text="Later episode",
                session_id="sess-B",
                session_date=_dt("2024-06-01"),
            ),
            FakeEp(
                id="ep-earlier",
                raw_text="Earlier episode",
                session_id="sess-A",
                session_date=_dt("2024-01-15"),
            ),
        ]
        storage = _make_storage(eps)
        result = _render_evidence_flat(storage, "agent1", ["ep-later", "ep-earlier"])
        idx_later = result.index("Later episode")
        idx_earlier = result.index("Earlier episode")
        assert idx_later < idx_earlier, "Input (relevance) order must be preserved"

    def test_flat_mode_inline_date(self) -> None:
        eps = [
            FakeEp(
                id="ep-1",
                raw_text="Hello world",
                session_id="sess-A",
                session_date=_dt("2024-05-27"),
                speaker="Jon",
            ),
        ]
        storage = _make_storage(eps)
        result = _render_evidence_flat(storage, "agent1", ["ep-1"])
        assert "[2024-05-27]" in result
        assert "Hello world" in result

    def test_render_evidence_context_flat_kwarg(self) -> None:
        eps = [
            FakeEp(
                id="ep-1",
                raw_text="Event text",
                session_id="sess-A",
                session_date=_dt("2024-05-01"),
            ),
        ]
        storage = _make_storage(eps)
        flat_result = _render_evidence_context(storage, "agent1", ["ep-1"], flat=True)
        grouped_result = _render_evidence_context(storage, "agent1", ["ep-1"])
        assert "## Session" not in flat_result
        assert "## Session" in grouped_result


class TestChronologicalOrdering:
    def test_chronological_kwarg_puts_earlier_session_first(self) -> None:
        # ep-b1 (June, later) appears first in retrieval — but with chronological=True
        # the earlier session (January) should come first in output.
        eps = [
            FakeEp(
                id="ep-b1",
                raw_text="Later session turn",
                session_id="sess-B",
                session_date=_dt("2024-06-01"),
            ),
            FakeEp(
                id="ep-a1",
                raw_text="Earlier session turn",
                session_id="sess-A",
                session_date=_dt("2024-01-15"),
            ),
        ]
        storage = _make_storage(eps)
        result = _render_evidence_context(
            storage, "agent1", ["ep-b1", "ep-a1"], chronological=True
        )
        idx_a = result.index("## Session 2024-01-15")
        idx_b = result.index("## Session 2024-06-01")
        assert idx_a < idx_b, "Chronological mode must put earlier session first"

    def test_default_mode_preserves_relevance_order(self) -> None:
        # ep-b1 (later date) appears first in retrieval; default mode keeps that.
        eps = [
            FakeEp(
                id="ep-b1",
                raw_text="Later session turn",
                session_id="sess-B",
                session_date=_dt("2024-06-01"),
            ),
            FakeEp(
                id="ep-a1",
                raw_text="Earlier session turn",
                session_id="sess-A",
                session_date=_dt("2024-01-15"),
            ),
        ]
        storage = _make_storage(eps)
        result = _render_evidence_context(storage, "agent1", ["ep-b1", "ep-a1"])
        idx_a = result.index("## Session 2024-01-15")
        idx_b = result.index("## Session 2024-06-01")
        assert idx_b < idx_a, "Default (relevance) mode must put first-retrieved session first"
