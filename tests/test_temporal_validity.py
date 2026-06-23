"""Bi-temporal validity: event-time-anchored ``valid_from`` and supersession close.

A fact about a past event is *valid from* that event's time, not from when it was
ingested, and a superseded fact stays valid until its successor's *event* time.
Together these make point-in-time recall (``recall(now=...)``) correct for
historical replay — the engine returns the value that held *at the query time*,
not merely the latest value or nothing at all.

Crucially this does NOT change recall without a ``now`` (now = current time):
every historical fact still satisfies ``valid_from <= now`` and stays active, so
production / LoCoMo behaviour (which never passes ``now``) is untouched.  These
tests lock both the new point-in-time semantics and that invariance.
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime

from ai_knot.knowledge import KnowledgeBase
from ai_knot.learning import _supersede_close
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import ConversationTurn, Fact, MemoryType


def _d(year: int, month: int = 6, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def _kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(
        agent_id="conv-0",
        storage=SQLiteStorage(db_path=str(tmp_path / "kb.db")),
    )


# --------------------------------------------------------------------------- #
# add(): valid_from anchored to event_time
# --------------------------------------------------------------------------- #
def test_add_anchors_valid_from_to_event_time(tmp_path: pathlib.Path) -> None:
    kb = _kb(tmp_path)
    fact = kb.add("User adopted a dog", event_time=_d(2021))
    assert fact.event_time == _d(2021)
    assert fact.valid_from == _d(2021), "valid_from must follow event_time"


def test_add_without_event_time_keeps_ingest_valid_from(tmp_path: pathlib.Path) -> None:
    before = datetime.now(UTC)
    kb = _kb(tmp_path)
    fact = kb.add("User adopted a dog")  # no event_time
    assert fact.event_time is None
    assert fact.valid_from >= before, "default valid_from stays at ingest time"


# --------------------------------------------------------------------------- #
# Point-in-time recall: a fact is not yet active before its event time
# --------------------------------------------------------------------------- #
def test_recall_excludes_event_not_yet_happened(tmp_path: pathlib.Path) -> None:
    kb = _kb(tmp_path)
    kb.add("User moved to Tokyo", event_time=_d(2024))

    # Query as of 2022 — the move has not happened yet.
    assert "Tokyo" not in kb.recall("where does the user live", top_k=5, now=_d(2022))
    # Query as of 2025 — the move has happened.
    assert "Tokyo" in kb.recall("where does the user live", top_k=5, now=_d(2025))
    # Query without `now` (current time) — historical fact stays active.
    assert "Tokyo" in kb.recall("where does the user live", top_k=5)


# --------------------------------------------------------------------------- #
# Supersession close-time = successor's event time (bi-temporal KU)
# --------------------------------------------------------------------------- #
def _ingest_city_history(kb: KnowledgeBase) -> None:
    kb.add_resolved(
        [
            Fact(
                content="User lives in Paris",
                entity="user",
                attribute="city",
                value_text="Paris",
                type=MemoryType.SEMANTIC,
                event_time=_d(2020),
            )
        ]
    )
    kb.add_resolved(
        [
            Fact(
                content="User lives in Berlin",
                entity="user",
                attribute="city",
                value_text="Berlin",
                type=MemoryType.SEMANTIC,
                event_time=_d(2023),
            )
        ]
    )


def test_supersession_closes_at_successor_event_time(tmp_path: pathlib.Path) -> None:
    kb = _kb(tmp_path)
    _ingest_city_history(kb)
    facts = [f for f in kb.list_facts() if f.attribute == "city"]
    paris = next(f for f in facts if f.value_text == "Paris")
    berlin = next(f for f in facts if f.value_text == "Berlin")
    # Paris is valid from its event time and closes exactly when Berlin begins.
    assert paris.valid_from == _d(2020)
    assert paris.valid_until == _d(2023), "old value closes at successor's event time"
    assert berlin.valid_from == _d(2023)
    assert berlin.valid_until is None


def test_point_in_time_recall_returns_value_valid_then(tmp_path: pathlib.Path) -> None:
    kb = _kb(tmp_path)
    _ingest_city_history(kb)

    # 2021: Paris was the truth.
    out_2021 = kb.recall("where does the user live", top_k=5, now=_d(2021))
    assert "Paris" in out_2021 and "Berlin" not in out_2021

    # 2024: Berlin is the truth; Paris is closed.
    out_2024 = kb.recall("where does the user live", top_k=5, now=_d(2024))
    assert "Berlin" in out_2024 and "Paris" not in out_2024

    # No `now` (current time): latest value only — unchanged from prior behaviour.
    out_now = kb.recall("where does the user live", top_k=5)
    assert "Berlin" in out_now and "Paris" not in out_now


# --------------------------------------------------------------------------- #
# Out-of-order ingestion: never invert [valid_from, valid_until)
# --------------------------------------------------------------------------- #
def test_out_of_order_event_time_falls_back(tmp_path: pathlib.Path) -> None:
    kb = _kb(tmp_path)
    # First-ingested fact has the LATER event time...
    kb.add_resolved(
        [
            Fact(
                content="User lives in Rome",
                entity="user",
                attribute="city",
                value_text="Rome",
                event_time=_d(2025),
            )
        ]
    )
    # ...the second-ingested fact has an EARLIER event time (out of order).
    kb.add_resolved(
        [
            Fact(
                content="User lives in Oslo",
                entity="user",
                attribute="city",
                value_text="Oslo",
                event_time=_d(2019),
            )
        ]
    )
    rome = next(f for f in kb.list_facts() if f.value_text == "Rome")
    # Successor event time (2019) precedes Rome.valid_from (2025): using it would
    # invert the interval, so the close falls back to ingest-now (a real datetime,
    # not None, and not before valid_from).
    assert rome.valid_until is not None
    assert rome.valid_until >= rome.valid_from


# --------------------------------------------------------------------------- #
# Default (no event_time) supersession path is unchanged
# --------------------------------------------------------------------------- #
def test_supersession_without_event_time_unchanged(tmp_path: pathlib.Path) -> None:
    kb = _kb(tmp_path)
    kb.add_resolved(
        [Fact(content="User drives a Ford", entity="user", attribute="car", value_text="Ford")]
    )
    kb.add_resolved(
        [Fact(content="User drives a Tesla", entity="user", attribute="car", value_text="Tesla")]
    )
    cars = [f for f in kb.list_facts() if f.attribute == "car"]
    ford = next(f for f in cars if f.value_text == "Ford")
    tesla = next(f for f in cars if f.value_text == "Tesla")
    assert ford.valid_until is not None  # closed at ingest-now (fallback)
    assert tesla.valid_until is None
    assert "Tesla" in kb.recall("what car", top_k=5, now=datetime.now(UTC))


# --------------------------------------------------------------------------- #
# _supersede_close unit semantics
# --------------------------------------------------------------------------- #
def test_supersede_close_helper() -> None:
    fallback = _d(2030)
    superseded = Fact(content="old", valid_from=_d(2020))

    # Normal: successor's event time wins.
    succ = Fact(content="new", event_time=_d(2022))
    assert _supersede_close(succ, superseded, fallback) == _d(2022)

    # No successor event time → fallback.
    succ_none = Fact(content="new")
    assert _supersede_close(succ_none, superseded, fallback) == fallback

    # Successor event time precedes superseded.valid_from → fallback (no inversion).
    succ_early = Fact(content="new", event_time=_d(2019))
    assert _supersede_close(succ_early, superseded, fallback) == fallback


# --------------------------------------------------------------------------- #
# learn() threads event_time onto extracted facts (extraction mocked)
# --------------------------------------------------------------------------- #
def test_learn_threads_event_time(tmp_path: pathlib.Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    kb = _kb(tmp_path)
    canned = [
        Fact(
            content="User works at Globex",
            entity="user",
            attribute="employer",
            value_text="Globex",
            slot_key="user::employer",
            type=MemoryType.SEMANTIC,
        )
    ]
    # Bypass the LLM extraction stage with canned candidates.
    monkeypatch.setattr(kb, "_extract_phase", lambda *a, **k: list(canned))

    inserted = kb.learn(
        [ConversationTurn(role="user", content="I just joined Globex")],
        event_time=_d(2022),
    )
    assert inserted, "canned fact should be inserted"
    assert all(f.event_time == _d(2022) for f in inserted)
    assert all(f.valid_from == _d(2022) for f in inserted), "valid_from anchored in learn()"
