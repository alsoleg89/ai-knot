"""Regression tests: knowledge-update supersession via the no-LLM ingest seam.

LongMemEval's knowledge-update (KU) ability requires that a later fact about the
same slot invalidates the earlier one (``valid_until`` set on the stale fact) so
that recall surfaces the *latest* value. The supersession engine lives in
``learning._resolve_phase`` but only ran inside ``learn()`` (LLM extraction).
``KnowledgeBase.add_resolved()`` exposes that engine for pre-structured facts
WITHOUT an LLM call, which is the clean seam the benchmark uses for KU ingest.

These tests prove:
  1. Same slot + new value -> old fact closed (``valid_until`` set), new inserted.
  2. ``recall(now=...)`` (the existing temporal filter) then returns only the
     latest value — i.e. KU demotion works end-to-end with no new recall code.
  3. Unslotted plain facts do NOT supersede (honest limitation: no slot to match).
"""

from __future__ import annotations

import pathlib
from datetime import UTC, datetime

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact, MemoryType


def _kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(
        agent_id="conv-0",
        storage=SQLiteStorage(db_path=str(tmp_path / "kb.db")),
    )


def test_add_resolved_supersedes_same_slot(tmp_path: pathlib.Path) -> None:
    """employer Acme -> Globex: the Acme fact is closed, Globex inserted v1."""
    kb = _kb(tmp_path)
    kb.add_resolved(
        [Fact(content="User works at Acme", entity="user", attribute="employer", value_text="Acme")]
    )
    kb.add_resolved(
        [
            Fact(
                content="User works at Globex",
                entity="user",
                attribute="employer",
                value_text="Globex",
            )
        ]
    )
    all_facts = kb.list_facts()
    employer_facts = [f for f in all_facts if f.attribute == "employer"]
    assert len(employer_facts) == 2  # old (closed) + new (active)
    acme = next(f for f in employer_facts if f.value_text == "Acme")
    globex = next(f for f in employer_facts if f.value_text == "Globex")
    assert acme.valid_until is not None, "stale fact must be temporally closed"
    assert globex.valid_until is None, "latest fact must stay active"
    assert globex.version == acme.version + 1


def test_recall_returns_latest_after_supersession(tmp_path: pathlib.Path) -> None:
    """The existing is_active recall filter surfaces only the latest value."""
    kb = _kb(tmp_path)
    kb.add_resolved(
        [Fact(content="User lives in Paris", entity="user", attribute="city", value_text="Paris")]
    )
    kb.add_resolved(
        [Fact(content="User lives in Berlin", entity="user", attribute="city", value_text="Berlin")]
    )
    # Query "now" — well after both ingests — must see only the active (latest) fact.
    out = kb.recall("where does the user live", top_k=5, now=datetime.now(UTC))
    assert "Berlin" in out
    assert "Paris" not in out


def test_unslotted_facts_do_not_supersede(tmp_path: pathlib.Path) -> None:
    """Honest limitation: without a slot there is nothing to supersede on."""
    kb = _kb(tmp_path)
    kb.add_resolved([Fact(content="The sky was grey on Monday")])
    kb.add_resolved([Fact(content="The sky was blue on Tuesday")])
    facts = kb.list_facts()
    # Both kept active — no slot_key/entity to detect they address the same thing.
    assert all(f.valid_until is None for f in facts)
    assert len(facts) >= 2


def test_add_resolved_derives_slot_key_from_entity_attribute(tmp_path: pathlib.Path) -> None:
    """slot_key is derived from entity+attribute when the caller omits it."""
    kb = _kb(tmp_path)
    inserted = kb.add_resolved(
        [Fact(content="User's role is CTO", entity="user", attribute="role", value_text="CTO")]
    )
    assert inserted[0].slot_key == "user::role"


def test_add_resolved_preserves_event_time(tmp_path: pathlib.Path) -> None:
    """The structured anchor survives the supersession pipeline + storage."""
    anchor = datetime(2023, 5, 8, tzinfo=UTC)
    kb = _kb(tmp_path)
    kb.add_resolved(
        [
            Fact(
                content="User joined Globex",
                entity="user",
                attribute="employer",
                value_text="Globex",
                type=MemoryType.SEMANTIC,
                event_time=anchor,
            )
        ]
    )
    stored = [f for f in kb.list_facts() if f.attribute == "employer"]
    assert stored and stored[0].event_time == anchor
