"""Integration regression tests for the temporal event_time anchor (add -> recall).

Locks the adopted production behaviour: ``add(content, event_time=anchor)`` resolves
relative-time expressions server-side, and ``recall()`` surfaces the COMPUTED absolute
date as ``(event date: X)``. This is the clean, general form used by production memory
systems (Mem0 ``timestamp``/``reference_date``, memvid ``ExplicitHeader``) — the
structured anchor, NOT a raw date prefixed into the indexed content (the banned
``dated`` ingest hack). ``_temporal.py`` covers the resolver in isolation; this file
covers the add->store->recall path and the no-leak contract.
"""

from __future__ import annotations

import pathlib
from datetime import datetime

import pytest

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage

# 8 May 2023 is a Monday — the conversation's date, used as the structured anchor.
ANCHOR = datetime(2023, 5, 8)


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    return KnowledgeBase(agent_id="anchor_test", storage=YAMLStorage(base_dir=str(tmp_path)))


def test_relative_time_resolved_and_rendered(kb: KnowledgeBase) -> None:
    """ "yesterday" + anchor 8 May 2023 -> recall renders "(event date: 7 May 2023)"."""
    kb.add("I went to the neighbourhood support group yesterday", event_time=ANCHOR)
    out = kb.recall("support group", top_k=5)
    assert "(event date: 7 May 2023)" in out


def test_anchor_not_leaked_into_content(kb: KnowledgeBase) -> None:
    """The raw anchor is NOT prefixed into indexed content (clean form != dated hack).

    The structured anchor is set on the fact and the RESOLVED date is captured as a
    qualifier (the qualifier is what persists and renders); the stored content stays the
    verbatim utterance — no ``[8 May 2023]`` text-prefix.

    Note: ``Fact.event_time`` is set in memory by ``_apply_temporal`` and is now
    persisted by all storage backends (mirroring ``valid_until``); a dedicated
    round-trip regression lives in ``test_event_time_persistence.py``. The resolved
    date is also captured as ``qualifiers["event_date"]`` for rendering.
    """
    fact = kb.add("I went to the support group yesterday", event_time=ANCHOR)
    # Structured anchor set in memory + resolved date captured as a (persisted) qualifier.
    assert fact.event_time == ANCHOR
    assert fact.qualifiers.get("event_date") == "7 May 2023"
    # The raw anchor is NOT prefixed into the indexed content.
    stored = kb.list_facts()
    assert stored and stored[0].content.startswith("I went to the support group")
    assert "8 May 2023" not in stored[0].content


def test_no_anchor_no_event_date(kb: KnowledgeBase) -> None:
    """No anchor -> no fabricated date (cannot resolve 'yesterday' without 'today')."""
    kb.add("I went to the support group yesterday")  # no event_time
    out = kb.recall("support group", top_k=5)
    assert "(event date:" not in out


def test_non_relative_content_no_event_date(kb: KnowledgeBase) -> None:
    """An anchor with no relative expression in the text yields no event_date render."""
    kb.add("I love hiking and painting", event_time=ANCHOR)
    out = kb.recall("hiking", top_k=5)
    assert "(event date:" not in out
