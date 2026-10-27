from __future__ import annotations

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.sqlite_storage import SQLiteStorage


def test_recall_cannot_restore_stale_content(tmp_path, monkeypatch):
    path = str(tmp_path / "facts.db")
    store = SQLiteStorage(path)
    kb = KnowledgeBase(agent_id="owner", storage=store)
    kb.add("Deployment uses blue service")
    original = getattr(kb, "_record_access", None)
    assert original is not None, "recall must use guarded metadata persistence"

    def interleave(ids, access_time):
        peer = SQLiteStorage(path)
        current = peer.load("owner")
        current[0].content = "Deployment uses green service"
        peer.save("owner", current)
        original(ids, access_time)

    monkeypatch.setattr(kb, "_record_access", interleave)
    kb.recall("Deployment")
    assert SQLiteStorage(path).load("owner")[0].content == "Deployment uses green service"
