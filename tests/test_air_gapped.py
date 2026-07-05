"""Air-gap guarantee: the deterministic write + read path makes zero network calls.

This is the reproducible artifact for regulated / air-gapped deployments. It blocks
*all* outbound sockets, then exercises the full loop — add, structured update,
recall, shared-pool publish/recall, and the audit ledger — and asserts it completes
without a single connection attempt. If any code path tried to reach the network,
these tests would fail rather than silently phone home.

The only components in ai-knot that ever open an outbound connection are the LLM
providers (used only by the opt-in ``learn()`` extraction and the optional semantic
resolver), the dense embedder (opt-in via ``embed_url``), and the PostgreSQL driver
(your own database). None are on the deterministic add/recall path exercised here.
"""

from __future__ import annotations

import pathlib
import socket
from typing import Any

import pytest

from ai_knot.knowledge import KnowledgeBase, SharedMemoryPool
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact, MemoryOp


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any outbound socket connection raise — a hard, in-process air-gap.

    SQLite is local file I/O and opens no socket, so a blocked network still lets
    the deterministic path run; anything that tried to reach out would raise here.
    """

    def _blocked(*args: Any, **kwargs: Any) -> None:
        raise AssertionError(f"outbound network attempted: {args[:2]!r}")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)


@pytest.mark.usefixtures("no_network")
def test_write_and_recall_make_zero_network_calls(tmp_path: pathlib.Path) -> None:
    kb = KnowledgeBase("agent", storage=SQLiteStorage(str(tmp_path / "k.db")), embed_url="")
    kb.add("User deploys with Docker and Kubernetes")
    kb.add_resolved(
        [Fact(content="User works at Acme", entity="user", attribute="employer", value_text="Acme")]
    )
    kb.add_resolved(
        [
            Fact(
                content="User now works at Globex",
                entity="user",
                attribute="employer",
                value_text="Globex",
                op=MemoryOp.UPDATE,
            )
        ]
    )

    context = kb.recall("where does the user work and deploy?")
    assert "Docker" in context
    assert "Globex" in context


@pytest.mark.usefixtures("no_network")
def test_shared_pool_and_audit_make_zero_network_calls(tmp_path: pathlib.Path) -> None:
    storage = SQLiteStorage(str(tmp_path / "pool.db"))
    pool = SharedMemoryPool(storage=storage, persist_stats=True)
    pool.register("planner")
    pool.register("coder")
    kb = KnowledgeBase("planner", storage=storage, embed_url="")
    fact = Fact(
        content="Deploy target is GKE",
        entity="prod",
        attribute="deploy",
        slot_key="prod::deploy",
        value_text="gke",
        topic_channel="arch",
    )
    kb.replace_facts([fact])
    pool.publish("planner", [fact.id], kb=kb)
    pool.recall("deploy target?", "coder", top_k=3, topic_channel="arch")

    # The governance audit ledger (trust + fact-usage) is written entirely offline.
    assert storage.load_trust_events()
    assert storage.load_usage_events()
