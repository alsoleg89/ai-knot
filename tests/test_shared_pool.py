"""Tests for SharedMemoryPool: publish, recall, sync_dirty, sync_slot_deltas."""

from __future__ import annotations

import pathlib

import pytest

from ai_knot.knowledge import KnowledgeBase, SharedMemoryPool
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import Fact, MESIState, SlotDelta

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sqlite_db(tmp_path: pathlib.Path) -> SQLiteStorage:
    return SQLiteStorage(str(tmp_path / "test.db"))


@pytest.fixture
def pool_sqlite(sqlite_db: SQLiteStorage) -> SharedMemoryPool:
    pool = SharedMemoryPool(storage=sqlite_db)
    pool.register("agent_a")
    pool.register("agent_b")
    return pool


@pytest.fixture
def pool_yaml(tmp_path: pathlib.Path) -> SharedMemoryPool:
    pool = SharedMemoryPool(storage=YAMLStorage(base_dir=str(tmp_path)))
    pool.register("agent_a")
    pool.register("agent_b")
    return pool


def _kb(agent_id: str, storage: SQLiteStorage) -> KnowledgeBase:
    return KnowledgeBase(agent_id, storage=storage)


# ---------------------------------------------------------------------------
# SlotDelta dataclass
# ---------------------------------------------------------------------------


class TestSlotDelta:
    def test_fields(self) -> None:
        d = SlotDelta(
            slot_key="Alex::salary",
            version=2,
            op="supersede",
            fact_id="abc123",
            content="Alex earns 95k",
            prompt_surface="",
        )
        assert d.slot_key == "Alex::salary"
        assert d.version == 2
        assert d.op == "supersede"
        assert d.fact_id == "abc123"
        assert d.content == "Alex earns 95k"
        assert d.prompt_surface == ""

    def test_default_prompt_surface(self) -> None:
        d = SlotDelta(slot_key="", version=1, op="new", fact_id="x", content="y")
        assert d.prompt_surface == ""


# ---------------------------------------------------------------------------
# publish() — slot-key CAS
# ---------------------------------------------------------------------------


class TestPublish:
    def test_publish_new_fact(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add("Alex earns 95k")
        published = pool_sqlite.publish("agent_a", [fact.id], kb=kb)
        assert len(published) == 1
        assert published[0].mesi_state == MESIState.SHARED
        assert published[0].origin_agent_id == "agent_a"

    def test_publish_unregistered_agent_raises(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb = _kb("unknown", sqlite_db)
        fact = kb.add("test")
        with pytest.raises(ValueError, match="not registered"):
            pool_sqlite.publish("unknown", [fact.id], kb=kb)

    def test_publish_same_id_twice_is_noop(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add("Alex earns 95k")
        pool_sqlite.publish("agent_a", [fact.id], kb=kb)
        second = pool_sqlite.publish("agent_a", [fact.id], kb=kb)
        assert second == []  # no-op

    def test_publish_supersedes_by_slot_key(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        """Publishing a new fact with the same slot_key closes the old one."""
        kb = _kb("agent_a", sqlite_db)
        old = kb.add("Alex earns 80k")
        old.slot_key = "Alex::salary"
        old.value_text = "80000"
        kb.replace_facts([old])

        pool_sqlite.publish("agent_a", [old.id], kb=kb)

        new = kb.add("Alex earns 95k")
        # Patch slot fields on the already-stored fact.
        facts = kb.list_facts()
        for f in facts:
            if f.id == new.id:
                f.slot_key = "Alex::salary"
                f.value_text = "95000"
                break
        kb.replace_facts(facts)

        published = pool_sqlite.publish("agent_a", [new.id], kb=kb)
        assert len(published) == 1
        assert published[0].mesi_state == MESIState.MODIFIED

        # Shared pool should have only one active fact for Alex::salary.
        all_shared = pool_sqlite.list_shared_facts()
        active = [f for f in all_shared if f.is_active() and f.slot_key == "Alex::salary"]
        assert len(active) == 1
        assert active[0].value_text == "95000"


# ---------------------------------------------------------------------------
# recall()
# ---------------------------------------------------------------------------


class TestPoolRecall:
    def test_recall_returns_relevant_facts(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add("Alex earns 95k annually")
        pool_sqlite.publish("agent_a", [fact.id], kb=kb)

        results = pool_sqlite.recall("Alex salary", "agent_b", top_k=3)
        assert len(results) >= 1
        assert any("Alex" in f.content for f, _ in results)

    def test_recall_empty_pool(self, pool_sqlite: SharedMemoryPool) -> None:
        results = pool_sqlite.recall("anything", "agent_a", top_k=5)
        assert results == []

    def test_foreign_facts_discounted(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb_a = _kb("agent_a", sqlite_db)
        fact = kb_a.add("User prefers Python for ML work")
        pool_sqlite.publish("agent_a", [fact.id], kb=kb_a)

        # Trust defaults to 0.8 for foreign agents.
        results = pool_sqlite.recall("Python ML", "agent_b", top_k=3)
        # Score should be discounted (< unmodified BM25 score)
        assert len(results) >= 1
        _, score = results[0]
        assert score < 1.0  # discounted from full score


# ---------------------------------------------------------------------------
# sync_dirty() — backward compat
# ---------------------------------------------------------------------------


class TestSyncDirty:
    def test_first_sync_gets_all(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb_a = _kb("agent_a", sqlite_db)
        for text in ["fact one", "fact two", "fact three"]:
            kb_a.add(text)
        all_ids = [f.id for f in kb_a.list_facts()]
        pool_sqlite.publish("agent_a", all_ids, kb=kb_a)

        dirty = pool_sqlite.sync_dirty("agent_b")
        assert len(dirty) == 3

    def test_second_sync_only_new_facts(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb_a = _kb("agent_a", sqlite_db)
        f1 = kb_a.add("first fact")
        pool_sqlite.publish("agent_a", [f1.id], kb=kb_a)

        pool_sqlite.sync_dirty("agent_b")  # Pull first batch.

        f2 = kb_a.add("second fact")
        pool_sqlite.publish("agent_a", [f2.id], kb=kb_a)

        dirty2 = pool_sqlite.sync_dirty("agent_b")
        assert len(dirty2) == 1
        assert dirty2[0].content == "second fact"

    def test_agent_does_not_see_own_facts(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb_a = _kb("agent_a", sqlite_db)
        fact = kb_a.add("agent A's own fact")
        pool_sqlite.publish("agent_a", [fact.id], kb=kb_a)

        dirty = pool_sqlite.sync_dirty("agent_a")
        assert dirty == []


# ---------------------------------------------------------------------------
# sync_slot_deltas() — new lightweight sync
# ---------------------------------------------------------------------------


class TestSyncSlotDeltas:
    def test_returns_slot_deltas(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb_a = _kb("agent_a", sqlite_db)
        fact = kb_a.add("Alex earns 95k")
        pool_sqlite.publish("agent_a", [fact.id], kb=kb_a)

        deltas = pool_sqlite.sync_slot_deltas("agent_b")
        assert len(deltas) == 1
        assert isinstance(deltas[0], SlotDelta)
        assert deltas[0].content == "Alex earns 95k"

    def test_delta_op_new_for_first_publish(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb_a = _kb("agent_a", sqlite_db)
        fact = kb_a.add("First publish")
        pool_sqlite.publish("agent_a", [fact.id], kb=kb_a)

        deltas = pool_sqlite.sync_slot_deltas("agent_b")
        assert deltas[0].op == "new"

    def test_delta_op_supersede_on_update(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb_a = _kb("agent_a", sqlite_db)
        old = kb_a.add("Alex earns 80k")
        old.slot_key = "Alex::salary"
        old.value_text = "80000"
        kb_a.replace_facts([old])
        pool_sqlite.publish("agent_a", [old.id], kb=kb_a)

        # Pull first delta.
        pool_sqlite.sync_slot_deltas("agent_b")

        # Now supersede: add new fact, patch slot fields.
        new = kb_a.add("Alex earns 95k")
        facts = kb_a.list_facts()
        for f in facts:
            if f.id == new.id:
                f.slot_key = "Alex::salary"
                f.value_text = "95000"
                break
        kb_a.replace_facts(facts)
        pool_sqlite.publish("agent_a", [new.id], kb=kb_a)

        deltas2 = pool_sqlite.sync_slot_deltas("agent_b")
        # Should have one supersede delta (the new fact) and possibly one invalidate.
        ops = {d.op for d in deltas2}
        assert "supersede" in ops

    def test_second_sync_only_new_deltas(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb_a = _kb("agent_a", sqlite_db)
        f1 = kb_a.add("fact A")
        pool_sqlite.publish("agent_a", [f1.id], kb=kb_a)

        pool_sqlite.sync_slot_deltas("agent_b")  # First pull.

        f2 = kb_a.add("fact B")
        pool_sqlite.publish("agent_a", [f2.id], kb=kb_a)

        deltas2 = pool_sqlite.sync_slot_deltas("agent_b")
        assert len(deltas2) >= 1
        assert any(d.content == "fact B" for d in deltas2)

    def test_agent_does_not_see_own_deltas(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb_a = _kb("agent_a", sqlite_db)
        fact = kb_a.add("agent A's own fact")
        pool_sqlite.publish("agent_a", [fact.id], kb=kb_a)

        deltas = pool_sqlite.sync_slot_deltas("agent_a")
        assert deltas == []

    def test_yaml_backend_fallback(self, pool_yaml: SharedMemoryPool) -> None:
        """sync_slot_deltas() works with YAML backend (Python fallback path)."""
        storage = pool_yaml._storage
        assert isinstance(storage, YAMLStorage)

        kb_a = KnowledgeBase("agent_a", storage=storage)
        fact = kb_a.add("test fact for YAML")
        pool_yaml.publish("agent_a", [fact.id], kb=kb_a)

        deltas = pool_yaml.sync_slot_deltas("agent_b")
        assert len(deltas) >= 1
        assert isinstance(deltas[0], SlotDelta)


# ---------------------------------------------------------------------------
# load_active_frontier() — SQLite
# ---------------------------------------------------------------------------


class TestLoadActiveFrontier:
    def test_returns_latest_per_slot(self, sqlite_db: SQLiteStorage) -> None:
        """Active frontier returns only the latest version per slot_key."""
        from datetime import UTC, datetime

        now = datetime(2024, 1, 1, tzinfo=UTC)
        old = Fact(content="Alex earns 80k", slot_key="Alex::salary", value_text="80000")
        old.valid_until = now  # closed
        new = Fact(content="Alex earns 95k", slot_key="Alex::salary", value_text="95000")
        new.version = 1
        sqlite_db.save("test_agent", [old, new])

        frontier = sqlite_db.load_active_frontier("test_agent")
        assert len(frontier) == 1
        assert frontier[0].value_text == "95000"

    def test_unslotted_facts_all_included(self, sqlite_db: SQLiteStorage) -> None:
        """Unslotted active facts are all returned (no collapsing by slot)."""
        f1 = Fact(content="User prefers Python")  # no slot_key
        f2 = Fact(content="User uses Docker")  # no slot_key
        sqlite_db.save("test_agent", [f1, f2])

        frontier = sqlite_db.load_active_frontier("test_agent")
        assert len(frontier) == 2


# ---------------------------------------------------------------------------
# Publish gating — utility_threshold
# ---------------------------------------------------------------------------


class TestPublishGating:
    def test_low_utility_fact_not_published(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        """Facts below utility_threshold are silently skipped."""
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add("low-importance noise")
        fact.importance = 0.1
        fact.state_confidence = 0.1
        kb.replace_facts([fact])  # Persist mutated values to storage.

        published = pool_sqlite.publish("agent_a", [fact.id], kb=kb, utility_threshold=0.5)
        assert published == []
        assert pool_sqlite.list_shared_facts() == []

    def test_high_utility_fact_published(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        """Facts at or above utility_threshold are published normally."""
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add("high-importance finding")
        fact.importance = 0.9
        fact.state_confidence = 0.9
        kb.replace_facts([fact])  # Persist mutated values to storage.

        published = pool_sqlite.publish("agent_a", [fact.id], kb=kb, utility_threshold=0.5)
        assert len(published) == 1

    def test_default_threshold_allows_all(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        """Default threshold (0.0) lets every fact through."""
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add("any fact")
        fact.importance = 0.01
        fact.state_confidence = 0.0
        kb.replace_facts([fact])  # Persist mutated values to storage.

        published = pool_sqlite.publish("agent_a", [fact.id], kb=kb)
        assert len(published) == 1


# ---------------------------------------------------------------------------
# Topic channels + visibility_scope
# ---------------------------------------------------------------------------


class TestTopicChannels:
    def test_channel_filter_isolates_results(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        """recall() with topic_channel only returns facts from that channel."""
        kb = _kb("agent_a", sqlite_db)
        f_devops = kb.add("deploy pipeline uses Kubernetes")
        f_devops.topic_channel = "devops"
        f_finance = kb.add("Q1 revenue is 5M")
        f_finance.topic_channel = "finance"
        kb.replace_facts([f_devops, f_finance])  # Persist channel tags.

        pool_sqlite.publish("agent_a", [f_devops.id, f_finance.id], kb=kb)

        results = pool_sqlite.recall("deploy pipeline", "agent_b", top_k=5, topic_channel="devops")
        channels = {f.topic_channel for f, _ in results}
        assert "finance" not in channels

    def test_no_channel_filter_returns_all(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        """recall() without topic_channel returns facts from all channels."""
        kb = _kb("agent_a", sqlite_db)
        f1 = kb.add("Python is great for ML")
        f1.topic_channel = "ml"
        f2 = kb.add("Python is great for scripting")
        f2.topic_channel = "scripting"
        kb.replace_facts([f1, f2])  # Persist channel tags.

        pool_sqlite.publish("agent_a", [f1.id, f2.id], kb=kb)

        results = pool_sqlite.recall("Python", "agent_b", top_k=5)
        channels = {f.topic_channel for f, _ in results}
        assert "ml" in channels or "scripting" in channels

    def test_local_visibility_excluded_from_recall(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        """Facts with visibility_scope='local' are not returned to other agents."""
        kb = _kb("agent_a", sqlite_db)
        f_local = kb.add("agent A private data")
        f_local.visibility_scope = "local"
        f_global = kb.add("agent A shared fact")
        f_global.visibility_scope = "global"
        kb.replace_facts([f_local, f_global])  # Persist visibility scopes.

        pool_sqlite.publish("agent_a", [f_local.id, f_global.id], kb=kb)

        results = pool_sqlite.recall("agent A", "agent_b", top_k=10)
        contents = [f.content for f, _ in results]
        assert "agent A private data" not in contents

    def test_topic_channel_persisted_in_sqlite(self, sqlite_db: SQLiteStorage) -> None:
        """topic_channel and visibility_scope survive a SQLite round-trip."""
        f = Fact(content="test", topic_channel="devops", visibility_scope="local")
        sqlite_db.save("agent_x", [f])
        loaded = sqlite_db.load("agent_x")
        assert loaded[0].topic_channel == "devops"
        assert loaded[0].visibility_scope == "local"
