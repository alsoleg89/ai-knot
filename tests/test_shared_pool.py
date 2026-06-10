"""Tests for SharedMemoryPool: publish, recall, sync_dirty, sync_slot_deltas."""

from __future__ import annotations

import pathlib

import pytest

from ai_knot._pool_recall import _abstention
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


def _add_slot(kb: KnowledgeBase, content: str, slot_key: str) -> Fact:
    """Add a fact and tag it with *slot_key* (mirrors backend ``add_structured``)."""
    fact = kb.add(content)
    facts = kb.list_facts()
    for f in facts:
        if f.id == fact.id:
            f.slot_key = slot_key
    kb.replace_facts(facts)
    return fact


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

    def test_require_evidence_skips_assertion_without_pointer(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        # add()-created facts carry no source pointer; the opt-in gate skips them.
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add("Alex earns 95k")
        published = pool_sqlite.publish("agent_a", [fact.id], kb=kb, require_evidence=True)
        assert published == []

    def test_require_evidence_admits_fact_with_verbatim(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add("Alex earns 95k")
        # Attach a provenance pointer, as learn()-extracted facts carry.
        facts = sqlite_db.load("agent_a")
        for f in facts:
            f.source_verbatim = "Alex told me he earns 95k a year"
        sqlite_db.save("agent_a", facts)
        published = pool_sqlite.publish("agent_a", [fact.id], kb=kb, require_evidence=True)
        assert len(published) == 1

    def test_require_evidence_skips_unsupported_fact(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add("Alex earns 95k")
        facts = sqlite_db.load("agent_a")
        for f in facts:
            f.source_verbatim = "Alex earns 95k"
            f.supported = False  # faithfulness filter rejected it
        sqlite_db.save("agent_a", facts)
        published = pool_sqlite.publish("agent_a", [fact.id], kb=kb, require_evidence=True)
        assert published == []

    def test_default_publish_ignores_evidence(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        # Backward-compat: default require_evidence=False publishes pointer-less facts.
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add("Alex earns 95k")
        published = pool_sqlite.publish("agent_a", [fact.id], kb=kb)
        assert len(published) == 1

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


# ---------------------------------------------------------------------------
# Auto-trust — get_trust() computed from publish/recall behaviour
# ---------------------------------------------------------------------------


class TestTrustOrdering:
    """Over-fetch ensures high-trust facts beat high-score low-trust facts."""

    def test_overfetch_preserves_high_trust_facts(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        """A medium-score high-trust fact must beat a high-score low-trust fact at top_k=1."""
        # Publish a highly-relevant fact from agent_a (will have raw high score).
        kb_a = _kb("agent_a", sqlite_db)
        noisy = kb_a.add("Python machine learning neural network deep learning AI")
        pool_sqlite.publish("agent_a", [noisy.id], kb=kb_a)

        # Publish a less-keyword-dense but still-relevant fact from agent_b.
        kb_b = _kb("agent_b", sqlite_db)
        solid = kb_b.add("Python is used for ML work")
        pool_sqlite.publish("agent_b", [solid.id], kb=kb_b)

        # Artificially tank agent_a's trust: one verifiable (slot) publish that was
        # quick-invalidated → quick_inv_rate = 1/1 = 1.0.  (quick_inv only accrues on
        # slot facts, so the slot-publish ledger must be set alongside it.)
        pool_sqlite._slot_publish_count["agent_a"] = 1
        pool_sqlite._quick_inv_count["agent_a"] = 1
        assert pool_sqlite.get_trust("agent_a") == pytest.approx(0.1)

        # Give agent_b high trust (used >> published overcomes Bayesian prior).
        pool_sqlite._publish_count["agent_b"] = 1
        pool_sqlite._used_count["agent_b"] = 5
        assert pool_sqlite.get_trust("agent_b") == pytest.approx(1.0)

        results = pool_sqlite.recall("Python ML", "agent_c", top_k=1)
        assert len(results) == 1
        # The returned fact must come from the high-trust agent_b.
        returned_fact, _ = results[0]
        assert returned_fact.origin_agent_id == "agent_b"


class TestConcurrentPublish:
    """_publish_lock prevents lost updates under concurrent publish calls."""

    def test_concurrent_publish_different_slots(self, sqlite_db: SQLiteStorage) -> None:
        """Two threads publishing different slot_keys both land in the pool."""
        import threading

        pool = SharedMemoryPool(storage=sqlite_db)
        pool.register("agent_a")
        pool.register("agent_b")

        kb_a = _kb("agent_a", sqlite_db)
        kb_b = _kb("agent_b", sqlite_db)
        fa = kb_a.add("Alex::role = engineer")
        fa.slot_key = "Alex::role"
        kb_a.replace_facts([fa])
        fb = kb_b.add("Bob::role = manager")
        fb.slot_key = "Bob::role"
        kb_b.replace_facts([fb])

        errors: list[Exception] = []

        def pub_a() -> None:
            try:
                pool.publish("agent_a", [fa.id], kb=kb_a)
            except Exception as exc:
                errors.append(exc)

        def pub_b() -> None:
            try:
                pool.publish("agent_b", [fb.id], kb=kb_b)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=pub_a)
        t2 = threading.Thread(target=pub_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        shared = pool.list_shared_facts()
        active_slots = {f.slot_key for f in shared if f.is_active() and f.slot_key}
        assert "Alex::role" in active_slots
        assert "Bob::role" in active_slots

    def test_concurrent_publish_same_slot(self, sqlite_db: SQLiteStorage) -> None:
        """Two threads racing to the same slot_key produce exactly one active fact."""
        import threading

        pool = SharedMemoryPool(storage=sqlite_db)
        pool.register("agent_a")
        pool.register("agent_b")

        kb_a = _kb("agent_a", sqlite_db)
        kb_b = _kb("agent_b", sqlite_db)
        fa = kb_a.add("Alex earns 80k")
        fa.slot_key = "Alex::salary"
        kb_a.replace_facts([fa])
        fb = kb_b.add("Alex earns 95k")
        fb.slot_key = "Alex::salary"
        kb_b.replace_facts([fb])

        errors: list[Exception] = []

        def pub_a() -> None:
            try:
                pool.publish("agent_a", [fa.id], kb=kb_a)
            except Exception as exc:
                errors.append(exc)

        def pub_b() -> None:
            try:
                pool.publish("agent_b", [fb.id], kb=kb_b)
            except Exception as exc:
                errors.append(exc)

        t1 = threading.Thread(target=pub_a)
        t2 = threading.Thread(target=pub_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        shared = pool.list_shared_facts()
        active_salary = [f for f in shared if f.is_active() and f.slot_key == "Alex::salary"]
        assert len(active_salary) == 1


class TestAutoTrust:
    def test_no_track_record_returns_default(self, pool_sqlite: SharedMemoryPool) -> None:
        """Agent with no published facts gets _PROVENANCE_DISCOUNT trust."""
        trust = pool_sqlite.get_trust("agent_a")
        assert trust == pytest.approx(0.8)

    def test_trust_grows_with_recall_hits(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        """trust rises after facts from this agent are returned in recall."""
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add("Python is the best ML language")
        pool_sqlite.publish("agent_a", [fact.id], kb=kb)

        # Trigger recall so agent_a's fact counts as "used".
        pool_sqlite.recall("Python ML", "agent_b", top_k=5)

        trust = pool_sqlite.get_trust("agent_a")
        assert trust > 0.1  # non-trivial trust

    def test_unpublished_agent_does_not_accumulate_trust(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        """An agent that never publishes stays at default trust regardless of queries."""
        # Publish from agent_a only.
        kb_a = _kb("agent_a", sqlite_db)
        fact = kb_a.add("irrelevant information")
        pool_sqlite.publish("agent_a", [fact.id], kb=kb_a)

        pool_sqlite.recall("Python", "agent_b", top_k=5)

        # agent_b never published — should stay at default.
        assert pool_sqlite.get_trust("agent_b") == pytest.approx(0.8)

    def test_quick_invalidation_lowers_trust(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        """Facts superseded quickly by a different agent penalise the originator."""
        kb_a = _kb("agent_a", sqlite_db)
        old = kb_a.add("Alex earns 50k")
        old.slot_key = "Alex::salary"
        kb_a.replace_facts([old])
        pool_sqlite.publish("agent_a", [old.id], kb=kb_a)

        # agent_b immediately supersedes agent_a's slot — within the window.
        kb_b = _kb("agent_b", sqlite_db)
        new = kb_b.add("Alex earns 95k")
        new.slot_key = "Alex::salary"
        kb_b.replace_facts([new])
        pool_sqlite.publish("agent_b", [new.id], kb=kb_b)

        # agent_a's fact was quickly superseded — its trust should be penalised.
        trust_a = pool_sqlite.get_trust("agent_a")
        # Published 1, used 0, quick_inv 1 → trust = 0 * (1-1) = 0, clamped to 0.1
        assert trust_a == pytest.approx(0.1)


class TestMonotonicCASAndTrustIntegrity:
    """Monotonic CAS + laundering-resistant, recoverable trust.

    The quick-invalidation penalty is a rate over *verifiable* (slot) publish
    events, not over total publish volume.  This keeps trust resistant to
    reputation-laundering (free-standing spam cannot lower the rate) while still
    allowing recovery (re-asserting a slot with a corrected value adds an event
    and dilutes the penalty).  The monotonic-CAS guard stops an agent from
    re-claiming a slot a peer has corrected by replaying its own stale fact.
    """

    def test_stale_replay_does_not_reclaim_peer_slot(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        """Re-publishing an already-superseded fact must not re-poison a peer's slot."""
        kb_a = _kb("agent_a", sqlite_db)
        x = _add_slot(kb_a, "Alex earns 50k", "Alex::salary")
        pool_sqlite.publish("agent_a", [x.id], kb=kb_a)

        # agent_b corrects the slot — its value is now authoritative.
        kb_b = _kb("agent_b", sqlite_db)
        y = _add_slot(kb_b, "Alex earns 95k", "Alex::salary")
        pool_sqlite.publish("agent_b", [y.id], kb=kb_b)

        # agent_a replays its stale fact (still active in its private KB).
        pool_sqlite.publish("agent_a", [x.id], kb=kb_a)

        active = [
            f
            for f in pool_sqlite.list_shared_facts()
            if f.slot_key == "Alex::salary" and f.valid_until is None
        ]
        assert len(active) == 1
        assert active[0].origin_agent_id == "agent_b"
        assert "95k" in active[0].content

    def test_fresh_correction_supersedes_peer_slot(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        """A NEW fact (not a stale replay) legitimately supersedes the slot holder."""
        kb_a = _kb("agent_a", sqlite_db)
        x = _add_slot(kb_a, "Alex earns 50k", "Alex::salary")
        pool_sqlite.publish("agent_a", [x.id], kb=kb_a)

        kb_b = _kb("agent_b", sqlite_db)
        y = _add_slot(kb_b, "Alex earns 95k", "Alex::salary")
        pool_sqlite.publish("agent_b", [y.id], kb=kb_b)

        # agent_a re-asserts with a brand-new fact (corrected value, fresh id).
        z = _add_slot(kb_a, "Alex earns 110k after raise", "Alex::salary")
        pool_sqlite.publish("agent_a", [z.id], kb=kb_a)

        active = [
            f
            for f in pool_sqlite.list_shared_facts()
            if f.slot_key == "Alex::salary" and f.valid_until is None
        ]
        assert len(active) == 1
        assert active[0].origin_agent_id == "agent_a"
        assert "110k" in active[0].content

    def test_freestanding_spam_cannot_launder_trust(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        """An agent caught publishing a bad slot cannot wash it out with noise."""
        kb_a = _kb("agent_a", sqlite_db)
        bad = _add_slot(kb_a, "Alex earns 50k", "Alex::salary")
        pool_sqlite.publish("agent_a", [bad.id], kb=kb_a)

        # Peer supersedes the only verifiable claim → trust hits the floor.
        kb_b = _kb("agent_b", sqlite_db)
        fix = _add_slot(kb_b, "Alex earns 95k", "Alex::salary")
        pool_sqlite.publish("agent_b", [fix.id], kb=kb_b)
        assert pool_sqlite.get_trust("agent_a") == pytest.approx(0.1)

        # agent_a floods the pool with unverifiable free-standing facts.
        for i in range(10):
            noise = kb_a.add(f"Unverifiable marketing claim number {i}")
            pool_sqlite.publish("agent_a", [noise.id], kb=kb_a)

        # Free-standing volume must NOT dilute the invalidation rate.
        assert pool_sqlite.get_trust("agent_a") == pytest.approx(0.1)

    def test_slot_reassertion_recovers_trust(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        """A penalised agent recovers by re-asserting slots with surviving values."""
        kb_a = _kb("agent_a", sqlite_db)
        s1 = _add_slot(kb_a, "Mia salary 80k", "Mia::salary")
        s2 = _add_slot(kb_a, "Mia title Junior", "Mia::title")
        pool_sqlite.publish("agent_a", [s1.id, s2.id], kb=kb_a)

        # Both verifiable claims superseded by a peer → floor.
        kb_b = _kb("agent_b", sqlite_db)
        c1 = _add_slot(kb_b, "Mia salary 120k", "Mia::salary")
        c2 = _add_slot(kb_b, "Mia title Senior", "Mia::title")
        pool_sqlite.publish("agent_b", [c1.id, c2.id], kb=kb_b)
        trust_floor = pool_sqlite.get_trust("agent_a")
        assert trust_floor == pytest.approx(0.1)

        # agent_a re-asserts both slots with fresh corrected facts (which survive).
        r1 = _add_slot(kb_a, "Mia salary 125k confirmed", "Mia::salary")
        r2 = _add_slot(kb_a, "Mia title Staff confirmed", "Mia::title")
        pool_sqlite.publish("agent_a", [r1.id, r2.id], kb=kb_a)
        pool_sqlite.recall("Mia salary title", "agent_c", top_k=5)

        # Slot events 2 → 4 while quick_inv stays 2 → penalty 1.0 → 0.5 → recovery.
        assert pool_sqlite.get_trust("agent_a") > trust_floor

    def test_wide_query_suppresses_known_malicious_agent(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        """Even a WIDE (empty-KB) query must not surface a known-malicious agent first.

        WIDE/onboarding queries skip the trust discount for ordinary agents (the
        querier has no basis to judge an unseen agent), but an agent whose verifiable
        claims peers actively superseded stays suppressed.
        """
        pool_sqlite.register("agent_x")

        # Make agent_a known-malicious: its only verifiable claim is superseded.
        kb_a = _kb("agent_a", sqlite_db)
        bad_slot = _add_slot(kb_a, "Alex earns 50k", "Alex::salary")
        pool_sqlite.publish("agent_a", [bad_slot.id], kb=kb_a)
        kb_b = _kb("agent_b", sqlite_db)
        fix = _add_slot(kb_b, "Alex earns 95k", "Alex::salary")
        pool_sqlite.publish("agent_b", [fix.id], kb=kb_b)
        assert pool_sqlite.get_trust("agent_a") == pytest.approx(0.1)

        # agent_a (malicious) and agent_x (neutral, never invalidated) publish
        # equally-relevant free-standing facts.
        bad = kb_a.add("alpha protocol handles message encryption")
        pool_sqlite.publish("agent_a", [bad.id], kb=kb_a)
        kb_x = _kb("agent_x", sqlite_db)
        good = kb_x.add("beta protocol handles message encryption")
        pool_sqlite.publish("agent_x", [good.id], kb=kb_x)

        # agent_c queries with an EMPTY private KB → WIDE exploration mode.
        results = pool_sqlite.recall("protocol encryption", "agent_c", top_k=2)
        ranked = [f.origin_agent_id for f, _ in results]
        assert ranked, "expected results"
        # The neutral agent's fact outranks the known-malicious agent's, even in WIDE.
        assert ranked[0] == "agent_x"


class TestVisibilityScope:
    """visibility_scope writer (add/publish) + per-agent read-access projection."""

    _CONTENT = "deployment uses FluxCD on cluster prod"
    _QUERY = "FluxCD deployment"

    def test_named_scope_hidden_without_grant(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add(self._CONTENT, visibility_scope="team:infra")
        pool_sqlite.publish("agent_a", [fact.id], kb=kb)
        # agent_b has no grant → cannot see the team:infra fact.
        results_b = pool_sqlite.recall(self._QUERY, "agent_b", top_k=5)
        assert all("FluxCD" not in f.content for f, _ in results_b)
        # The origin agent always sees its own fact.
        results_a = pool_sqlite.recall(self._QUERY, "agent_a", top_k=5)
        assert any("FluxCD" in f.content for f, _ in results_a)

    def test_named_scope_visible_after_grant(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add(self._CONTENT, visibility_scope="team:infra")
        pool_sqlite.publish("agent_a", [fact.id], kb=kb)
        pool_sqlite.grant_read("agent_b", "team:infra")
        results_b = pool_sqlite.recall(self._QUERY, "agent_b", top_k=5)
        assert any("FluxCD" in f.content for f, _ in results_b)

    def test_global_scope_visible_to_all(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add(self._CONTENT)  # default "global"
        pool_sqlite.publish("agent_a", [fact.id], kb=kb)
        results_b = pool_sqlite.recall(self._QUERY, "agent_b", top_k=5)
        assert any("FluxCD" in f.content for f, _ in results_b)

    def test_publish_scope_override(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        # publish(visibility_scope=...) overrides the add()-time scope.
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add(self._CONTENT)  # global at add()
        pool_sqlite.publish("agent_a", [fact.id], kb=kb, visibility_scope="team:infra")
        results_b = pool_sqlite.recall(self._QUERY, "agent_b", top_k=5)
        assert all("FluxCD" not in f.content for f, _ in results_b)


class TestAbstention:
    """Deterministic abstention signal (_abstention helper + pool accessors)."""

    def test_helper_empty_results(self) -> None:
        risk, abstain = _abstention([], 0.0)
        assert risk == 1.0
        assert abstain is True

    def test_helper_no_evidence_abstains(self) -> None:
        fact = Fact(content="deployment uses FluxCD")  # no source pointer
        risk, abstain = _abstention([(fact, 1.0)], coverage=1.0)
        assert abstain is True  # evidence_frac == 0
        assert risk == 1.0

    def test_helper_evidence_high_coverage_no_abstain(self) -> None:
        fact = Fact(content="deployment uses FluxCD", source_verbatim="we deploy FluxCD")
        risk, abstain = _abstention([(fact, 1.0)], coverage=1.0)
        assert abstain is False
        assert risk == 0.0

    def test_helper_low_coverage_abstains(self) -> None:
        fact = Fact(content="deployment uses FluxCD", source_verbatim="we deploy FluxCD")
        risk, abstain = _abstention([(fact, 1.0)], coverage=0.3)
        assert abstain is True  # coverage < 0.5
        assert risk == pytest.approx(0.7)

    def test_recall_no_results_abstains(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        results = pool_sqlite.recall("nothing in the empty pool", "agent_b", top_k=5)
        assert results == []
        assert pool_sqlite.last_recall_abstains is True
        assert pool_sqlite.last_recall_risk == pytest.approx(1.0)

    def test_recall_over_unsupported_facts_abstains(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add("deployment uses FluxCD on cluster prod")  # no evidence pointer
        pool_sqlite.publish("agent_a", [fact.id], kb=kb)
        pool_sqlite.recall("FluxCD deployment", "agent_b", top_k=5)
        assert pool_sqlite.last_recall_abstains is True

    def test_recall_over_evidence_facts_does_not_abstain(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add("deployment uses FluxCD on cluster prod")
        facts = sqlite_db.load("agent_a")
        for f in facts:
            f.source_verbatim = "we deploy with FluxCD on prod"
        sqlite_db.save("agent_a", facts)
        pool_sqlite.publish("agent_a", [fact.id], kb=kb)
        pool_sqlite.recall("FluxCD deployment", "agent_b", top_k=5)
        assert pool_sqlite.last_recall_abstains is False


class TestProvenance:
    """Provenance lineage (published_by / promoted_by / supersedes_id) persisted via qualifiers."""

    def test_published_by_set_and_persists(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add("Alex earns 95k")
        pool_sqlite.publish("agent_a", [fact.id], kb=kb)
        # Reload from storage to prove the lineage persisted (not just in-memory).
        shared = pool_sqlite.list_shared_facts()
        pub = next(f for f in shared if f.content == "Alex earns 95k")
        assert pub.provenance.published_by == "agent_a"
        assert pub.provenance.origin_agent == "agent_a"

    def test_supersedes_id_records_cas_predecessor(
        self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool
    ) -> None:
        kb_a = _kb("agent_a", sqlite_db)
        kb_b = _kb("agent_b", sqlite_db)
        f1 = kb_a.add_resolved(
            [
                Fact(
                    content="Alex at Globex",
                    entity="Alex",
                    attribute="employer",
                    value_text="Globex",
                )
            ]
        )[0]
        pool_sqlite.publish("agent_a", [f1.id], kb=kb_a)
        f2 = kb_b.add_resolved(
            [
                Fact(
                    content="Alex at Initech",
                    entity="Alex",
                    attribute="employer",
                    value_text="Initech",
                )
            ]
        )[0]
        pool_sqlite.publish("agent_b", [f2.id], kb=kb_b)
        shared = pool_sqlite.list_shared_facts()
        active = [f for f in shared if f.is_active() and f.slot_key == "Alex::employer"]
        assert len(active) == 1
        assert active[0].provenance.supersedes_id == f1.id

    def test_promoted_by_set(self, sqlite_db: SQLiteStorage, pool_sqlite: SharedMemoryPool) -> None:
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add("Alex earns 95k")
        pub = pool_sqlite.publish("agent_a", [fact.id], kb=kb)[0]
        pool_sqlite.promote("agent_a", [pub.id], tier="org")
        promoted = next(f for f in pool_sqlite.list_shared_facts() if f.id == pub.id)
        assert promoted.provenance.promoted_by == "agent_a"
        assert promoted.memory_tier == "org"

    def test_private_fact_has_empty_lineage(self) -> None:
        # A freshly created fact has no publish/promote lineage yet.
        fact = Fact(content="x", origin_agent_id="agent_a")
        assert fact.provenance.published_by == ""
        assert fact.provenance.promoted_by == ""
        assert fact.provenance.supersedes_id == ""


class TestPersistTrust:
    """Opt-in durable trust/usage telemetry (PoolStatsCapable + persist_stats)."""

    def test_trust_survives_restart_when_enabled(self, sqlite_db: SQLiteStorage) -> None:
        pool = SharedMemoryPool(storage=sqlite_db, persist_stats=True)
        pool.register("agent_a")
        pool.register("agent_b")
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add("Alex earns 95k annually")
        pool.publish("agent_a", [fact.id], kb=kb)
        pool.recall("Alex salary", "agent_b", top_k=3)  # accrues used_count
        pool.flush_stats()

        # A fresh pool on the same DB restores the telemetry.
        pool2 = SharedMemoryPool(storage=sqlite_db, persist_stats=True)
        assert pool2._publish_count.get("agent_a", 0) == 1
        assert pool2.get_trust("agent_a") == pool.get_trust("agent_a")

    def test_trust_not_persisted_by_default(self, sqlite_db: SQLiteStorage) -> None:
        pool = SharedMemoryPool(storage=sqlite_db)  # persist_stats=False
        pool.register("agent_a")
        kb = _kb("agent_a", sqlite_db)
        fact = kb.add("Alex earns 95k")
        pool.publish("agent_a", [fact.id], kb=kb)
        # Default pool writes no telemetry; a fresh instance restores nothing.
        pool2 = SharedMemoryPool(storage=sqlite_db, persist_stats=True)
        assert pool2._publish_count.get("agent_a", 0) == 0

    def test_yaml_backend_persists(self, tmp_path: pathlib.Path) -> None:
        storage = YAMLStorage(base_dir=str(tmp_path))
        pool = SharedMemoryPool(storage=storage, persist_stats=True)
        pool.register("agent_a")
        kb = KnowledgeBase("agent_a", storage=storage)
        fact = kb.add("Alex earns 95k")
        pool.publish("agent_a", [fact.id], kb=kb)

        pool2 = SharedMemoryPool(storage=YAMLStorage(base_dir=str(tmp_path)), persist_stats=True)
        assert pool2._publish_count.get("agent_a", 0) == 1
