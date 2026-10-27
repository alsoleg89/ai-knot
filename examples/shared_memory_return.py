"""Two agents share current context after a pause; no model or network required.

Run: PYTHONPATH=src python examples/shared_memory_return.py
The host's cache/receipt file is saved atomically BEFORE acknowledging delivery.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import (
    AlwaysOnMemory,
    ContextReview,
    CursorExpired,
    EvidenceRef,
    WatchSelector,
)
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact


def example(directory: Path) -> dict[str, object]:
    database = directory / "memory.db"

    def reader() -> AlwaysOnMemory:
        store = SQLiteStorage(str(database))
        return AlwaysOnMemory(
            KnowledgeBase("planner", storage=store, embed_url=""),
            pool=SharedMemoryPool(store, persist_stats=True),
        )

    writer_store = SQLiteStorage(str(database))
    writer = KnowledgeBase("researcher", storage=writer_store, embed_url="")
    pool = SharedMemoryPool(writer_store, persist_stats=True)
    pool.register("researcher")

    def save_cache(delivery_id: str, context: ContextReview) -> None:
        # One durable host transaction contains both replacement and receipt.
        with closing(sqlite3.connect(directory / "cache.db")) as cache, cache:
            cache.execute(
                "CREATE TABLE IF NOT EXISTS cache "
                "(id INTEGER PRIMARY KEY, delivery_id TEXT, context TEXT, refs TEXT)"
            )
            cache.execute(
                "INSERT INTO cache VALUES(1,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                "delivery_id=excluded.delivery_id,context=excluded.context,refs=excluded.refs",
                (
                    delivery_id,
                    json.dumps([item.content for item in context.accepted]),
                    json.dumps([asdict(item.ref) for item in context.accepted]),
                ),
            )

    def publish(region: str) -> None:
        fact = Fact(f"Deploy in {region}", slot_key="deployment::region", value_text=region)
        writer.add_resolved([fact])
        assert pool.publish("researcher", [fact.id], kb=writer)

    memory = reader()
    memory.subscribe(
        subscription_id="deployment",
        selector=WatchSelector("__shared__", slot_keys=("deployment::region",)),
    )
    publish("EU")
    initial = memory.resync("deployment")
    old_refs = tuple(EvidenceRef(e.namespace, e.fact_id, e.after_revision) for e in initial.events)
    save_cache(initial.delivery_id, memory.review_context(old_refs))
    memory.ack("deployment", delivery_id=initial.delivery_id)
    del old_refs

    # The planner sleeps. Its teammate changes the decision; retention runs.
    publish("US")
    memory.prune_changes("__shared__", force=True)
    memory = reader()
    with closing(sqlite3.connect(directory / "cache.db")) as cache:
        saved = cache.execute("SELECT refs FROM cache WHERE id=1").fetchone()
    old_refs = tuple(EvidenceRef(**item) for item in json.loads(saved[0]))
    stale = memory.review_context(old_refs)
    assert not stale.accepted and not stale.complete
    try:
        memory.poll("deployment")
    except CursorExpired:
        snapshot = memory.resync("deployment")
    else:
        raise AssertionError("Expected an explicit retention gap")
    refs = tuple(EvidenceRef(e.namespace, e.fact_id, e.after_revision) for e in snapshot.events)
    view = memory.review_context(refs)
    assert view.complete and [f.content for f in view.accepted] == ["Deploy in US"]

    # Cache + receipt form ONE host commit. Restart can safely replay or ack.
    save_cache(snapshot.delivery_id, view)
    memory.ack("deployment", delivery_id=snapshot.delivery_id)
    idle = memory.poll("deployment")
    assert idle.delivery_id is None and not idle.has_more
    return {
        "current_context": [f.content for f in view.accepted],
        "stale_context_rejected": True,
        "snapshot_recovered": True,
        "model_calls": 0,
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="knot-return-") as temporary:
        print(json.dumps(example(Path(temporary)), sort_keys=True))
