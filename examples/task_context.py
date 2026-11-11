"""One team corpus, two recipient roles, and a retracted outcome.

Run: PYTHONPATH=src python examples/task_context.py
Lessons below are host-authored and conditional; no reflection/model is invoked.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import AlwaysOnMemory, SnapshotTooLarge, WorkspaceMemory
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact, MemoryType


def example(directory: Path) -> dict[str, object]:
    store = SQLiteStorage(str(directory / "team.db"))
    pool = SharedMemoryPool(store, persist_stats=True)
    pool.register("trainer")
    workspace = WorkspaceMemory(store, "release-team")
    workspace.grant("trainer", "maintain")
    writer = KnowledgeBase("trainer", storage=store, embed_url="")
    trainer = AlwaysOnMemory(writer, pool=pool, workspace=workspace)

    def receipt(value: str) -> Fact:
        fact = Fact(
            "Recorded restoration result: " + value, slot_key="release42::result", value_text=value
        )
        writer.add_resolved([fact])
        return pool.publish("trainer", [fact.id], kb=writer)[0]

    source = receipt("verified")
    basis = trainer.capture_evidence((("__shared__", source.id),))
    expected: dict[str, set[str]] = {"operator": set(), "reviewer": set()}
    for role, first, second in (
        (
            "operator",
            "Rollback checkpoint: restore the verified image.",
            "Certificate rotation: stage credentials before switching traffic.",
        ),
        (
            "reviewer",
            "Rollback checkpoint: inspect the restoration receipt.",
            "Certificate rotation: audit certificate validity before approval.",
        ),
    ):
        for content in (first, second):
            lesson = trainer.derive_fact(
                content,
                basis=basis,
                namespace="__shared__",
                memory_type=MemoryType.PROCEDURAL,
                task_key="release42",
                for_role=role,
            )
            expected[role].add(lesson.id)
    trainer.derive_fact(
        "Restore warning: preserve an independent backup.",
        basis=basis,
        namespace="__shared__",
        memory_type=MemoryType.PROCEDURAL,
    )

    requirements = ("rollback checkpoint", "certificate rotation", "restore warning")
    readers = {}
    views = {}
    for role in ("operator", "reviewer"):
        readers[role] = AlwaysOnMemory(
            KnowledgeBase(role, storage=SQLiteStorage(str(directory / "team.db")), embed_url=""),
            pool=SharedMemoryPool(SQLiteStorage(str(directory / "team.db")), persist_stats=True),
        )
        views[role] = readers[role].select_context(
            requirements, task_key="release42", role=role, max_facts=3, max_chars=600
        )
        assert views[role].complete
        assert expected[role] <= {entry.ref.fact_id for entry in views[role].facts}

    # A fitting task corpus can be loaded without inventing search requirements.
    snapshot = readers["operator"].task_snapshot(task_key="release42")
    assert len(snapshot) == 6
    try:
        readers["operator"].task_snapshot(task_key="release42", max_facts=3)
    except SnapshotTooLarge:
        overflow_handled = True  # The host can use the explicit selector above.
    else:
        raise AssertionError("A partial task snapshot must not be returned")

    # A host rebuilding from observations can omit generated lessons explicitly.
    source_only = readers["operator"].task_snapshot(
        task_key="release42", include_derived=False, max_facts=3, max_chars=600
    )
    assert tuple(item.ref for item in source_only) == basis
    assert readers["operator"].review_context(tuple(item.ref for item in source_only)).complete

    # A corrected shared receipt retracts every projection tied to its old revision.
    receipt("withdrawn")
    rejected = readers["operator"].review_context(
        tuple(entry.ref for entry in views["operator"].facts)
    )
    fresh = readers["operator"].select_context(requirements, task_key="release42", role="operator")
    assert not rejected.accepted and not fresh.facts and not fresh.complete
    return {
        "operator_role_fit": True,
        "reviewer_role_fit": True,
        "retracted_source_blocks_reuse": True,
        "snapshot_facts": len(snapshot),
        "snapshot_overflow_handled": overflow_handled,
        "source_only_facts": len(source_only),
        "model_calls": 0,
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="knot-task-context-") as directory:
        print(json.dumps(example(Path(directory)), sort_keys=True))
