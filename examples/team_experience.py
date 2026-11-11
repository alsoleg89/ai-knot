"""Host-authored team lessons retain their observed outcome and exact sources.

Run: PYTHONPATH=src python examples/team_experience.py
The host supplies observations; this example does not execute a remote rollout.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import AlwaysOnMemory, WorkspaceMemory
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact, MemoryType


def example(directory: Path) -> dict[str, object]:
    with (
        patch("ai_knot.embedder.embed_texts", side_effect=AssertionError("embedding call")),
        patch("httpx.Client.send", side_effect=AssertionError("inference HTTP call")),
        patch("httpx.AsyncClient.send", side_effect=AssertionError("inference HTTP call")),
    ):
        return _example(directory)


def _example(directory: Path) -> dict[str, object]:
    path = str(directory / "team-experience.db")
    store = SQLiteStorage(path)
    pool = SharedMemoryPool(store, persist_stats=True)
    workspace = WorkspaceMemory(store, "release-team")
    workspace.grant("trainer", "maintain")
    trainer = AlwaysOnMemory(
        KnowledgeBase("trainer", storage=store, embed_url=""), pool=pool, workspace=workspace
    )
    sources = [
        Fact(
            "Probe observed restoration succeeded.", id="success-receipt", type=MemoryType.EPISODIC
        ),
        Fact(
            "Current rollout was reported failed.", id="failure-receipt", type=MemoryType.EPISODIC
        ),
        Fact("Attempt one timed out before retry.", id="attempt-receipt", type=MemoryType.EPISODIC),
    ]
    store.save("__shared__", sources)
    lessons = []
    failed_action = failed_lease = None
    for outcome, role, content, source in (
        ("succeeded", "operator", "Checkpoint restore: verify health before traffic.", sources[0]),
        (
            "failed",
            "reviewer",
            "Retry warning: inspect the failure receipt before retry.",
            sources[1],
        ),
    ):
        intent = trainer.create_intent(client_key=outcome, goal="Inspect a host-recorded rollout")
        lease = trainer.claim(intent.id, worker_id="host", ttl_seconds=3600)
        action = trainer.prepare_action(
            intent.id, lease_token=lease.token, action_key="rollout", request={"host_task": outcome}
        )
        trainer.begin_action(action.id, lease_token=lease.token)
        action = trainer.finish_action(
            action.id, lease_token=lease.token, outcome=outcome, result={"observed": outcome}
        )
        lessons.append(
            trainer.record_experience(
                action.id,
                expected_action_revision=action.revision,
                experience_key="conditional-lesson",
                content=content,
                basis=trainer.capture_evidence((("__shared__", source.id),)),
                namespace="__shared__",
                task_key="rollout42",
                for_role=role,
            )
        )
        if outcome == "failed":
            failed_action, failed_lease = action, lease

    # Historical attempt evidence is intentionally separate from the current outcome.
    historical = trainer.derive_fact(
        "Historical attempt warning: attempt one timed out; retain its audit receipt.",
        basis=trainer.capture_evidence((("__shared__", sources[2].id),)),
        namespace="__shared__",
        memory_type=MemoryType.PROCEDURAL,
        task_key="rollout42",
    )
    failed_ref = lessons[1].lesson.ref
    copies = []
    for _ in range(2):
        copied = trainer.derive_fact(
            "Retry warning: copied guidance still depends on the current failed outcome.",
            basis=(copies[-1] if copies else failed_ref,),
            namespace="__shared__",
            memory_type=MemoryType.PROCEDURAL,
            task_key="rollout42",
            for_role="reviewer",
        )
        copies.append(trainer.capture_evidence((("__shared__", copied.id),))[0])
    roots = trainer.independent_evidence((failed_ref, *copies))
    assert len(roots) == 1

    contexts = {}
    peer_reused = 0
    peers = {}
    for role, requirement, lesson in zip(
        ("operator", "reviewer"), ("checkpoint restore", "retry warning"), lessons, strict=True
    ):
        independent_store = SQLiteStorage(path)
        peer = AlwaysOnMemory(
            KnowledgeBase(role, storage=independent_store, embed_url=""),
            pool=SharedMemoryPool(independent_store, persist_stats=True),
        )
        peers[role] = peer
        peer_reused += len(peer.review_context((lesson.lesson.ref,)).accepted)
        view = peer.select_context(
            (requirement,), task_key="rollout42", role=role, max_facts=2, max_chars=600
        )
        assert view.complete
        contexts[role] = [item.content for item in view.facts]
    assert peer_reused == 2 and failed_action is not None and failed_lease is not None

    # Reopening without a pool still discovers and retracts the shared binding.
    reopened = AlwaysOnMemory(KnowledgeBase("trainer", storage=SQLiteStorage(path), embed_url=""))
    reopened.reconcile_action(
        failed_action.id,
        lease_token=failed_lease.token,
        outcome="succeeded",
        evidence={"independent_receipt": "eventual restoration confirmed"},
    )
    reviewer = peers["reviewer"]
    rejected = reviewer.review_context((failed_ref, *copies))
    retained = reviewer.capture_evidence((("__shared__", historical.id),))
    survives = reviewer.review_context(retained).complete
    assert not rejected.accepted and survives
    current = reviewer.select_context(("retry warning",), task_key="rollout42")
    assert not {item.ref for item in current.facts}.intersection((failed_ref, *copies))
    assert peers["operator"].review_context((lessons[0].lesson.ref,)).complete
    return {
        "observed_outcomes": [lesson.outcome for lesson in lessons],
        "peer_reused_current_lessons": peer_reused,
        "contexts_before_correction": contexts,
        "independent_roots_for_original_and_copies": len(roots),
        "old_current_lesson_rejected_after_reconciliation": not rejected.accepted,
        "historical_attempt_warning_survives": survives,
        "model_calls": 0,
        "scope": "Scripted observations and consistency; "
        "remote success and model accuracy unmeasured",
    }


if __name__ == "__main__":
    with tempfile.TemporaryDirectory(prefix="knot-team-experience-") as directory:
        print(json.dumps(example(Path(directory)), ensure_ascii=False, indent=2))
