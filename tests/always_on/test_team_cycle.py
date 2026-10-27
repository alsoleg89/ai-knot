"""Public host recipe composes memory without inferring effects."""

import importlib.util
import sys
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from ai_knot import KnowledgeBase
from ai_knot.always_on import (
    AlwaysOnMemory,
    IntentState,
    RevisionConflict,
    VerificationOutcome,
    VerificationResult,
    WaitUntil,
    WatchSelector,
)
from ai_knot.types import Fact


@pytest.fixture
def recipe():
    path = Path(__file__).resolve().parents[2] / "examples/always_on_team.py"
    spec = importlib.util.spec_from_file_location("always_on_team", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def setup(recipe, sql_storage, clock, tmp_path):
    kb = KnowledgeBase("owner", storage=sql_storage, embed_url="")
    memory = AlwaysOnMemory(kb, clock=clock)
    source = kb.add("Backup receipt verified")
    intent = memory.create_intent(client_key="restore", goal="Restore after timer")
    memory.subscribe(
        subscription_id="watch", selector=WatchSelector("owner", fact_ids=(source.id,))
    )
    inbox = recipe.Inbox(tmp_path / "inbox.db")

    def inspect(**kwargs):
        return recipe.inspect_turn(
            memory,
            inbox,
            intent_id=intent.id,
            subscription_id="watch",
            requirements=("backup receipt",),
            **kwargs,
        )

    return kb, memory, source, intent, inbox, inspect


def wait(memory, intent, clock, basis=()):
    lease = memory.claim(intent.id, worker_id="host")
    current = memory.get_intent(intent.id)
    memory.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=current.revision,
        state=IntentState.WAITING,
        progress={},
        waiting_for=WaitUntil(clock()),
        basis=basis,
    )


def test_ready_is_not_suppressed_by_receiving_context_and_claims_are_separate(
    recipe, sql_storage, clock, tmp_path
):
    _, memory, _, intent, _, inspect = setup(recipe, sql_storage, clock, tmp_path)
    first, second = inspect(), inspect()
    assert first.status == second.status == "ready" and first.wake_model and second.wake_model
    memory.claim(intent.id, worker_id="one")
    with pytest.raises(RevisionConflict):
        memory.claim(intent.id, worker_id="two")
    assert inspect().status == "lease_busy" and not inspect().wake_model


def test_filtered_backlog_never_looks_ready(recipe, sql_storage, clock, tmp_path):
    kb, memory, source, _, _, inspect = setup(recipe, sql_storage, clock, tmp_path)
    kb.replace_facts([source, *[Fact(f"Noise {i}") for i in range(1200)]])
    kb.replace_facts(
        [
            replace(f, content="Backup receipt corrected") if f.id == source.id else f
            for f in kb.list_facts()
        ]
    )
    first = inspect(max_pages=1)
    assert first.status == "catching_up" and not first.wake_model
    assert first.diagnostics()["through_seq"] < first.diagnostics()["head_seq"]
    current = inspect()
    assert current.wake_model and current.context.facts[0].content == "Backup receipt corrected"


def test_inbox_commit_precedes_ack_and_replay_survives_crash(
    recipe, sql_storage, clock, tmp_path, monkeypatch
):
    kb, memory, _, intent, inbox, inspect = setup(recipe, sql_storage, clock, tmp_path)
    memory.unsubscribe("watch")
    memory.subscribe(subscription_id="watch", selector=WatchSelector("owner"))
    kb.add("Backup receipt latest")
    original = memory.ack

    def crash(*args, **kwargs):
        assert inbox.saved_refs("owner", "watch")
        raise RuntimeError("Crash after durable host commit")

    monkeypatch.setattr(memory, "ack", crash)
    with pytest.raises(RuntimeError, match="Crash after"):
        inspect()
    assert memory.poll("watch").delivery_id is not None
    monkeypatch.setattr(memory, "ack", original)
    assert inspect().wake_model
    assert memory.poll("watch").delivery_id is None
    assert memory.get_intent(intent.id).lease is None


def test_retention_gap_resync_replaces_saved_refs(recipe, sql_storage, clock, tmp_path):
    kb, memory, source, _, inbox, inspect = setup(recipe, sql_storage, clock, tmp_path)
    old = inspect().context.facts[0].ref
    kb.replace_facts([replace(source, content="Backup receipt corrected")])
    memory.prune_changes(force=True)
    current = inspect()
    assert current.snapshots == 1 and current.rejected_saved == 1
    assert old not in inbox.saved_refs("owner", "watch")
    assert current.context.facts[0].content == "Backup receipt corrected"


def test_wall_clock_verification_due_blocks_inference_without_new_events(
    recipe, sql_storage, clock, tmp_path
):
    _, memory, source, intent, _, inspect = setup(recipe, sql_storage, clock, tmp_path)
    refs = memory.capture_evidence((("owner", source.id),))
    wait(memory, intent, clock, refs)
    memory.set_freshness(refs[0], source_ref="remote", recheck_after_seconds=10)
    memory.record_verification(
        refs[0], result=VerificationResult(VerificationOutcome.CONFIRMED, {"checked": True}, "v1")
    )
    assert inspect().wake_model
    assert memory.poll("watch").delivery_id is None
    clock.advance(seconds=11)
    expired = inspect()
    assert expired.status == "verification_due" and not expired.wake_model
    assert not expired.context.facts


def test_unknown_effect_is_reconciliation_not_an_inference_retry(
    recipe, sql_storage, clock, tmp_path
):
    _, memory, _, intent, _, inspect = setup(recipe, sql_storage, clock, tmp_path)
    lease = memory.claim(intent.id, worker_id="host")
    action = memory.prepare_action(
        intent.id, lease_token=lease.token, action_key="effect", request={}
    )
    memory.begin_action(action.id, lease_token=lease.token)
    memory.finish_action(action.id, lease_token=lease.token, outcome="unknown", result={})
    clock.advance(seconds=61)
    view = inspect()
    assert view.status == "reconcile" and not view.wake_model
    assert view.packet.open_actions[0].state == "unknown"


def test_missing_context_never_counts_as_action_ready(recipe, sql_storage, clock, tmp_path):
    kb, _, source, _, _, inspect = setup(recipe, sql_storage, clock, tmp_path)
    kb.forget(source.id)
    view = inspect()
    assert view.status == "missing_context" and not view.wake_model
    assert view.context.missing_requirements == ("backup receipt",)


def test_waiting_ticks_are_quiet_and_timer_needs_no_journal_event(
    recipe, sql_storage, clock, tmp_path
):
    _, memory, _, intent, _, inspect = setup(recipe, sql_storage, clock, tmp_path)
    lease = memory.claim(intent.id, worker_id="host")
    memory.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=memory.get_intent(intent.id).revision,
        state=IntentState.WAITING,
        progress={},
        waiting_for=WaitUntil(clock() + timedelta(seconds=10)),
    )
    assert all(inspect().status == "waiting" for _ in range(5))
    assert memory.poll("watch").delivery_id is None
    clock.advance(seconds=11)
    assert inspect().wake_model and memory.poll("watch").delivery_id is None


def test_debounce_returns_to_timer_without_spinning(recipe, sql_storage, clock, tmp_path):
    kb, memory, source, _, _, inspect = setup(recipe, sql_storage, clock, tmp_path)
    memory.unsubscribe("watch")
    memory.subscribe(subscription_id="watch", selector=WatchSelector("owner"), debounce_seconds=10)
    kb.replace_facts([replace(source, content="Backup receipt amended")])
    waiting = inspect()
    assert waiting.status == "debounced" and waiting.pages == 1 and not waiting.wake_model
    clock.advance(seconds=11)
    assert inspect().wake_model


def test_snapshot_overflow_never_acknowledges_partial_recovery(
    recipe, sql_storage, clock, tmp_path
):
    from ai_knot.always_on import CursorExpired

    kb, memory, source, _, inbox, inspect = setup(recipe, sql_storage, clock, tmp_path)
    memory.unsubscribe("watch")
    memory.subscribe(subscription_id="watch", selector=WatchSelector("owner"))
    inspect()
    saved = inbox.saved_refs("owner", "watch")
    kb.replace_facts([source, *[Fact(f"Noise {i}") for i in range(501)]])
    memory.prune_changes(force=True)
    view = inspect()
    assert view.status == "snapshot_too_large" and not view.wake_model
    assert inbox.saved_refs("owner", "watch") == saved
    with pytest.raises(CursorExpired):
        memory.poll("watch")


def test_failed_host_cache_commit_does_not_ack(recipe, sql_storage, clock, tmp_path, monkeypatch):
    kb, memory, source, _, inbox, inspect = setup(recipe, sql_storage, clock, tmp_path)
    kb.replace_facts([replace(source, content="Backup receipt updated")])
    pending = memory.poll("watch").delivery_id

    def fail(*args, **kwargs):
        raise OSError("Host cache unavailable")

    monkeypatch.setattr(inbox, "save", fail)
    with pytest.raises(OSError, match="Host cache unavailable"):
        inspect()
    assert memory.poll("watch").delivery_id == pending


@pytest.mark.parametrize("state", ["blocked", "completed", "cancelled", "deadline_exceeded"])
def test_non_ready_states_do_not_infer(recipe, sql_storage, clock, tmp_path, state):
    _, memory, _, intent, inbox, inspect = setup(recipe, sql_storage, clock, tmp_path)
    if state == "deadline_exceeded":
        expired = memory.create_intent(
            client_key="expired", goal="Expired", deadline=clock() - timedelta(seconds=1)
        )
        view = recipe.inspect_turn(
            memory,
            inbox,
            intent_id=expired.id,
            subscription_id="watch",
            requirements=("backup receipt",),
        )
    else:
        lease = memory.claim(intent.id, worker_id="host")
        memory.checkpoint(
            intent.id,
            lease_token=lease.token,
            expected_revision=memory.get_intent(intent.id).revision,
            state=IntentState(state),
            progress={},
        )
        view = inspect()
    assert view.status == state and not view.wake_model


def test_integrated_team_example(recipe, sql_storage, tmp_path):
    address = getattr(sql_storage, "_db_path", None) or sql_storage._dsn
    report = recipe.example(tmp_path, make_store=lambda: type(sql_storage)(address))
    assert report["quiet_ticks"] == 5 and report["idle_model_wakes"] == 0
    assert report["backlog_deferred"] and report["resyncs"] == 1
    assert report["stale_action_rejected"] and report["peer_experience_reused"]
    assert report["source_retraction"] and report["outcome_retraction"]
    assert report["completed"] and report["model_calls"] == 0


def test_diagnostics_do_not_include_context_or_work_secrets(recipe, sql_storage, clock, tmp_path):
    import json

    _, _, source, intent, _, inspect = setup(recipe, sql_storage, clock, tmp_path)
    encoded = json.dumps(inspect().diagnostics())
    assert source.content not in encoded and source.id not in encoded
    assert intent.id not in encoded and intent.goal not in encoded


def test_team_cycle_cli_is_executable(tmp_path):
    import json
    import os
    import subprocess

    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(root / "examples/always_on_team.py")],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True,
        text=True,
        timeout=20,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["completed"] and report["model_calls"] == 0


def test_manual_block_survives_source_changes(recipe, sql_storage, clock, tmp_path):
    kb, memory, source, intent, _, inspect = setup(recipe, sql_storage, clock, tmp_path)
    basis = memory.capture_evidence((("owner", source.id),))
    lease = memory.claim(intent.id, worker_id="host")
    memory.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=memory.get_intent(intent.id).revision,
        state=IntentState.BLOCKED,
        progress={"manual_review_required": True},
        basis=basis,
    )
    kb.replace_facts([replace(source, content="Backup receipt corrected")])
    view = inspect()
    assert view.context.complete and view.packet.stale_basis
    assert "host_blocked" in view.packet.reasons
    assert view.status == "blocked" and not view.wake_model
