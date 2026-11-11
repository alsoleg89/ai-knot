"""Continued coding stages: proposals never clear saved work before admission."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import asdict
from datetime import timedelta

from ai_knot.always_on import (
    EvidenceRef,
    SemanticBudget,
    SemanticContext,
    SnapshotTooLarge,
    StaleEvidence,
    WaitUntil,
)
from ai_knot.always_on.changes import evidence_revision
from ai_knot.always_on.codec import canonical_json
from ai_knot.always_on.context import task_applicable
from ai_knot.always_on.lineage import derivation_qualifiers
from ai_knot.types import Fact, MemoryType
from tests.eval.team_repos.protocol import (
    coder_messages,
    context_chars,
    parse_patch,
    patch_schema,
    safe_output,
)
from tests.eval.team_repos.runner import OWNER, moment

from .state import memory_for

ARMS = (
    "direct_sources",
    "current_docs_handoff",
    "raw_team_history",
    "lexical",
    "semantic",
    "cited_sources",
)
CODER_OUTPUT = 8192


def sha(value) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def render_entries(rows) -> list[dict]:
    ordered = sorted(
        rows,
        key=lambda row: (
            row.get("observed_at", ""),
            row["ref"]["namespace"],
            row["ref"]["fact_id"],
            row["ref"]["revision"],
        ),
    )
    return [{**row, "index": i} for i, row in enumerate(ordered, 1)]


def current_entries(storage, memory, case) -> list[dict]:
    view = memory.task_snapshot(task_key=case.task_key, max_facts=128, max_chars=65536)
    metadata = {(ns, f.id): f for ns in (OWNER, "__shared__") for f in storage.load(ns)}
    rows = []
    for item in view:
        fact = metadata[item.ref.namespace, item.ref.fact_id]
        if evidence_revision(fact) != item.ref.revision:
            raise StaleEvidence("Source changed during metadata projection")
        rows.append(
            {
                "index": 0,
                "content": item.content,
                "version": fact.version,
                "observed_at": fact.created_at.isoformat(),
                "ref": asdict(item.ref),
                "publisher": fact.origin_agent_id,
            }
        )
    return render_entries(rows)


def publish_handoff(storage, case, index, note) -> None:
    if note is None:
        return
    memory = memory_for(storage, index)
    refs = tuple(EvidenceRef(**r) for r in note["basis"])
    if not refs or any(ref.namespace != "__shared__" for ref in refs):
        raise StaleEvidence("Handoff roots unavailable")
    summary = note["summary"]
    if not isinstance(summary, str) or not summary.strip() or len(summary) > 1200:
        raise ValueError("Invalid handoff size")
    with storage.always_on_transaction((OWNER, "__shared__"), write=True) as session:
        if memory._validate_basis(session, refs, memory._now()):
            raise StaleEvidence("Handoff roots unavailable")
        roots = [session.load_facts("__shared__", fact_id=ref.fact_id)[0] for ref in refs]
        if any(
            f.origin_agent_id != "repo-observer" or not task_applicable(f, case.task_key)
            for f in roots
        ):
            raise StaleEvidence("Handoff roots unavailable")
        content = "Researcher handoff\n" + summary
        stamp = moment(index) + timedelta(microseconds=500000)
        fact = Fact(
            content,
            id=sha(["handoff", case.id])[:32],
            type=MemoryType.PROCEDURAL,
            qualifiers={**derivation_qualifiers("__shared__", roots), "task_key": case.task_key},
            version=index + 1,
            created_at=stamp,
            valid_from=stamp,
            last_accessed=stamp,
            event_time=stamp,
            slot_key=canonical_json(["handoff", case.id]),
            value_text=content,
            origin_agent_id="repo-researcher",
            visibility_scope="global",
            verification_source="aggregate",
        )
        session.replace_facts(
            "__shared__", [f for f in session.load_facts("__shared__") if f.id != fact.id] + [fact]
        )


def merge_history(previous, current):
    by_ref = {tuple(sorted(row["ref"].items())): row for row in previous}
    by_ref.update({tuple(sorted(row["ref"].items())): row for row in current})
    return render_entries(list(by_ref.values()))


async def acquire(storage, memory, case, arm, note, client, history):
    current = current_entries(storage, memory, case)
    by_ref = {EvidenceRef(**r["ref"]): r for r in current}
    required = tuple(EvidenceRef(**r) for r in note["basis"]) if arm == "cited_sources" else ()
    if arm == "direct_sources":
        selected = [r for r in current if r["publisher"] != "repo-researcher"]
    elif arm == "current_docs_handoff":
        selected = current
    elif arm == "raw_team_history":
        selected = merge_history(history, current)
    elif arm == "lexical":
        result = memory.select_context(
            case.requirements, task_key=case.task_key, max_facts=8, max_chars=2000
        )
        selected = [by_ref[f.ref] for f in result.facts]
    elif arm in ("semantic", "cited_sources"):
        result = await SemanticContext(memory, embed=client.embed).acquire(
            case.requirements,
            task_key=case.task_key,
            required_refs=required,
            max_facts=8,
            max_chars=2000,
            budget=SemanticBudget(
                max_candidates=128, max_provider_calls=1, max_input_bytes=65536, timeout=30
            ),
        )
        if result.status not in ("ok", "snapshot") or result.review.issues:
            raise ValueError("Semantic context unavailable")
        selected = [by_ref[f.ref] for f in result.review.accepted]
    else:
        raise ValueError("Unknown acquisition arm")
    selected = render_entries(selected)
    if not selected or context_chars(selected) > 24000:
        raise ValueError("Context unavailable or oversized")
    return selected, merge_history(history, current), required


def state(intent):
    return {
        "state": intent.state.value,
        "revision": intent.revision,
        "code_sha256": sha(intent.progress.get("code", "")),
        "basis": [asdict(r) for r in intent.basis],
        "lease_present": intent.lease is not None,
    }


async def run_stage(
    storage,
    case,
    index,
    arm,
    intent_id,
    note,
    client,
    sandbox,
    on_update=lambda row: None,
    observed_history=None,
):
    memory = memory_for(storage, index)
    report = {
        "stage": index,
        "arm": arm,
        "status": "running",
        "attempts": [],
        "tests_passed": False,
        "before_resume": state(memory.get_intent(intent_id)),
    }
    source_before = sha(sorted(storage.load("__shared__"), key=lambda f: f.id))
    try:
        packet = memory.resume(intent_id)
        report.update(
            after_resume=state(packet.intent), stale_basis=[r.reason for r in packet.stale_basis]
        )
        if packet.open_actions or {
            "unknown_action",
            "host_blocked",
            "deadline_exceeded",
            "live_lease",
        } & set(packet.reasons):
            report["status"] = "blocked"
            return report
        if packet.intent.state.value not in ("ready", "blocked"):
            report["status"] = "blocked"
            return report
        guard = memory.task_guard(task_key=case.task_key)
        report["guard_digest"] = guard.digest
        progress = dict(packet.intent.progress)
        selected, history, required = await acquire(
            storage,
            memory,
            case,
            arm,
            note,
            client,
            progress.get("history", []) if observed_history is None else observed_history,
        )
        if memory.task_guard(task_key=case.task_key) != guard:
            raise StaleEvidence("Source set changed")
        report.update(
            context=selected,
            context_sha256=sha(selected),
            rendered_context_chars=context_chars(selected),
            evidence_chars=sum(len(r["content"]) for r in selected),
            required_refs=[asdict(r) for r in required],
            observed_history=history,
        )
        code, receipt = progress["code"], progress.get("last_receipt")
        applied = False
        for attempt in range(2):
            expected = memory.get_intent(intent_id)
            raw = await client.complete(
                coder_messages(case, case.stages[index], code, selected, receipt),
                max_output_tokens=CODER_OUTPUT,
                response_schema=patch_schema(len(selected)),
            )
            safe, redacted = safe_output(raw)
            row = {"reader_output": safe, "status": "returned", "applied": False}
            report["attempts"].append(row)
            on_update(report)
            try:
                if getattr(client, "completion_truncated", False):
                    row["truncated"] = True
                    raise ValueError("Truncated completion")
                if redacted:
                    raise ValueError("Credential-shaped output")
                candidate, refs = parse_patch(raw, selected)
                if candidate is None:
                    row["status"] = "wait"
                    break
                refs = tuple(dict.fromkeys((*required, *refs)))
                if not memory.review_context(refs).complete:
                    raise StaleEvidence("Selected basis unavailable")
            except (ValueError, StaleEvidence) as exc:
                if memory.task_guard(task_key=case.task_key) != guard:
                    raise StaleEvidence("Source set changed") from None
                row.update(status="invalid_proposal", error_type=type(exc).__name__)
                receipt = {
                    "status": "error",
                    "tests_run": 0,
                    "failures": 0,
                    "errors": 1,
                    "failed_test_names": [],
                }
                on_update(report)
                continue
            if memory.task_guard(task_key=case.task_key) != guard:
                raise StaleEvidence("Source set changed")
            if memory.get_intent(intent_id) != expected:
                raise StaleEvidence("Intent changed during inference")
            lease = memory.claim(
                intent_id, worker_id=f"proposal-{index}-{attempt}", ttl_seconds=300
            )
            proposed = dict(progress, code=candidate, history=history)
            memory.checkpoint(
                intent_id,
                lease_token=lease.token,
                expected_revision=memory.get_intent(intent_id).revision,
                state="ready",
                progress=proposed,
                basis=refs,
                context_guard=guard,
            )
            lease = memory.claim(intent_id, worker_id=f"effect-{index}-{attempt}", ttl_seconds=300)
            action = memory.prepare_action(
                intent_id,
                lease_token=lease.token,
                action_key=f"{index}:{attempt}",
                request={"candidate_sha256": sha(candidate)},
                basis=refs,
            )
            memory.begin_action(action.id, lease_token=lease.token, context_guard=guard)
            code = candidate
            row["candidate_code"] = code
            receipt = await asyncio.to_thread(sandbox.run, code, case.stages[index].test_source)
            passed = receipt["status"] == "pass"
            memory.finish_action(
                action.id,
                lease_token=lease.token,
                outcome="succeeded" if passed else "failed",
                result=receipt,
            )
            progress = dict(proposed, last_receipt=receipt, stage=index)
            memory.checkpoint(
                intent_id,
                lease_token=lease.token,
                expected_revision=memory.get_intent(intent_id).revision,
                state="ready",
                progress=progress,
                basis=refs,
                context_guard=guard,
            )
            row.update(status="tested", applied=True, candidate_sha256=sha(code), receipt=receipt)
            applied = True
            report["tests_passed"] = passed
            if passed:
                break
            on_update(report)
        if applied:
            lease = memory.claim(intent_id, worker_id=f"schedule-{index}", ttl_seconds=300)
            memory.checkpoint(
                intent_id,
                lease_token=lease.token,
                expected_revision=memory.get_intent(intent_id).revision,
                state="waiting"
                if index < 2
                else "completed"
                if report["tests_passed"]
                else "blocked",
                waiting_for=WaitUntil(moment(index + 1) + timedelta(seconds=1))
                if index < 2
                else None,
                progress=progress,
                basis=memory.get_intent(intent_id).basis,
                context_guard=guard,
            )
        report["status"] = "complete"
    except (StaleEvidence, SnapshotTooLarge) as exc:
        report.update(
            status="context_changed" if isinstance(exc, StaleEvidence) else "context_overflow",
            error_type=type(exc).__name__,
        )
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
    finally:
        report.update(
            after=state(memory.get_intent(intent_id)),
            source_unchanged=source_before
            == sha(sorted(storage.load("__shared__"), key=lambda f: f.id)),
        )
        on_update(report)
    return report
