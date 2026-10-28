"""Continued code work; generated candidates execute only in the Docker port."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import (
    AlwaysOnMemory,
    EvidenceRef,
    SemanticBudget,
    SemanticContext,
    SnapshotTooLarge,
    StaleEvidence,
    WaitUntil,
)
from ai_knot.always_on.changes import evidence_revision
from ai_knot.always_on.codec import canonical_json
from ai_knot.always_on.lineage import derivation_qualifiers
from ai_knot.types import Fact, MemoryType

from .protocol import (
    CODER_OUTPUT,
    MAX_CONTEXT_CHARS,
    coder_messages,
    context_chars,
    parse_patch,
    patch_schema,
    safe_output,
)

NOW = datetime(2026, 10, 28, 12, tzinfo=UTC)
OWNER = "repo-executor"


def moment(index):
    return NOW + timedelta(hours=index)


def identity(case_id, path):
    return hashlib.sha256(f"{case_id}:{path}".encode()).hexdigest()[:32]


def projected_documents(case, index):
    previous = {}
    for number, stage in enumerate(case.stages[: index + 1]):
        current = {}
        for doc in stage.documents:
            key = identity(case.id, doc.path)
            text = doc.path + "\n" + doc.text
            qualifiers = {"task_key": doc.task_key} if doc.task_key else {}
            old = previous.get(key)
            if old is not None and old.content == text and old.qualifiers == qualifiers:
                current[key] = old
                continue
            current[key] = Fact(
                text,
                id=key,
                type=MemoryType.SEMANTIC,
                qualifiers=qualifiers,
                created_at=moment(number),
                valid_from=moment(number),
                event_time=moment(number),
                version=1 if old is None else old.version + 1,
                slot_key=case.id + "::" + doc.path,
                value_text=text,
                origin_agent_id="repo-observer",
                visibility_scope="global",
            )
        previous = current
    return tuple(previous.values())


def memory_for(storage, index):
    return AlwaysOnMemory(
        KnowledgeBase(OWNER, storage=storage, embed_url=""),
        pool=SharedMemoryPool(
            storage, persist_stats=True, embed_url="", clock=lambda: moment(index)
        ),
        clock=lambda: moment(index),
    )


def entries(facts):
    return [
        {
            "index": i,
            "content": f.content,
            "version": f.version,
            "observed_at": f.created_at.isoformat(),
            "ref": asdict(EvidenceRef("__shared__", f.id, evidence_revision(f))),
        }
        for i, f in enumerate(facts, 1)
    ]


def seed_stage(storage, case, index, note):
    facts = projected_documents(case, index)
    if note is not None:
        by_id = {f.id: f for f in facts}
        refs = tuple(EvidenceRef(**item) for item in note["basis"])
        for ref in refs:
            if (
                ref.namespace != "__shared__"
                or ref.fact_id not in by_id
                or evidence_revision(by_id[ref.fact_id]) != ref.revision
            ):
                raise ValueError("Frozen handoff source changed")
        summary = "handoff.md\n" + note["summary"]
        handoff = Fact(
            summary,
            id=identity(case.id, "handoff"),
            type=MemoryType.PROCEDURAL,
            qualifiers={
                **derivation_qualifiers("__shared__", [by_id[r.fact_id] for r in refs]),
                "task_key": case.task_key,
                "for_role": "coder",
            },
            created_at=moment(index),
            valid_from=moment(index),
            event_time=moment(index),
            version=index + 1,
            slot_key=case.id + "::handoff",
            value_text=summary,
            origin_agent_id="repo-researcher",
            visibility_scope="global",
            verification_source="aggregate",
        )
        facts = (*facts, handoff)
    storage.save("__shared__", list(facts))


async def context_for(storage, memory, case, index, arm, client, history):
    facts = [
        f
        for f in storage.load("__shared__")
        if f.qualifiers.get("task_key", "") in ("", case.task_key)
    ]
    current_entries = {EvidenceRef(**e["ref"]): e for e in entries(facts)}
    if arm == "direct_sources":
        selected = entries([f for f in facts if f.origin_agent_id != "repo-researcher"])
        mode = "current_sources"
    elif arm == "current_docs_handoff":
        selected, mode = entries(facts), "current_sources_and_latest_handoff"
    elif arm == "raw_team_history":
        selected = [{**item, "index": i} for i, item in enumerate(history, 1)]
        mode = "all_history"
    elif arm == "task_memory":
        try:
            view = memory.task_snapshot(
                task_key=case.task_key, max_facts=8, max_chars=MAX_CONTEXT_CHARS
            )
            selected = [{**current_entries[f.ref], "index": i} for i, f in enumerate(view, 1)]
            mode = "snapshot"
        except SnapshotTooLarge:
            result = await SemanticContext(memory, embed=client.embed).acquire(
                case.requirements,
                budget=SemanticBudget(
                    max_candidates=128, max_provider_calls=1, max_input_bytes=65536, timeout=30
                ),
                task_key=case.task_key,
                role="coder",
                max_facts=8,
                max_chars=MAX_CONTEXT_CHARS,
            )
            if result.status not in ("snapshot", "ok"):
                raise ValueError("Semantic acquisition failed") from None
            selected = [
                {**current_entries[f.ref], "index": i}
                for i, f in enumerate(result.review.accepted, 1)
            ]
            mode = "semantic"
    else:
        raise ValueError("Unknown acquisition policy")
    if context_chars(selected) > MAX_CONTEXT_CHARS:
        raise ValueError("Full context exceeds the frozen ceiling")
    return selected, mode


async def run_stage(storage, case, index, arm, intent_id, client, sandbox):
    memory = memory_for(storage, index)
    result = {
        "stage": index,
        "arm": arm,
        "status": "running",
        "attempts": [],
        "tests_passed": False,
        "source_unchanged": None,
    }
    before = canonical_json(storage.load("__shared__"))
    try:
        packet = memory.resume(intent_id)
        if {"unknown_action", "host_blocked", "deadline_exceeded"} & set(packet.reasons):
            raise ValueError("Work is blocked")
        lease = memory.claim(intent_id, worker_id=f"coder-{index}", ttl_seconds=300)
        progress = dict(memory.get_intent(intent_id).progress)
        if packet.stale_basis:
            memory.checkpoint(
                intent_id,
                lease_token=lease.token,
                expected_revision=memory.get_intent(intent_id).revision,
                state="ready",
                progress=progress,
                basis=(),
            )
            lease = memory.claim(intent_id, worker_id=f"coder-{index}", ttl_seconds=300)
        selected, mode = await context_for(
            storage, memory, case, index, arm, client, progress.get("history", [])
        )
        result.update(
            context=selected, acquisition=mode, rendered_context_chars=context_chars(selected)
        )
        code = progress["code"]
        receipt = progress.get("last_receipt")
        last_basis = ()
        for attempt in range(2):
            raw = await client.complete(
                coder_messages(case, case.stages[index], code, selected, receipt),
                max_output_tokens=CODER_OUTPUT,
                response_schema=patch_schema(len(selected)),
            )
            safe, redacted = safe_output(raw)
            row = {"reader_output": safe, "basis_valid": False, "applied": False}
            result["attempts"].append(row)
            try:
                if redacted:
                    raise ValueError("Credential-shaped output rejected")
                candidate, refs = parse_patch(raw, selected)
                if candidate is None:
                    row["status"] = "wait"
                    break
                row["candidate_code"] = candidate
                review = memory.review_context(refs)
                if not review.complete:
                    raise ValueError("Proposed basis is no longer current")
                row["basis_valid"] = True
                action = memory.prepare_action(
                    intent_id,
                    lease_token=lease.token,
                    action_key=f"stage:{index}:attempt:{attempt}",
                    request={"candidate_sha256": hashlib.sha256(candidate.encode()).hexdigest()},
                    basis=refs,
                )
                memory.begin_action(action.id, lease_token=lease.token)
                code = candidate
                receipt = await asyncio.to_thread(sandbox.run, code, case.stages[index].test_source)
                passed = receipt["status"] == "pass"
                memory.finish_action(
                    action.id,
                    lease_token=lease.token,
                    outcome="succeeded" if passed else "failed",
                    result=receipt,
                )
                last_basis = refs
                row.update(
                    status="tested",
                    applied=True,
                    receipt=receipt,
                    candidate_sha256=hashlib.sha256(code.encode()).hexdigest(),
                )
                result["tests_passed"] = passed
                if passed:
                    break
            except (ValueError, StaleEvidence) as exc:
                row.update(status="invalid_proposal", error_type=type(exc).__name__)
                receipt = {
                    "status": "invalid_proposal",
                    "tests_run": 0,
                    "failures": 0,
                    "errors": 1,
                    "failed_test_names": [],
                }
        progress.update(code=code, last_receipt=receipt, stage=index)
        waiting = index + 1 < len(case.stages)
        memory.checkpoint(
            intent_id,
            lease_token=lease.token,
            expected_revision=memory.get_intent(intent_id).revision,
            state="waiting" if waiting else "completed" if result["tests_passed"] else "blocked",
            waiting_for=WaitUntil(moment(index + 1)) if waiting else None,
            progress=progress,
            basis=last_basis,
        )
        result["source_unchanged"] = before == canonical_json(storage.load("__shared__"))
        if not result["source_unchanged"]:
            raise ValueError("Selection changed source state")
        result.update(status="complete", code_sha256=hashlib.sha256(code.encode()).hexdigest())
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__)
    finally:
        result.update(calls=client.calls, spent_tokens=client.spent_tokens)
    return result
