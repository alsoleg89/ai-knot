"""Bounded maintenance with durable reservations and revision-guarded commits."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, fields
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Literal, Protocol, cast
from uuid import uuid4

from ai_knot.types import Fact, MemoryType, MESIState

from . import _state
from .changes import evidence_revision
from .codec import canonical_json, decode_value
from .models import (
    EvidenceRef,
    FreshnessRecord,
    IdempotencyConflict,
    RevisionConflict,
    VerificationOutcome,
    VerificationRequest,
    VerificationResult,
    nonempty,
)
from .state_storage import save_freshness, timestamp
from .storage import AlwaysOnSession, SQLRow, SQLStatement, sql

if TYPE_CHECKING:
    from .memory import AlwaysOnMemory


@dataclass(frozen=True)
class MaintenanceBudget:
    max_items: int = 100
    max_model_calls: int = 0
    max_input_bytes: int = 65536
    max_wall_seconds: float = 1.0

    def __post_init__(self) -> None:
        for name in ("max_items", "max_model_calls", "max_input_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "max_model_calls" else 1):
                raise ValueError("Invalid maintenance budget.")
        if (
            type(self.max_wall_seconds) not in (int, float)
            or not math.isfinite(self.max_wall_seconds)
            or self.max_wall_seconds <= 0
        ):
            raise ValueError("Invalid wall-time budget.")


@dataclass(frozen=True)
class MaintenanceReport:
    run_id: str
    status: Literal["completed", "budget_exhausted", "conflict", "failed"]
    inspected: int = 0
    verified: int = 0
    consolidated: int = 0
    expired: int = 0
    model_calls_reserved: int = 0
    skipped: int = 0
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConsolidationProposal:
    facts: tuple[Fact, ...]
    covered_episode_ids: tuple[str, ...]
    input_refs: tuple[EvidenceRef, ...]


class VerificationProvider(Protocol):
    config_key: str
    model_cost: int

    async def verify(self, request: VerificationRequest) -> VerificationResult: ...


class ConsolidationProvider(Protocol):
    config_key: str
    model_cost: int

    async def propose(self, episodes: tuple[Fact, ...]) -> ConsolidationProposal: ...


_DDL = """CREATE TABLE IF NOT EXISTS ao_maintenance_runs (
    owner TEXT NOT NULL, run_key TEXT NOT NULL, run_id TEXT NOT NULL, digest TEXT NOT NULL,
    generation BIGINT NOT NULL, expires_at TEXT NOT NULL, reserved BIGINT NOT NULL DEFAULT 0,
    input_bytes BIGINT NOT NULL DEFAULT 0, verified BIGINT NOT NULL DEFAULT 0,
    consolidated BIGINT NOT NULL DEFAULT 0, expired BIGINT NOT NULL DEFAULT 0,
    report_json TEXT, diagnostic_json TEXT, PRIMARY KEY(owner,run_key))"""


def initialize_maintenance(session: AlwaysOnSession) -> None:
    pair = sql(_DDL)
    session.execute(
        SQLStatement(
            pair.sqlite, pair.postgres.replace("expires_at TEXT", "expires_at TIMESTAMPTZ")
        )
    )
    session.execute(
        sql(
            "CREATE INDEX IF NOT EXISTS ao_maintenance_expiry ON "
            "ao_maintenance_runs(owner,expires_at)"
        )
    )
    session.execute(sql("UPDATE ao_schema SET version=6 WHERE id=1 AND version<6"))


def _row(memory: AlwaysOnMemory, session: AlwaysOnSession, key: str) -> SQLRow:
    row = session.fetch_one(
        sql("SELECT * FROM ao_maintenance_runs WHERE owner=? AND run_key=?"), (memory._owner, key)
    )
    if row is None:
        raise RevisionConflict("maintenance run missing")
    return row


def _guard(memory: AlwaysOnMemory, session: AlwaysOnSession, key: str, generation: int) -> SQLRow:
    row = _row(memory, session, key)
    if (
        int(row["generation"] or 0) != generation
        or str(row["expires_at"]) <= str(timestamp(memory._now()))
        or row["report_json"] is not None
    ):
        raise RevisionConflict("maintenance generation expired")
    return row


def _begin(
    memory: AlwaysOnMemory, key: str, digest: str, budget: MaintenanceBudget
) -> tuple[int, str, MaintenanceReport | None]:
    with memory._transaction() as session:
        initialize_maintenance(session)
        row = session.fetch_one(
            sql("SELECT * FROM ao_maintenance_runs WHERE owner=? AND run_key=?"),
            (memory._owner, key),
        )
        if row is not None:
            if row["digest"] != digest:
                raise IdempotencyConflict("maintenance key reused")
            if row["report_json"]:
                return (
                    int(row["generation"] or 0),
                    str(row["run_id"]),
                    cast(
                        MaintenanceReport,
                        decode_value(json.loads(str(row["report_json"])), MaintenanceReport),
                    ),
                )
            if str(row["expires_at"]) > str(timestamp(memory._now())):
                raise RevisionConflict("maintenance run in progress")
        generation = int(row["generation"] or 0) + 1 if row else 1
        run_id = str(row["run_id"]) if row else uuid4().hex
        expires = timestamp(memory._now() + timedelta(seconds=max(60, budget.max_wall_seconds)))
        if row:
            session.execute(
                sql(
                    "UPDATE ao_maintenance_runs SET generation=?,expires_at=? WHERE owner=? "
                    "AND run_key=?"
                ),
                (generation, expires, memory._owner, key),
            )
        else:
            session.execute(
                sql(
                    "INSERT INTO "
                    "ao_maintenance_runs(owner,run_key,run_id,digest,generation,expires_at) "
                    "VALUES(?,?,?,?,?,?)"
                ),
                (memory._owner, key, run_id, digest, generation, expires),
            )
        return generation, run_id, None


class _BudgetExhausted(Exception):
    pass


def _reserve(
    memory: AlwaysOnMemory,
    key: str,
    generation: int,
    budget: MaintenanceBudget,
    cost: int,
    input_bytes: int,
) -> None:
    with memory._transaction() as session:
        row = _guard(memory, session, key, generation)
        if int(row["reserved"] or 0) + cost > budget.max_model_calls:
            raise _BudgetExhausted("model_budget")
        if int(row["input_bytes"] or 0) + input_bytes > budget.max_input_bytes:
            raise _BudgetExhausted("input_budget")
        session.execute(
            sql(
                "UPDATE ao_maintenance_runs SET "
                "reserved=reserved+?,input_bytes=input_bytes+? WHERE owner=? AND "
                "run_key=? AND generation=?"
            ),
            (cost, input_bytes, memory._owner, key, generation),
        )


def _fact_from_sql(row: SQLRow) -> Fact:
    payload: dict[str, object] = {
        field.name: row[field.name] for field in fields(Fact) if field.name in row
    }
    payload["retention_score"] = row["retention"]
    for key in ("tags", "source_snippets", "source_spans", "access_intervals", "qualifiers"):
        payload[key] = json.loads(str(row[key]))
    for key in ("created_at", "last_accessed", "valid_from", "valid_until", "event_time"):
        payload[key] = datetime.fromisoformat(str(row[key])) if row[key] else None
    payload["supported"] = bool(row["supported"])
    payload["type"] = MemoryType(str(row["type"]))
    payload["mesi_state"] = MESIState(str(row["mesi_state"]))
    return Fact(**payload)  # type: ignore[arg-type]


def select_episodes(
    memory: AlwaysOnMemory, limit: int, *, older_than_hours: float = 24
) -> tuple[Fact, ...]:
    if limit <= 0:
        return ()
    now = timestamp(memory._now())
    cutoff = timestamp(memory._now() - timedelta(hours=older_than_hours))
    statement = """SELECT * FROM facts WHERE agent_id=? AND type='episodic' AND supported=1
        AND created_at<=? AND valid_from<=? AND (valid_until IS NULL OR valid_until>?)
        ORDER BY created_at,id LIMIT ?"""
    pair = SQLStatement(
        statement,
        statement.replace("FROM facts", 'FROM "ai-knot_facts"')
        .replace("supported=1", "supported=1")
        .replace("?", "%s"),
    )
    with memory._transaction(write=False) as session:
        return tuple(
            _fact_from_sql(row)
            for row in session.fetch_all(pair, (memory._owner, cutoff, now, now, limit))
        )


def validate_proposal(
    proposal: ConsolidationProposal,
    selected: tuple[EvidenceRef, ...],
    *,
    now: datetime | None = None,
) -> tuple[tuple[Fact, ...], tuple[str, ...]]:
    if not isinstance(proposal, ConsolidationProposal):
        raise ValueError("typed consolidation proposal required")
    if not proposal.facts:
        return (), ()
    covered = proposal.covered_episode_ids
    if (
        not covered
        or len(set(covered)) != len(covered)
        or set(proposal.input_refs) != set(selected)
        or len(set(proposal.input_refs)) != len(proposal.input_refs)
    ):
        raise ValueError("invalid consolidation input mapping")
    selected_ids = {ref.fact_id for ref in selected}
    if not set(covered) <= selected_ids or len({ref.fact_id for ref in selected}) != len(selected):
        raise ValueError("unknown consolidation source")
    outputs = copy.deepcopy(proposal.facts)
    now = now or datetime.now(UTC)
    mapped: set[str] = set()
    if len({fact.id for fact in outputs}) != len(outputs):
        raise ValueError("duplicate output IDs")
    for fact in outputs:
        for field_name in (
            "created_at",
            "last_accessed",
            "valid_from",
            "valid_until",
            "event_time",
        ):
            value = getattr(fact, field_name)
            if value is not None and (
                not isinstance(value, datetime)
                or value.tzinfo is None
                or value.utcoffset() != timedelta(0)
            ):
                raise ValueError("invalid consolidation output datetime")
        if (
            not fact.is_active(now)
            or fact.created_at > now
            or (fact.event_time is not None and fact.event_time > now)
        ):
            raise ValueError("consolidation output is inactive or future-dated")
        sources = json.loads(fact.qualifiers.get("consolidation_ids", "[]"))
        if (
            not isinstance(sources, list)
            or not sources
            or any(not isinstance(source, str) or not source for source in sources)
            or len(sources) != len(set(sources))
            or not set(sources) <= set(covered)
        ):
            raise ValueError("invalid consolidation lineage")
        if fact.type != MemoryType.SEMANTIC or not fact.content.strip() or not fact.supported:
            raise ValueError("invalid semantic output")
        if len(canonical_json(fact).encode()) > 65536:
            raise ValueError("output fact exceeds 65536 bytes")
        mapped.update(sources)
    if mapped != set(covered):
        raise ValueError("unrepresented consolidation source")
    return outputs, covered


def commit_consolidation(
    memory: AlwaysOnMemory,
    proposal: ConsolidationProposal,
    selected: tuple[EvidenceRef, ...],
    *,
    run_key: str | None = None,
    generation: int | None = None,
    anchor_seq: int | None = None,
) -> int:
    """Validate complete lineage before one atomic fact/journal/ledger mutation."""
    outputs, covered = validate_proposal(proposal, selected, now=memory._now())
    if not outputs:
        return 0
    with memory._transaction(tuple(sorted({ref.namespace for ref in selected}))) as session:
        now = memory._now()
        if any(not output.is_active(now) for output in outputs):
            raise ValueError("consolidation output expired before commit")
        if run_key is not None:
            assert generation is not None
            _guard(memory, session, run_key, generation)
        for ref in selected:
            source = memory._fact_for_ref(session, ref, now)
            if source.type != MemoryType.EPISODIC or ref.namespace != memory._owner:
                raise ValueError("invalid consolidation source type")
        current = session.load_facts(memory._owner)
        if {fact.id for fact in outputs} & {fact.id for fact in current}:
            raise RevisionConflict("output ID exists")
        if anchor_seq is not None and any(output.slot_key for output in outputs):
            stream = session.fetch_one(
                sql("SELECT min_retained_seq FROM ao_streams WHERE namespace=?"), (memory._owner,)
            )
            if stream is None or int(stream["min_retained_seq"] or 0) > anchor_seq + 1:
                raise RevisionConflict("consolidation conflict history was pruned")
        for output in outputs:
            if output.slot_key:
                if anchor_seq is None:
                    if any(
                        target.slot_key == output.slot_key and target.is_active(now)
                        for target in current
                    ):
                        raise RevisionConflict("slot target was not captured before preparation")
                elif (
                    session.fetch_one(
                        sql(
                            "SELECT seq FROM ao_changes WHERE namespace=? AND seq>? "
                            "AND (slot_key=? "
                            "OR before_slot_key=?) LIMIT 1"
                        ),
                        (memory._owner, anchor_seq, output.slot_key, output.slot_key),
                    )
                    is not None
                ):
                    raise RevisionConflict("slot changed during preparation")
        # Current targets, rather than preparation snapshots, determine slot closure.
        for output in outputs:
            output.created_at = output.valid_from = now
            for target in current:
                if output.slot_key and target.slot_key == output.slot_key and target.is_active(now):
                    target.valid_until = now
            output.qualifiers = dict(output.qualifiers)
        for fact in current:
            if fact.id in covered:
                fact.valid_until = now
        session.replace_facts(memory._owner, [*current, *outputs])
        if run_key is not None:
            session.execute(
                sql(
                    "UPDATE ao_maintenance_runs SET consolidated=consolidated+? WHERE "
                    "owner=? AND run_key=? AND generation=?"
                ),
                (len(covered), memory._owner, run_key, generation),
            )
        return len(covered)


def _verify(
    memory: AlwaysOnMemory,
    key: str,
    generation: int,
    request: VerificationRequest,
    result: VerificationResult,
    expected_policy: FreshnessRecord,
    expected_revision: int,
) -> bool:
    with memory._transaction((request.ref.namespace,)) as session:
        _guard(memory, session, key, generation)
        now = memory._now()
        fact = memory._fact_for_ref(session, request.ref, now)
        policy = memory._policy(session, request.ref.namespace, fact)
        revision_row = session.fetch_one(
            sql(
                "SELECT revision FROM ao_freshness WHERE namespace=? AND subject_kind=? "
                "AND subject_key=?"
            ),
            (request.ref.namespace, expected_policy.subject_kind, expected_policy.subject_key),
        )
        if (
            policy is None
            or policy != expected_policy
            or revision_row is None
            or int(revision_row["revision"] or 0) != expected_revision
        ):
            raise RevisionConflict("freshness policy changed during verification")
        current = _state.record_verification(
            policy, request.ref, current=request.ref, result=result, now=now
        )
        save_freshness(session, current)
        accepted = result.outcome == VerificationOutcome.CONFIRMED
        if accepted:
            session.execute(
                sql(
                    "UPDATE ao_maintenance_runs SET verified=verified+1 WHERE owner=? AND "
                    "run_key=? AND generation=?"
                ),
                (memory._owner, key, generation),
            )
        return bool(accepted)


def _cleanup(memory: AlwaysOnMemory, key: str, generation: int, limit: int) -> tuple[int, int]:
    if limit <= 0:
        return 0, 0
    namespaces = [memory._owner]
    if memory._pool is not None and memory._pool._persist_stats:
        memory._check_namespace("__shared__")
        namespaces.append("__shared__")
    inspected = pruned = 0
    with memory._transaction(tuple(namespaces)) as session:
        _guard(memory, session, key, generation)
        now = memory._now()
        if "__shared__" in namespaces:
            assert memory._pool is not None
            cutoff = timestamp(now - timedelta(seconds=memory._pool._POOL_TTL_SECONDS))
            statement = (
                "SELECT id FROM facts WHERE agent_id=? AND memory_tier='pool' AND "
                "last_accessed<? AND (valid_until IS NULL OR valid_until>?) ORDER BY "
                "last_accessed,id LIMIT ?"
            )
            pair = SQLStatement(
                statement,
                statement.replace("FROM facts", 'FROM "ai-knot_facts"').replace("?", "%s"),
            )
            ids = {
                str(row["id"])
                for row in session.fetch_all(pair, ("__shared__", cutoff, timestamp(now), limit))
            }
            if ids:
                facts = session.load_facts("__shared__")
                for fact in facts:
                    if fact.id in ids:
                        fact.valid_until = now
                session.replace_facts("__shared__", facts)
                inspected += len(ids)
                session.execute(
                    sql(
                        "UPDATE ao_maintenance_runs SET expired=expired+? WHERE owner=? AND "
                        "run_key=? AND generation=?"
                    ),
                    (len(ids), memory._owner, key, generation),
                )
        for namespace in namespaces:
            if inspected >= limit:
                break
            stream = session.fetch_one(
                sql("SELECT head_seq FROM ao_streams WHERE namespace=?"), (namespace,)
            )
            if stream is None:
                continue
            frontier = int(stream["head_seq"] or 0)
            row = session.fetch_one(
                sql("SELECT MIN(cursor) AS cursor FROM ao_subscriptions WHERE namespace=?"),
                (namespace,),
            )
            if row is not None and row["cursor"] is not None:
                frontier = min(frontier, int(row["cursor"]))
            rows = session.fetch_all(
                sql("SELECT seq FROM ao_changes WHERE namespace=? AND seq<=? ORDER BY seq LIMIT ?"),
                (namespace, frontier, limit - inspected),
            )
            if not rows:
                continue
            through = int(rows[-1]["seq"] or 0)
            pruned += session.execute(
                sql("DELETE FROM ao_changes WHERE namespace=? AND seq<=?"), (namespace, through)
            )
            session.execute(
                sql("UPDATE ao_streams SET min_retained_seq=? WHERE namespace=?"),
                (through + 1, namespace),
            )
            inspected += len(rows)
    return inspected, pruned


async def run_maintenance(
    memory: AlwaysOnMemory,
    *,
    run_key: str,
    budget: MaintenanceBudget,
    verifier: VerificationProvider | None = None,
    consolidator: ConsolidationProvider | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> MaintenanceReport:
    nonempty(run_key, "run_key")
    if not isinstance(budget, MaintenanceBudget):
        raise ValueError("typed maintenance budget required")
    providers = (verifier, consolidator)
    for provider in providers:
        if provider is not None:
            nonempty(provider.config_key, "config_key")
            if type(provider.model_cost) is not int or provider.model_cost < 0:
                raise ValueError("provider model_cost must be a nonnegative integer")
    digest = hashlib.sha256(
        canonical_json(
            [budget, [[p.config_key, p.model_cost] if p else None for p in providers]]
        ).encode()
    ).hexdigest()
    generation, run_id, cached = _begin(memory, run_key, digest, budget)
    if cached is not None:
        return cached
    deadline = monotonic() + budget.max_wall_seconds
    inspected = skipped = 0
    status: Literal["completed", "budget_exhausted", "conflict", "failed"] = "completed"
    reasons: list[str] = []
    cancelled = False

    def remaining() -> float:
        seconds = deadline - monotonic()
        if seconds <= 0:
            raise _BudgetExhausted("wall_budget")
        return seconds

    async def verify_one(
        provider: VerificationProvider, request: VerificationRequest
    ) -> VerificationResult:
        seconds = remaining()
        _reserve(
            memory,
            run_key,
            generation,
            budget,
            provider.model_cost,
            len(canonical_json(request).encode()),
        )
        return await asyncio.wait_for(provider.verify(request), timeout=seconds)

    try:
        for request in memory.due_verifications(limit=min(500, budget.max_items)):
            remaining()
            inspected += 1
            if verifier is None:
                skipped += 1
                reasons.append("verifier_disabled")
                continue
            with memory._transaction((request.ref.namespace,), write=False) as snapshot:
                fact = memory._fact_for_ref(snapshot, request.ref, memory._now())
                expected_policy = memory._policy(snapshot, request.ref.namespace, fact)
                if expected_policy is None or expected_policy.source_ref != request.source_ref:
                    raise RevisionConflict("freshness policy changed before verification")
                revision_row = snapshot.fetch_one(
                    sql(
                        "SELECT revision FROM ao_freshness WHERE namespace=? "
                        "AND subject_kind=? AND subject_key=?"
                    ),
                    (
                        request.ref.namespace,
                        expected_policy.subject_kind,
                        expected_policy.subject_key,
                    ),
                )
                if revision_row is None:
                    raise RevisionConflict("freshness policy disappeared before verification")
                expected_revision = int(revision_row["revision"] or 0)
            try:
                result = await verify_one(verifier, request)
            except asyncio.CancelledError:
                with suppress(Exception):
                    _verify(
                        memory,
                        run_key,
                        generation,
                        request,
                        VerificationResult(
                            VerificationOutcome.UNAVAILABLE, {"reason": "cancelled"}, ""
                        ),
                        expected_policy,
                        expected_revision,
                    )
                raise
            except (_BudgetExhausted, RevisionConflict):
                raise
            except Exception as exc:
                reasons.append("source_unavailable:" + type(exc).__name__)
                result = VerificationResult(
                    VerificationOutcome.UNAVAILABLE, {"reason": type(exc).__name__}, ""
                )
            _verify(
                memory, run_key, generation, request, result, expected_policy, expected_revision
            )
        remaining()
        episodes = select_episodes(memory, budget.max_items - inspected)
        inspected += len(episodes)
        if episodes and consolidator is None:
            skipped += len(episodes)
            reasons.append("consolidator_disabled")
        elif episodes and consolidator is not None:
            seconds = remaining()
            with memory._transaction(write=False) as snapshot:
                stream = snapshot.fetch_one(
                    sql("SELECT head_seq FROM ao_streams WHERE namespace=?"), (memory._owner,)
                )
                anchor = int(stream["head_seq"] or 0) if stream else 0
            refs = tuple(
                EvidenceRef(memory._owner, fact.id, evidence_revision(fact)) for fact in episodes
            )
            _reserve(
                memory,
                run_key,
                generation,
                budget,
                consolidator.model_cost,
                len(canonical_json(episodes).encode()),
            )
            proposal = await asyncio.wait_for(
                consolidator.propose(copy.deepcopy(episodes)), timeout=seconds
            )
            try:
                commit_consolidation(
                    memory,
                    proposal,
                    refs,
                    run_key=run_key,
                    generation=generation,
                    anchor_seq=anchor,
                )
            except (ValueError, RevisionConflict) as exc:
                diagnostic = {"reason": type(exc).__name__}
                with suppress(RevisionConflict), memory._transaction() as session:
                    _guard(memory, session, run_key, generation)
                    session.execute(
                        sql(
                            "UPDATE ao_maintenance_runs SET diagnostic_json=? "
                            "WHERE owner=? AND run_key=? AND generation=?"
                        ),
                        (canonical_json(diagnostic), memory._owner, run_key, generation),
                    )
                raise
            if not proposal.facts:
                reasons.append("empty_proposal")
        remaining()
        cleaned, pruned = _cleanup(memory, run_key, generation, budget.max_items - inspected)
        inspected += cleaned
        if pruned:
            reasons.append("journal_pruned")
        if inspected >= budget.max_items:
            status = "budget_exhausted"
            reasons.append("item_limit_reached")
    except _BudgetExhausted as exc:
        status = "budget_exhausted"
        reasons.append(str(exc))
    except RevisionConflict:
        status = "conflict"
        reasons.append("revision_conflict")
    except asyncio.CancelledError:
        cancelled = True
        status = "failed"
        reasons.append("cancelled")
        raise
    except Exception as exc:
        status = "failed"
        reasons.append(type(exc).__name__)
    finally:
        try:
            with memory._transaction() as session:
                row = _row(memory, session, run_key)
                report = MaintenanceReport(
                    run_id,
                    status,
                    inspected,
                    int(row["verified"] or 0),
                    int(row["consolidated"] or 0),
                    int(row["expired"] or 0),
                    int(row["reserved"] or 0),
                    skipped,
                    tuple(dict.fromkeys(reasons)),
                )
                if (
                    int(row["generation"] or 0) == generation
                    and str(row["expires_at"]) > str(timestamp(memory._now()))
                    and row["report_json"] is None
                ):
                    session.execute(
                        sql(
                            "UPDATE ao_maintenance_runs SET report_json=? WHERE owner=? AND "
                            "run_key=? AND generation=?"
                        ),
                        (canonical_json(report), memory._owner, run_key, generation),
                    )
        except Exception:
            if not cancelled:
                raise
    return report
