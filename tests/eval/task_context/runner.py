"""Equal-budget, offline SQL selection comparison with real process handoff.

Run ``python -m tests.eval.task_context.runner --backend sqlite --output-dir PATH``.
PostgreSQL requires AI_KNOT_TEST_PG_DSN and uses a newly owned temporary schema.
No model or embedding endpoint is invoked. Latency is descriptive, never a gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch
from urllib.parse import quote
from uuid import uuid4

from ai_knot.always_on import AlwaysOnMemory, EvidenceRef
from ai_knot.always_on.changes import evidence_revision
from ai_knot.always_on.codec import canonical_json, decode_value, encode_always_on
from ai_knot.always_on.models import ContextFact
from ai_knot.knowledge import KnowledgeBase
from ai_knot.multi_agent.facets import _content_tokens
from ai_knot.pool import SharedMemoryPool
from ai_knot.storage.base import StorageBackend
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact, MemoryType

from .fixtures import (
    CHECKPOINTS,
    CORPUS_SHA256,
    MAX_CHARS,
    MAX_FACTS,
    READER,
    TRACES,
    Gold,
    Trace,
    checkpoint_time,
    corpus,
    grants,
    manifest,
    manifest_sha256,
)

ARMS = (
    "ordinary_recall",
    "scoped_ordinary_recall",
    "task_only",
    "task_role",
    "ordinary_integrate_recall",
    "scoped_ordinary_integrate_recall",
)


class ApplicabilityProjection:
    """Benchmark-only candidate projection; source transactions stay unmodified."""

    def __init__(self, storage: StorageBackend, task_key: str) -> None:
        self.storage = storage
        self.task_key = task_key

    def load(self, namespace: str) -> list[Fact]:
        facts = self.storage.load(namespace)
        return [
            fact
            for fact in facts
            if namespace != "__shared__"
            or fact.qualifiers.get("task_key", "")
            in (
                "",
                self.task_key,
            )
        ]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.storage, name)


def storage_descriptor(storage: StorageBackend) -> dict[str, str]:
    """For already isolated caller-owned test storage only; never put this in reports."""
    if isinstance(storage, SQLiteStorage):
        return {"backend": "sqlite", "path": str(Path(storage._db_path).resolve())}
    from ai_knot.storage.postgres_storage import PostgresStorage

    if isinstance(storage, PostgresStorage):
        return {"backend": "postgres", "dsn": storage._dsn}
    raise TypeError("trace harness requires an isolated SQLite or PostgreSQL backend")


def _reopen(descriptor: dict[str, str]) -> StorageBackend:
    if descriptor["backend"] == "sqlite":
        return SQLiteStorage(descriptor["path"])
    from ai_knot.storage.postgres_storage import PostgresStorage

    return PostgresStorage(descriptor["dsn"])


@contextmanager
def isolated_backend(kind: str, directory: Path) -> Iterator[StorageBackend]:
    """Only create/drop our own temporary database or UUID-named schema."""
    if kind == "sqlite":
        yield SQLiteStorage(str(directory / "task-context.db"))
        return
    dsn = os.environ.get("AI_KNOT_TEST_PG_DSN")
    if not dsn:
        raise ValueError("AI_KNOT_TEST_PG_DSN is required for PostgreSQL traces")
    import psycopg
    from psycopg import sql
    from psycopg.conninfo import make_conninfo

    from ai_knot.storage.postgres_storage import PostgresStorage

    schema = "task_context_" + uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        yield PostgresStorage(make_conninfo(dsn, options=f"-csearch_path={schema}"))
    finally:
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def _digest(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _source_state(storage: StorageBackend) -> dict[str, Any]:
    return {
        "shared": sorted(storage.load("__shared__"), key=lambda fact: fact.id),
        "private": storage.load(READER),
        "grants": {key: sorted(value) for key, value in storage.load_grants().items()},
        "stats": storage.load_pool_stats(),
        "usage": storage.load_usage_events(),
        "trust": storage.load_trust_events(),
    }


def _seed(storage: StorageBackend, trace: Trace, point: int) -> None:
    storage.save(READER, [])
    storage.save("__shared__", corpus(trace.family, point))
    for scope in storage.load_grants().get(READER, set()):
        storage.revoke_grant(READER, scope)
    for scope in grants(trace.family, point):
        storage.save_grant(READER, scope, granted_at=checkpoint_time(point).isoformat())


def _memory(storage: StorageBackend, point: int) -> AlwaysOnMemory:
    now = checkpoint_time(point)
    pool = SharedMemoryPool(storage, persist_stats=True, clock=lambda: now)
    return AlwaysOnMemory(
        KnowledgeBase(READER, storage=storage, embed_url=""),
        pool=pool,
        clock=lambda: now,
    )


def _lexical_covered(requirements: tuple[str, ...], facts: tuple[ContextFact, ...]) -> list[str]:
    # Independent direct overlap check: no rank score, role text, or target IDs.
    tokens = set().union(*(set(_content_tokens(fact.content)) for fact in facts))
    return [text for text in requirements if tokens.intersection(_content_tokens(text))]


def _select(
    storage: StorageBackend,
    memory: AlwaysOnMemory,
    trace: Trace,
    point: int,
    arm: str,
) -> tuple[tuple[ContextFact, ...], dict[str, Any]]:
    if arm in ("task_only", "task_role"):
        view = memory.select_context(
            trace.requirements,
            task_key=trace.task_key,
            role=trace.role if arm == "task_role" else "",
            max_facts=MAX_FACTS,
            max_chars=MAX_CHARS,
        )
        return view.facts, {
            "reported_covered_requirements": list(view.covered_requirements),
            "reported_missing_requirements": list(view.missing_requirements),
            "reported_complete": view.complete,
            "budget_limited": view.budget_limited,
            "coverage_report_available": True,
        }
    candidate_storage = (
        ApplicabilityProjection(storage, trace.task_key)
        if arm in ("scoped_ordinary_recall", "scoped_ordinary_integrate_recall")
        else storage
    )
    # A new nonpersistent pool per call prevents cross-arm usage/trust feedback.
    pool = SharedMemoryPool(candidate_storage, persist_stats=False)
    pool._read_scopes = storage.load_grants()
    if arm in ("ordinary_integrate_recall", "scoped_ordinary_integrate_recall"):
        first, second, third = trace.requirements
        query = (
            f"Integrate {first}, {second}, and {third} "
            f"for task {trace.task_key} recipient role {trace.role}"
        )
    else:
        query = f"Task {trace.task_key}. Recipient role {trace.role}. Requirements: " + "; ".join(
            trace.requirements
        )
    pairs = pool.recall(query, READER, top_k=MAX_FACTS, now=checkpoint_time(point))
    refs = tuple(EvidenceRef("__shared__", fact.id, evidence_revision(fact)) for fact, _ in pairs)
    reviewed = memory.review_context(refs, max_chars=MAX_CHARS)
    meta = pool._last_recall_meta
    return reviewed.accepted, {
        "query": query,
        "raw_recalled_ids": [fact.id for fact, _ in pairs],
        "review_issues": encode_always_on(reviewed.issues),
        "review_omitted": encode_always_on(reviewed.omitted),
        # Ordinary recall does not expose explicit supplied-requirement labels.
        "reported_covered_requirements": None,
        "reported_missing_requirements": None,
        "reported_complete": None,
        "coverage_report_available": False,
        "ordinary_facet_report": {
            "covered": sorted(meta.covered_facets),
            "missing": sorted(meta.uncovered_facets),
            "coverage": meta.coverage,
        }
        if meta is not None
        else None,
        "budget_limited": bool(reviewed.omitted),
    }


def _metrics(
    trace: Trace,
    point: int,
    gold: Gold,
    facts: tuple[ContextFact, ...],
    report: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    selected = {fact.ref.fact_id for fact in facts}
    acceptable = set(gold.acceptable)
    current = {fact.id: fact for fact in corpus(trace.family, point)}
    required = [set(group) for group in gold.required_groups if group]
    covered = [bool(selected.intersection(group)) for group in gold.required_groups]
    lexical = set(_lexical_covered(trace.requirements, facts))
    reported = set(report["reported_covered_requirements"] or ())
    has_report = report["coverage_report_available"]
    role_ids = selected.intersection(trace.family.procedure_ids) if trace.matches else set()
    preferred = selected.intersection(gold.preferred_role_ids)
    wrong = [
        key
        for key in selected
        if current[key].qualifiers.get("task_key", "")
        not in (
            "",
            trace.task_key,
        )
    ]
    stale = selected.intersection(
        {
            f"{trace.family.id}-stale",
            *(trace.family.receipt_ids if point == 3 else ()),
        }
    )
    forbidden = selected.intersection(gold.forbidden)
    unsupported = [key for key in selected if not current[key].supported]
    chars = sum(len(fact.content) for fact in facts)
    reasons = []
    for text, group, found in zip(trace.requirements, gold.required_groups, covered, strict=True):
        if not group:
            reasons.append(f"No applicable current gold evidence exists for: {text}")
        elif not found:
            reasons.append(
                f"Required evidence omitted for: {text}; acceptable IDs: {','.join(group)}"
            )
    if forbidden:
        reasons.append("Forbidden selected: " + ",".join(sorted(forbidden)))
    if role_ids and not preferred:
        reasons.append("Valid task evidence selected for the other recipient role.")
    if not reasons:
        reasons.append("All available required evidence selected within the fixed context budget.")
    metrics = {
        "available_required_groups": len(required),
        "required_groups_covered": sum(covered),
        "required_evidence_coverage": sum(covered) / len(required) if required else 1.0,
        "requirements_with_gold_evidence": [
            text for text, found in zip(trace.requirements, covered, strict=True) if found
        ],
        "unavailable_requirements": [
            text
            for text, group in zip(trace.requirements, gold.required_groups, strict=True)
            if not group
        ],
        "recipient_role_choices": len(role_ids),
        "preferred_role_choices": len(preferred),
        "recipient_role_fit": len(preferred) / len(role_ids) if role_ids else None,
        "wrong_applicability_selected": len(wrong),
        "stale_selected": len(stale),
        "unsupported_selected": len(unsupported),
        "forbidden_selected": len(forbidden),
        "raw_episode_selected": sum(current[key].type == MemoryType.EPISODIC for key in selected),
        "role_only_selected": int(f"{trace.family.id}-role-only" in selected),
        "lexically_covered_requirements": [text for text in trace.requirements if text in lexical],
        "missed_lexical_coverage": len(lexical - reported) if has_report else None,
        "falsely_reported_lexical_coverage": len(reported - lexical) if has_report else None,
        "reported_without_gold_evidence": sum(
            text in reported and not found
            for text, found in zip(trace.requirements, covered, strict=True)
        )
        if has_report
        else None,
        "context_chars": chars,
        "selected_count": len(facts),
        "context_precision": len(selected.intersection(acceptable)) / len(selected)
        if selected
        else 1.0,
        "irrelevant_chars": sum(
            len(fact.content) for fact in facts if fact.ref.fact_id not in acceptable
        ),
    }
    return metrics, reasons


def _checkpoint(
    storage: StorageBackend,
    trace: Trace,
    point: int,
    *,
    saved_path: Path | None = None,
    parent_pid: int | None = None,
    expected_source_digest: str | None = None,
) -> list[dict[str, Any]]:
    saved = json.loads(saved_path.read_text()) if saved_path is not None else None
    if saved is not None and set(saved) != set(ARMS):
        raise AssertionError("handoff requires independently serialized references for all arms")
    initial_state = _source_state(storage)
    initial_digest = _digest(initial_state)
    source_digest = _digest(initial_state["shared"])
    expected = sorted(corpus(trace.family, point), key=lambda fact: fact.id)
    same_source = source_digest == _digest(expected) and initial_state["private"] == []
    if not same_source:
        raise AssertionError("persisted corpus differs from the frozen source manifest")
    if expected_source_digest is not None and source_digest != expected_source_digest:
        raise AssertionError("fresh worker failed to reload the saved source corpus")
    rows = []
    for arm in ARMS:
        memory = _memory(storage, point)
        reopened: dict[str, Any] | None = None
        if saved is not None:
            refs = tuple(decode_value(item["ref"], EvidenceRef) for item in saved[arm])
            review = memory.review_context(refs, max_chars=MAX_CHARS)
            reopened = {
                "saved_ids": [ref.fact_id for ref in refs],
                "accepted_ids": [fact.ref.fact_id for fact in review.accepted],
                "issues": encode_always_on(review.issues),
                "complete": review.complete,
            }
        # SQL connections remain available; inference/embedding HTTP is forbidden.
        with (
            patch("ai_knot.embedder.embed_texts", side_effect=AssertionError("embedding call")),
            patch("httpx.Client.send", side_effect=AssertionError("selection HTTP call")),
            patch("httpx.AsyncClient.send", side_effect=AssertionError("selection HTTP call")),
        ):
            started = time.perf_counter()
            facts, selection = _select(storage, memory, trace, point, arm)
            elapsed = time.perf_counter() - started
        metrics, reasons = _metrics(trace, point, trace.checkpoints[point], facts, selection)
        unchanged = _digest(_source_state(storage)) == initial_digest
        if not unchanged:
            raise AssertionError("an arm changed source data, grants, usage or trust")
        rows.append(
            {
                "trace": trace.id,
                "family": trace.family.id,
                "role": trace.role,
                "matching_applicability": trace.matches,
                "task_key": trace.task_key,
                "checkpoint": CHECKPOINTS[point],
                "arm": arm,
                "requirements": list(trace.requirements),
                "gold": asdict(trace.checkpoints[point]),
                "selected": encode_always_on(facts),
                "selection": selection,
                "metrics": metrics,
                "reasons": reasons,
                "latency_seconds": elapsed,
                "selection_model_calls": 0,
                "selection_embedding_calls": 0,
                "source_corpus_sha256": source_digest,
                "checked_at": checkpoint_time(point).isoformat(),
                "reopened_saved_context": reopened,
                "worker_pid": os.getpid(),
                "parent_pid": parent_pid or os.getpid(),
                "controls": {
                    "corpus_unchanged": unchanged,
                    "same_source_corpus": same_source,
                    "same_grants": initial_state["grants"]
                    == {
                        READER: sorted(grants(trace.family, point)),
                    }
                    if grants(trace.family, point)
                    else initial_state["grants"] == {},
                    "max_facts": MAX_FACTS,
                    "max_chars": MAX_CHARS,
                    "fresh_process": parent_pid is not None and parent_pid != os.getpid(),
                    "saved_refs_reloaded": saved is not None,
                    "source_data_reloaded": expected_source_digest is not None,
                    "inference_endpoints_blocked": True,
                },
            }
        )
    return rows


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(encode_always_on(value), ensure_ascii=False, indent=2) + "\n")


def _worker_diagnostic(stderr: object, descriptor: dict[str, str]) -> str:
    text = str(stderr or "")
    if dsn := descriptor.get("dsn"):
        from psycopg import Error
        from psycopg.conninfo import conninfo_to_dict

        try:
            params = conninfo_to_dict(dsn)
        except (Error, ValueError):
            return "[worker output omitted: test DSN could not be sanitized]"
        text = text.replace(dsn, "[test DSN]")
        for key in ("password", "sslpassword"):
            if secret := params.get(key):
                text = text.replace(secret, "[redacted]").replace(
                    quote(secret, safe=""), "[redacted]"
                )
    return text[-4000:]


def run_trace(
    storage: StorageBackend,
    trace: Trace,
    work: Path,
    descriptor: dict[str, str],
) -> list[dict[str, Any]]:
    """Mutates only caller-supplied already isolated test storage; no production DSN discovery."""
    if manifest_sha256() != CORPUS_SHA256:
        raise AssertionError("frozen task-context fixture manifest has changed")
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    saved_path = work / f"{trace.id}-saved.json"
    for point in range(4):
        if point != 2:
            _seed(storage, trace, point)
        if point == 2:
            job = work / f"{trace.id}-handoff.json"
            payload = {
                "storage": descriptor,
                "trace": trace.id,
                "saved_path": str(saved_path),
                "parent_pid": os.getpid(),
                "corpus_manifest_sha256": CORPUS_SHA256,
                "source_digest": _digest(sorted(storage.load("__shared__"), key=lambda f: f.id)),
            }
            # This transient job can contain a test DSN. It never enters reports.
            job.touch(mode=0o600, exist_ok=False)
            _write_json(job, payload)
            try:
                completed = subprocess.run(
                    [sys.executable, "-m", "tests.eval.task_context.runner", "--worker", str(job)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                checkpoint_rows = json.loads(completed.stdout)
            except subprocess.CalledProcessError as exc:
                exc.add_note(
                    f"Trace {trace.id} worker stderr: {_worker_diagnostic(exc.stderr, descriptor)}"
                )
                raise
            finally:
                job.unlink(missing_ok=True)
        else:
            checkpoint_rows = _checkpoint(
                storage,
                trace,
                point,
                saved_path=saved_path if point == 3 else None,
            )
        rows.extend(checkpoint_rows)
        _write_json(saved_path, {row["arm"]: row["selected"] for row in checkpoint_rows})
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for arm in ARMS:
        cases = [row for row in rows if row["arm"] == arm]
        available = sum(row["metrics"]["available_required_groups"] for row in cases)
        covered = sum(row["metrics"]["required_groups_covered"] for row in cases)
        role_choices = sum(row["metrics"]["recipient_role_choices"] for row in cases)
        preferred = sum(row["metrics"]["preferred_role_choices"] for row in cases)
        total_chars = sum(row["metrics"]["context_chars"] for row in cases)
        irrelevant_chars = sum(row["metrics"]["irrelevant_chars"] for row in cases)
        reporting = [row for row in cases if row["selection"]["coverage_report_available"]]
        totals = {
            key: sum(row["metrics"][key] for row in cases)
            for key in (
                "wrong_applicability_selected",
                "stale_selected",
                "unsupported_selected",
                "forbidden_selected",
                "raw_episode_selected",
                "role_only_selected",
            )
        }
        result[arm] = {
            "checkpoints": len(cases),
            "available_required_groups": available,
            "required_groups_covered": covered,
            "required_evidence_coverage": covered / available if available else 1.0,
            "recipient_role_fit": preferred / role_choices if role_choices else None,
            "recipient_role_choices": role_choices,
            "preferred_role_choices": preferred,
            **totals,
            "context_chars": total_chars,
            "irrelevant_chars": irrelevant_chars,
            "character_precision": (total_chars - irrelevant_chars) / total_chars
            if total_chars
            else 1.0,
            "coverage_reporting_checkpoints": len(reporting),
            **{
                key: sum(row["metrics"][key] for row in reporting) if reporting else None
                for key in (
                    "missed_lexical_coverage",
                    "falsely_reported_lexical_coverage",
                    "reported_without_gold_evidence",
                )
            },
            "latency_median_seconds": statistics.median(row["latency_seconds"] for row in cases),
            "latency_max_seconds": max(row["latency_seconds"] for row in cases),
            "selection_model_calls": 0,
            "selection_embedding_calls": 0,
        }
    return result


def run_benchmark(
    storage: StorageBackend,
    work: Path,
    *,
    backend: str,
) -> dict[str, Any]:
    descriptor = storage_descriptor(storage)
    rows = [row for trace in TRACES for row in run_trace(storage, trace, work, descriptor)]
    # Applicability/role pairs receive byte-equivalent data at every checkpoint.
    equivalent = all(
        len(
            {
                row["source_corpus_sha256"]
                for row in rows
                if row["family"] == family and row["checkpoint"] == checkpoint
            }
        )
        == 1
        for family in {trace.family.id for trace in TRACES}
        for checkpoint in CHECKPOINTS
    )
    equivalent = equivalent and all(row["controls"]["same_grants"] for row in rows)
    assert equivalent and len(rows) == 16 * 4 * len(ARMS)
    return {
        "backend": backend,
        "corpus_manifest_sha256": CORPUS_SHA256,
        "trace_count": 16,
        "checkpoints_per_arm": 64,
        "dataset_arm_equivalence": equivalent,
        "max_facts": MAX_FACTS,
        "max_chars": MAX_CHARS,
        "summary": summarize(rows),
        "raw_results": rows,
        "limitations": [
            "Synthetic editorial gold measures selection, "
            "not actual task or model answer accuracy.",
            "Lexical completeness is overlap, not entailment, "
            "task completion, or action authority.",
            "Ordinary recall has no supplied-requirement coverage report; those metrics are null.",
            "Role is a preference; another role's applicable procedure remains valid evidence.",
            "No latency threshold, model calls, embedding calls, or selector tuning to labels.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sqlite", "postgres"), default="sqlite")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        job = json.loads(args.worker.read_text())
        if job["corpus_manifest_sha256"] != CORPUS_SHA256:
            raise AssertionError("worker manifest differs")
        trace = next(trace for trace in TRACES if trace.id == job["trace"])
        rows = _checkpoint(
            _reopen(job["storage"]),
            trace,
            2,
            saved_path=Path(job["saved_path"]),
            parent_pid=job["parent_pid"],
            expected_source_digest=job["source_digest"],
        )
        print(json.dumps(rows, ensure_ascii=False))
        return
    if args.output_dir is None:
        parser.error("--output-dir is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ai-knot-task-context-") as directory:
        work = Path(directory)
        with isolated_backend(args.backend, work) as storage:
            result = run_benchmark(storage, work, backend=args.backend)
    _write_json(args.output_dir / "manifest.json", manifest())
    _write_json(args.output_dir / "results.json", result)
    print(
        json.dumps({key: value for key, value in result.items() if key != "raw_results"}, indent=2)
    )


if __name__ == "__main__":
    main()
