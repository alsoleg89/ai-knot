"""Executable maintenance design model; run this file for stdlib-only checks.

This is NOT the future public AlwaysOnMemory implementation. SQLite tables model
one namespace, a persistent reservation ledger, and revision/generation guards.
Full Fact validation, journal, ACL, retention and PostgreSQL remain release tests.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
import tempfile
import time
import unittest
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


@dataclass(frozen=True)
class Budget:
    max_items: int = 100
    max_model_calls: int = 0
    max_input_bytes: int = 65536
    max_wall_seconds: float = 1.0

    def __post_init__(self) -> None:
        for name in ("max_items", "max_model_calls", "max_input_bytes"):
            value = getattr(self, name)
            minimum = 0 if name == "max_model_calls" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(name)
        if (
            isinstance(self.max_wall_seconds, bool)
            or not math.isfinite(self.max_wall_seconds)
            or self.max_wall_seconds <= 0
        ):
            raise ValueError("max_wall_seconds")


@dataclass(frozen=True)
class Ref:
    fact_id: str
    revision: str


@dataclass(frozen=True)
class Proposal:
    facts: tuple[dict[str, Any], ...]
    covered_episode_ids: tuple[str, ...]
    input_refs: tuple[Ref, ...]


class Conflict(Exception):
    pass


class Store:
    """Synchronous short SQL units: no await, provider call, or lock across I/O."""

    def __init__(self, path: str = ":memory:") -> None:
        self.db = sqlite3.connect(path, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                id TEXT PRIMARY KEY, revision TEXT NOT NULL, content TEXT NOT NULL,
                kind TEXT NOT NULL, active INTEGER NOT NULL, qualifiers TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS policies (
                id TEXT PRIMARY KEY, revision TEXT NOT NULL, verified_until REAL NOT NULL,
                retry_at REAL NOT NULL, failures INTEGER NOT NULL,
                checked_at REAL, last_outcome TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (
                key TEXT PRIMARY KEY, digest TEXT NOT NULL, generation INTEGER NOT NULL,
                expires REAL NOT NULL, reserved INTEGER NOT NULL, input_bytes INTEGER NOT NULL,
                report TEXT, proposal TEXT, consolidated INTEGER NOT NULL,
                verified INTEGER NOT NULL);
        """)

    def add(self, fact_id: str, content: str, *, kind: str = "episodic") -> None:
        revision = hashlib.sha256(canonical([content, kind]).encode()).hexdigest()
        self.db.execute(
            "INSERT OR REPLACE INTO facts VALUES (?, ?, ?, ?, 1, '{}')",
            (fact_id, revision, content, kind),
        )

    def refs(self, rows: list[sqlite3.Row]) -> tuple[Ref, ...]:
        return tuple(Ref(row["id"], row["revision"]) for row in rows)

    def reserve_run(self, key: str, digest: str, now: float) -> tuple[int, dict[str, Any] | None]:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT * FROM runs WHERE key=?", (key,)).fetchone()
            if row and row["digest"] != digest:
                raise Conflict("IdempotencyConflict")
            if row and row["report"]:
                self.db.commit()
                return row["generation"], json.loads(row["report"])
            if row and row["expires"] > now:
                raise Conflict("run_in_progress")
            generation = row["generation"] + 1 if row else 1
            self.db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, 0, 0, NULL, NULL, 0, 0) "
                "ON CONFLICT(key) DO UPDATE SET generation=excluded.generation, "
                "expires=excluded.expires",
                (key, digest, generation, now + 60),
            )
            self.db.commit()
            return generation, None
        except BaseException:
            self.db.rollback()
            raise

    def guard(self, key: str, generation: int, now: float) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM runs WHERE key=?", (key,)).fetchone()
        if not row or row["generation"] != generation or row["expires"] <= now or row["report"]:
            raise Conflict("maintenance_lease_lost")
        return row

    def reserve_call(
        self, key: str, generation: int, now: float, budget: Budget, size: int, cost: int
    ) -> None:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.guard(key, generation, now)
            if row["reserved"] + cost > budget.max_model_calls:
                raise Conflict("model_budget")
            if row["input_bytes"] + size > budget.max_input_bytes:
                raise Conflict("input_budget")
            self.db.execute(
                "UPDATE runs SET reserved=reserved+?, input_bytes=input_bytes+? WHERE key=?",
                (cost, size, key),
            )
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def verify(self, ref: Ref, outcome: str, now: float, interval: int = 60) -> bool:
        if outcome not in {"confirmed", "changed", "unavailable"}:
            raise ValueError("invalid verification outcome")
        row = self.db.execute("SELECT * FROM facts WHERE id=?", (ref.fact_id,)).fetchone()
        if not row or not row["active"] or row["revision"] != ref.revision:
            return False
        old = self.db.execute("SELECT * FROM policies WHERE id=?", (ref.fact_id,)).fetchone()
        failures = old["failures"] + 1 if old and outcome == "unavailable" else 0
        until = now + interval if outcome == "confirmed" else old["verified_until"] if old else now
        retry = now + min(3600, 30 * 2 ** min(failures - 1, 7)) if failures else until
        if outcome == "changed":
            until, retry = min(until, now), now + interval
        checked = old["checked_at"] if old and outcome == "unavailable" else now
        self.db.execute(
            "INSERT OR REPLACE INTO policies VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ref.fact_id, ref.revision, until, retry, failures, checked, outcome),
        )
        return True

    def guarded_verify(self, key: str, generation: int, ref: Ref, outcome: str, now: float) -> bool:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.guard(key, generation, now)
            accepted = self.verify(ref, outcome, now)
            if accepted and outcome == "confirmed":
                self.db.execute("UPDATE runs SET verified=verified+1 WHERE key=?", (key,))
            self.db.commit()
            return accepted
        except BaseException:
            self.db.rollback()
            raise

    def commit(
        self, key: str, generation: int, proposal: Proposal, selected: tuple[Ref, ...], now: float
    ) -> int:
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.guard(key, generation, now)
            # Persist diagnostics even when validation/guards reject the proposal.
            self.db.execute(
                "UPDATE runs SET proposal=? WHERE key=?", (canonical(asdict(proposal)), key)
            )
            supplied = {ref.fact_id: ref.revision for ref in proposal.input_refs}
            expected = {ref.fact_id: ref.revision for ref in selected}
            covered = set(proposal.covered_episode_ids)
            mapped: set[str] = set()
            if not proposal.facts or not covered:
                raise Conflict("empty_proposal")
            if len(supplied) != len(proposal.input_refs) or supplied != expected:
                raise Conflict("invalid_input_refs")
            if len(covered) != len(proposal.covered_episode_ids) or not covered <= expected.keys():
                raise Conflict("invalid_source_mapping")
            output_ids = set()
            for fact in proposal.facts:
                sources = fact.get("source_ids", [])
                if (
                    set(fact) != {"id", "content", "source_ids"}
                    or not isinstance(fact["id"], str)
                    or not fact["id"]
                    or fact["id"] in output_ids
                    or not isinstance(fact["content"], str)
                    or not fact["content"].strip()
                    or not isinstance(sources, list)
                    or not sources
                    or any(not isinstance(source, str) for source in sources)
                    or len(sources) != len(set(sources))
                    or not set(sources) <= covered
                ):
                    raise Conflict("invalid_source_mapping")
                if self.db.execute("SELECT 1 FROM facts WHERE id=?", (fact["id"],)).fetchone():
                    raise Conflict("output_id_collision")
                output_ids.add(fact["id"])
                mapped.update(sources)
            if mapped != covered:
                raise Conflict("unrepresented_episode")
            for ref in selected:
                row = self.db.execute("SELECT * FROM facts WHERE id=?", (ref.fact_id,)).fetchone()
                if not row or row["revision"] != ref.revision or not row["active"]:
                    raise Conflict("StaleEvidence")
            for fact in proposal.facts:
                qualifiers = {"consolidation_ids": canonical(fact["source_ids"])}
                revision = hashlib.sha256(canonical([fact, qualifiers]).encode()).hexdigest()
                self.db.execute(
                    "INSERT INTO facts VALUES (?, ?, ?, 'semantic', 1, ?)",
                    (fact["id"], revision, fact["content"], canonical(qualifiers)),
                )
            for fact_id in covered:
                self.db.execute("UPDATE facts SET active=0 WHERE id=?", (fact_id,))
            self.db.execute(
                "UPDATE runs SET consolidated=consolidated+? WHERE key=?", (len(covered), key)
            )
            self.db.commit()
            return len(covered)
        except BaseException:
            self.db.rollback()
            # Proposal is diagnostic data, not authority to mutate facts.
            self.db.execute(
                "UPDATE runs SET proposal=? WHERE key=? AND generation=?",
                (canonical(asdict(proposal)), key, generation),
            )
            raise


async def run_maintenance(
    store: Store,
    *,
    run_key: str,
    budget: Budget,
    verifier: Any = None,
    consolidator: Any = None,
    clock: Any = time.time,
    monotonic: Any = time.monotonic,
) -> dict[str, Any]:
    """Providers expose stable config_key and nonnegative model_cost; no URL fetching.

    Verifier receives Ref, returns a typed-outcome string in this small model;
    production VerificationResult must also carry the connector evidence pointer.
    """
    providers = [verifier, consolidator]
    for provider in providers:
        if provider is not None and (
            not isinstance(provider.config_key, str)
            or not provider.config_key
            or type(provider.model_cost) is not int
            or provider.model_cost < 0
        ):
            raise ValueError("provider accounting/config identity")
    digest = canonical(
        [asdict(budget), [[p.config_key, p.model_cost] if p else None for p in providers]]
    )
    generation, cached = store.reserve_run(run_key, digest, clock())
    if cached is not None:
        return cached
    deadline = monotonic() + budget.max_wall_seconds
    report: dict[str, Any] = dict(
        run_id=run_key,
        status="completed",
        inspected=0,
        verified=0,
        consolidated=0,
        expired=0,
        model_calls_reserved=0,
        skipped=0,
        reasons=[],
    )
    cancelled = False

    async def invoke(provider: Any, method: str, payload: Any, wire: Any) -> Any:
        remaining = deadline - monotonic()
        if remaining <= 0:
            raise Conflict("wall_budget")
        store.reserve_call(
            run_key,
            generation,
            clock(),
            budget,
            len(canonical(wire).encode("utf-8")),
            provider.model_cost,
        )
        assert not store.db.in_transaction
        return await asyncio.wait_for(getattr(provider, method)(payload), timeout=remaining)

    try:
        due = store.db.execute(
            "SELECT f.* FROM facts f JOIN policies p ON p.id=f.id "
            "WHERE f.active=1 AND (p.retry_at<=? OR p.revision<>f.revision) "
            "ORDER BY f.id LIMIT ?",
            (clock(), budget.max_items),
        ).fetchall()
        for ref in store.refs(due):
            report["inspected"] += 1
            if verifier is None:
                report["skipped"] += 1
                report["reasons"].append("verifier_disabled")
                continue
            try:
                result = await invoke(verifier, "verify", ref, asdict(ref))
            except Conflict:
                raise
            except asyncio.CancelledError:
                report["reasons"].append("source_unavailable:CancelledError")
                # Lease loss or diagnostic failure must not replace cancellation.
                with suppress(Exception):
                    store.guarded_verify(run_key, generation, ref, "unavailable", clock())
                raise
            except Exception as error:
                result = "unavailable"
                report["reasons"].append("source_unavailable:" + type(error).__name__)
                store.guarded_verify(run_key, generation, ref, result, clock())
                continue
            if (
                store.guarded_verify(run_key, generation, ref, result, clock())
                and result == "confirmed"
            ):
                report["verified"] += 1
        remaining_items = budget.max_items - report["inspected"]
        episodes = store.db.execute(
            "SELECT * FROM facts WHERE active=1 AND kind='episodic' ORDER BY id LIMIT ?",
            (remaining_items,),
        ).fetchall()
        report["inspected"] += len(episodes)
        if episodes and consolidator is None:
            report["skipped"] += len(episodes)
            report["reasons"].append("consolidator_disabled")
        elif episodes:
            wire = [dict(row) for row in episodes]
            proposal = await invoke(consolidator, "propose", tuple(wire), wire)
            report["consolidated"] = store.commit(
                run_key, generation, proposal, store.refs(episodes), clock()
            )
        if report["inspected"] == budget.max_items:
            report["status"] = "budget_exhausted"
            report["reasons"].append("item_limit_reached")
    except Conflict as error:
        reason = str(error)
        report["status"] = "budget_exhausted" if reason.endswith("budget") else "conflict"
        report["reasons"].append(reason)
    except asyncio.CancelledError:
        cancelled = True
        report["status"] = "failed"
        report["reasons"].append("cancelled")
        raise
    except Exception as error:
        report["status"] = "failed"
        report["reasons"].append(type(error).__name__)
    finally:
        try:
            row = store.db.execute("SELECT * FROM runs WHERE key=?", (run_key,)).fetchone()
            report["model_calls_reserved"] = row["reserved"]
            report["consolidated"], report["verified"] = row["consolidated"], row["verified"]
            # Lost generation may report locally; never overwrite a successor's report.
            store.db.execute(
                "UPDATE runs SET report=? WHERE key=? AND generation=? "
                "AND expires>? AND report IS NULL",
                (canonical(report), run_key, generation, clock()),
            )
        except Exception:
            if not cancelled:
                raise
    return report


class Provider:
    config_key = "test-consolidator-v1"
    model_cost = 1

    def __init__(self, store: Store, *, mode: str = "valid") -> None:
        self.store, self.mode, self.calls = store, mode, 0

    async def propose(self, episodes: tuple[dict[str, Any], ...]) -> Proposal:
        self.calls += 1
        assert not self.store.db.in_transaction
        ids = tuple(row["id"] for row in episodes)
        refs = tuple(Ref(row["id"], row["revision"]) for row in episodes)
        if self.mode == "timeout":
            raise TimeoutError("connector timeout")
        if self.mode == "foreground":
            self.store.add("foreground", "new episode")
        if self.mode == "stale":
            self.store.add(ids[0], "corrected episode")
        if self.mode == "late":
            self.store.db.execute("UPDATE runs SET generation=generation+1")
        facts = ({"id": "summary", "content": "summary", "source_ids": list(ids)},)
        if self.mode == "empty":
            facts = ()
        if self.mode == "invalid":
            facts[0]["source_ids"] = ["unselected"]
        return Proposal(facts, ids, refs)


class ReferenceChecks(unittest.TestCase):
    def test_cancellation_survives_expiry_takeover_and_failed_diagnostics(self) -> None:
        for failure in ("expiry", "takeover", "closed"):
            with self.subTest(failure=failure):
                store = Store()
                self.addCleanup(store.db.close)
                store.add("fact", "stable", kind="semantic")
                ref = store.refs(store.db.execute("SELECT * FROM facts").fetchall())[0]
                store.verify(ref, "confirmed", -60)

                async def scenario(store: Store = store, failure: str = failure) -> None:
                    entered = asyncio.Event()
                    blocked = asyncio.Event()
                    now = [0.0]

                    class WaitingVerifier:
                        config_key = "waiting-source"
                        model_cost = 0

                        async def verify(self, request: Ref) -> str:
                            entered.set()
                            await blocked.wait()
                            return "confirmed"

                    task = asyncio.create_task(
                        run_maintenance(
                            store,
                            run_key="cancel",
                            budget=Budget(),
                            verifier=WaitingVerifier(),
                            clock=lambda: now[0],
                        )
                    )
                    await entered.wait()
                    now[0] = 61.0
                    if failure == "takeover":
                        row = store.db.execute(
                            "SELECT digest FROM runs WHERE key='cancel'"
                        ).fetchone()
                        generation, _ = store.reserve_run("cancel", row["digest"], now[0])
                        self.assertEqual(generation, 2)
                    elif failure == "closed":
                        store.db.close()
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task
                    self.assertTrue(task.cancelled())
                    if failure == "takeover":
                        row = store.db.execute("SELECT * FROM runs WHERE key='cancel'").fetchone()
                        self.assertEqual(row["generation"], 2)
                        self.assertIsNone(row["report"])

                asyncio.run(scenario())

    def test_duplicate_sources_reject_entire_proposal(self) -> None:
        store = Store()
        self.addCleanup(store.db.close)
        store.add("episode", "known")

        class DuplicateSources(Provider):
            async def propose(self, episodes: tuple[dict[str, Any], ...]) -> Proposal:
                proposal = await super().propose(episodes)
                proposal.facts[0]["source_ids"] *= 2
                return proposal

        report = asyncio.run(
            run_maintenance(
                store,
                run_key="duplicate",
                budget=Budget(max_model_calls=1),
                consolidator=DuplicateSources(store),
            )
        )
        self.assertEqual(report["status"], "conflict")
        self.assertEqual(
            store.db.execute("SELECT active FROM facts WHERE id='episode'").fetchone()[0], 1
        )
        self.assertIsNone(store.db.execute("SELECT * FROM facts WHERE id='summary'").fetchone())

    def batch(self, mode: str = "valid", **kwargs: Any) -> tuple[Store, Provider, dict[str, Any]]:
        store = Store()
        self.addCleanup(store.db.close)
        store.add("episode", "known")
        provider = Provider(store, mode=mode)
        report = asyncio.run(
            run_maintenance(
                store,
                run_key="batch",
                consolidator=provider,
                budget=kwargs.pop("budget", Budget(max_model_calls=1)),
                **kwargs,
            )
        )
        return store, provider, report

    def test_valid_commit_and_foreground_survives(self) -> None:
        store, _, report = self.batch("foreground")
        self.assertEqual(report["consolidated"], 1)
        rows = {r["id"]: r for r in store.db.execute("SELECT * FROM facts")}
        self.assertEqual(rows["episode"]["active"], 0)
        self.assertEqual(rows["foreground"]["active"], 1)
        self.assertEqual(
            json.loads(json.loads(rows["summary"]["qualifiers"])["consolidation_ids"]), ["episode"]
        )

    def test_empty_invalid_and_stale_proposals_preserve_inputs(self) -> None:
        for mode in ("empty", "invalid", "stale", "late"):
            with self.subTest(mode=mode):
                store, _, report = self.batch(mode)
                self.assertEqual(report["status"], "conflict")
                self.assertEqual(
                    store.db.execute("SELECT active FROM facts WHERE id='episode'").fetchone()[0], 1
                )
                self.assertIsNone(
                    store.db.execute("SELECT * FROM facts WHERE id='summary'").fetchone()
                )

    def test_zero_and_utf8_budgets_prevent_call(self) -> None:
        for budget in (Budget(), Budget(max_model_calls=1, max_input_bytes=1)):
            _, provider, report = self.batch(budget=budget)
            self.assertEqual(provider.calls, 0)
            self.assertEqual(report["model_calls_reserved"], 0)
            self.assertEqual(report["status"], "budget_exhausted")

    def test_wall_budget_prevents_call(self) -> None:
        ticks = iter((0.0, 2.0))
        _, provider, report = self.batch(monotonic=lambda: next(ticks))
        self.assertEqual(provider.calls, 0)
        self.assertIn("wall_budget", report["reasons"])

    def test_timeout_consumes_call_and_retry_is_cached(self) -> None:
        store, provider, report = self.batch("timeout")
        replay = asyncio.run(
            run_maintenance(
                store, run_key="batch", budget=Budget(max_model_calls=1), consolidator=provider
            )
        )
        self.assertEqual(replay, report)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(report["model_calls_reserved"], 1)

    def test_concurrent_reservation_and_expired_recovery_keep_budget(self) -> None:
        store = Store()
        self.addCleanup(store.db.close)
        generation, _ = store.reserve_run("run", "digest", 0)
        store.reserve_call("run", generation, 0, Budget(max_model_calls=1), 1, 1)
        with self.assertRaisesRegex(Conflict, "run_in_progress"):
            store.reserve_run("run", "digest", 1)
        recovered, _ = store.reserve_run("run", "digest", 61)
        with self.assertRaisesRegex(Conflict, "model_budget"):
            store.reserve_call("run", recovered, 61, Budget(max_model_calls=1), 1, 1)
        with self.assertRaisesRegex(Conflict, "maintenance_lease_lost"):
            store.guard("run", generation, 61)
        with self.assertRaisesRegex(Conflict, "IdempotencyConflict"):
            store.reserve_run("run", "different", 61)

    def test_source_error_backoff_and_old_revision_rejection(self) -> None:
        store = Store()
        self.addCleanup(store.db.close)
        store.add("fact", "stable", kind="semantic")
        ref = store.refs(store.db.execute("SELECT * FROM facts").fetchall())[0]
        store.verify(ref, "confirmed", -60)

        class Unavailable:
            config_key = "source-v1"
            model_cost = 0

            async def verify(self, request: Ref) -> str:
                raise OSError("offline")

        for index in range(2):
            report = asyncio.run(
                run_maintenance(
                    store,
                    run_key=f"verify-{index}",
                    budget=Budget(),
                    verifier=Unavailable(),
                    clock=lambda index=index: index * 30.0,
                )
            )
            self.assertEqual(report["verified"], 0)
        policy = store.db.execute("SELECT * FROM policies").fetchone()
        self.assertEqual(
            (policy["failures"], policy["verified_until"], policy["retry_at"]), (2, 0, 90)
        )
        store.add("fact", "new", kind="semantic")
        self.assertFalse(store.verify(ref, "confirmed", 91))
        self.assertEqual(store.db.execute("SELECT COUNT(*) FROM facts").fetchone()[0], 1)

    def test_ledger_survives_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "maintenance.db")
            store = Store(path)
            store.add("e", "episode")
            report = asyncio.run(
                run_maintenance(
                    store,
                    run_key="r",
                    budget=Budget(max_model_calls=1),
                    consolidator=Provider(store),
                )
            )
            store.db.close()
            reopened = Store(path)
            try:
                provider = Provider(reopened)
                replay = asyncio.run(
                    run_maintenance(
                        reopened,
                        run_key="r",
                        budget=Budget(max_model_calls=1),
                        consolidator=provider,
                    )
                )
                self.assertEqual(replay, report)
                self.assertEqual(provider.calls, 0)
            finally:
                reopened.db.close()

    def test_async_late_proposal_with_independent_connection(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / "async.db")
            store, other = Store(path), Store(path)
            self.addCleanup(store.db.close)
            self.addCleanup(other.db.close)
            store.add("episode", "original")

            async def scenario() -> None:
                started, release = asyncio.Event(), asyncio.Event()

                class Gated(Provider):
                    async def propose(self, episodes: tuple[dict[str, Any], ...]) -> Proposal:
                        started.set()
                        await release.wait()
                        return await super().propose(episodes)

                provider = Gated(store)
                task = asyncio.create_task(
                    run_maintenance(
                        store,
                        run_key="r",
                        budget=Budget(max_model_calls=1),
                        consolidator=provider,
                    )
                )
                await started.wait()
                self.assertFalse(store.db.in_transaction)
                with self.assertRaisesRegex(Conflict, "run_in_progress"):
                    await run_maintenance(
                        other,
                        run_key="r",
                        budget=Budget(max_model_calls=1),
                        consolidator=Provider(other),
                    )
                other.add("foreground", "arrived while provider awaits")
                other.db.execute(
                    "UPDATE runs SET generation=2, report=? WHERE key='r'",
                    (canonical({"run_id": "successor"}),),
                )
                release.set()
                report = await task
                self.assertEqual(report["status"], "conflict")
                self.assertEqual(
                    json.loads(
                        other.db.execute("SELECT report FROM runs WHERE key='r'").fetchone()[0]
                    ),
                    {"run_id": "successor"},
                )
                self.assertEqual(
                    other.db.execute("SELECT COUNT(*) FROM facts WHERE active=1").fetchone()[0], 2
                )

            asyncio.run(scenario())

    def test_unavailable_and_changed_keep_truth_and_block_freshness(self) -> None:
        store = Store()
        self.addCleanup(store.db.close)
        store.add("f", "unchanged", kind="semantic")
        ref = store.refs(store.db.execute("SELECT * FROM facts").fetchall())[0]
        store.verify(ref, "confirmed", 0, interval=600)
        store.verify(ref, "unavailable", 1, interval=600)
        row = store.db.execute("SELECT * FROM policies").fetchone()
        self.assertEqual((row["verified_until"], row["checked_at"], row["retry_at"]), (600, 0, 31))
        self.assertNotEqual(row["last_outcome"], "confirmed")  # Until alone cannot authorize.
        store.verify(ref, "changed", 2, interval=600)
        row = store.db.execute("SELECT * FROM policies").fetchone()
        self.assertEqual((row["verified_until"], row["checked_at"], row["retry_at"]), (2, 2, 602))
        self.assertEqual(store.db.execute("SELECT content FROM facts").fetchone()[0], "unchanged")

    def test_item_limit_and_disabled_provider(self) -> None:
        store = Store()
        self.addCleanup(store.db.close)
        for index in range(101):
            store.add(str(index), "episode")
        report = asyncio.run(run_maintenance(store, run_key="r", budget=Budget()))
        self.assertEqual((report["inspected"], report["skipped"]), (100, 100))
        self.assertEqual(report["status"], "budget_exhausted")

    def test_budget_validation(self) -> None:
        for kwargs in (
            {"max_items": True},
            {"max_model_calls": -1},
            {"max_input_bytes": 0},
            {"max_wall_seconds": float("nan")},
            {"max_wall_seconds": float("inf")},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                Budget(**kwargs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
