"""Internal memoization for one context read's existing SQL snapshot.

Discard at transaction exit and before provider I/O. Fact objects stay private to
read-only admission code; public results contain immutable text/ref projections.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from ai_knot.types import Fact

from .storage import AlwaysOnSession, SQLRow, SQLScalar, SQLStatement


class ReadCache:
    def __init__(self, session: AlwaysOnSession) -> None:
        self._session = session
        self._facts: dict[tuple[str, str | None, str | None], list[Fact]] = {}
        self._ids: dict[str, dict[str, Fact]] = {}
        self._slots: dict[str, dict[str, list[Fact]]] = {}
        self._rows: dict[tuple[SQLStatement, tuple[SQLScalar, ...]], list[SQLRow]] = {}

    def load_task_facts(self, namespace: str, *, task_key: str) -> Iterator[Fact]:
        facts = list(self._session.load_task_facts(namespace, task_key=task_key))
        total = self._session.fetch_one(
            SQLStatement(
                "SELECT COUNT(*) AS n FROM facts WHERE agent_id=?",
                'SELECT COUNT(*) AS n FROM "ai-knot_facts" WHERE agent_id=%s',
            ),
            (namespace,),
        )
        if total is not None and int(total["n"] or 0) == len(facts):
            # Same SQL snapshot: equality proves the task filter omitted no row.
            self._remember_namespace(namespace, facts)
            self._facts[namespace, None, None] = facts
        for fact in facts:
            # A task subset cannot answer complete namespace or slot lookups.
            self._facts[namespace, fact.id, None] = [fact]
            yield fact

    def _remember_namespace(self, namespace: str, facts: list[Fact]) -> None:
        self._ids[namespace] = {fact.id: fact for fact in facts}
        slots: dict[str, list[Fact]] = {}
        for fact in facts:
            slots.setdefault(fact.slot_key, []).append(fact)
        self._slots[namespace] = slots

    def load_facts(
        self, namespace: str, *, fact_id: str | None = None, slot_key: str | None = None
    ) -> list[Fact]:
        if namespace in self._ids:
            if fact_id is not None:
                fact = self._ids[namespace].get(fact_id)
                return (
                    [fact]
                    if fact is not None and (slot_key is None or fact.slot_key == slot_key)
                    else []
                )
            if slot_key is not None:
                return list(self._slots[namespace].get(slot_key, ()))
        key = namespace, fact_id, slot_key
        if key not in self._facts:
            facts = self._session.load_facts(namespace, fact_id=fact_id, slot_key=slot_key)
            self._facts[key] = facts
            if fact_id is None and slot_key is None:
                self._remember_namespace(namespace, facts)
        return list(self._facts[key])

    def fetch_all(self, statement: SQLStatement, params: Sequence[SQLScalar] = ()) -> list[SQLRow]:
        key = statement, tuple(params)
        if key not in self._rows:
            self._rows[key] = self._session.fetch_all(statement, params)
        return [dict(row) for row in self._rows[key]]

    def fetch_one(self, statement: SQLStatement, params: Sequence[SQLScalar] = ()) -> SQLRow | None:
        rows = self.fetch_all(statement, params)
        return rows[0] if rows else None

    def replace_facts(self, namespace: str, facts: list[Fact]) -> None:
        raise ValueError("read-only transaction")

    def execute(self, statement: SQLStatement, params: Sequence[SQLScalar] = ()) -> int:
        raise ValueError("read-only transaction")
