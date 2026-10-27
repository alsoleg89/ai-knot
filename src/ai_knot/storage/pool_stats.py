"""Transaction-local shared-pool telemetry; legacy snapshots need no migration."""

from __future__ import annotations

import json
from typing import Any

from ai_knot.always_on.storage import AlwaysOnSession, SQLStatement

COUNTERS = ("publish_count", "used_count", "quick_inv_count", "slot_publish_count")


def stats_delta(current: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in COUNTERS:
        old = baseline.get(key, {})
        delta[key] = {
            agent: value - old.get(agent, 0)
            for agent, value in current.get(key, {}).items()
            if value != old.get(agent, 0)
        }
    old_consumers = baseline.get("fact_consumers", {})
    delta["fact_consumers"] = {
        fact: sorted(set(consumers) - set(old_consumers.get(fact, [])))
        for fact, consumers in current.get("fact_consumers", {}).items()
        if set(consumers) - set(old_consumers.get(fact, []))
    }
    return delta


def merge_stats(session: AlwaysOnSession, delta: dict[str, Any]) -> dict[str, Any]:
    row = session.fetch_one(
        SQLStatement(
            "SELECT data FROM pool_stats WHERE id=0",
            'SELECT data FROM "ai-knot_pool_stats" WHERE id=0',
        )
    )
    result = json.loads(str(row["data"])) if row else {}
    for key in COUNTERS:
        values = result.setdefault(key, {})
        for agent, value in delta.get(key, {}).items():
            values[agent] = values.get(agent, 0) + value
    consumers = result.setdefault("fact_consumers", {})
    for fact, agents in delta.get("fact_consumers", {}).items():
        consumers[fact] = sorted(set(consumers.get(fact, [])) | set(agents))
    session.execute(
        SQLStatement(
            "INSERT INTO pool_stats (id,data) VALUES (0,?) "
            "ON CONFLICT(id) DO UPDATE SET data=excluded.data",
            'INSERT INTO "ai-knot_pool_stats" (id,data) VALUES (0,%s) '
            "ON CONFLICT(id) DO UPDATE SET data=excluded.data",
        ),
        (json.dumps(result),),
    )
    return result


def trust_event(
    session: AlwaysOnSession, ts: str, agent: str, kind: str, delta: float, reason: str = ""
) -> None:
    session.execute(
        SQLStatement(
            "INSERT INTO trust_events (ts,agent_id,event_type,delta,reason) VALUES (?,?,?,?,?)",
            'INSERT INTO "ai-knot_trust_events" (ts,agent_id,event_type,delta,reason) '
            "VALUES (%s,%s,%s,%s,%s)",
        ),
        (ts, agent, kind, delta, reason),
    )
