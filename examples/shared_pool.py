"""SharedMemoryPool — multi-agent knowledge sharing demo.

Three agents (planner, coder, reviewer) publish structured facts to a
shared pool. Demonstrates:

  - Slot-addressed deduplication (entity::attribute — no duplicate facts)
  - Slot supersession (one agent updates another's slot value)
  - Topic channel filtering (only "architecture" facts returned for arch queries)
  - Auto-trust based on recall quality
  - Lightweight slot-delta sync

No LLM required — all facts are inserted manually.

Run::

    python examples/shared_pool.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ai_knot.knowledge import KnowledgeBase, SharedMemoryPool
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import Fact, MemoryType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def slotted(
    content: str,
    *,
    entity: str,
    attribute: str,
    value_text: str,
    channel: str = "",
    importance: float = 0.9,
    type: MemoryType = MemoryType.SEMANTIC,
) -> Fact:
    """Build a slot-addressed Fact.

    entity and attribute are normalised to lowercase so slot_key matching
    is case-insensitive regardless of LLM casing drift.
    """
    ent = entity.lower()
    att = attribute.lower()
    return Fact(
        content=content,
        type=type,
        importance=importance,
        entity=ent,
        attribute=att,
        slot_key=f"{ent}::{att}",
        value_text=value_text,
        topic_channel=channel,
    )


# ---------------------------------------------------------------------------
# Setup: three agents sharing a SQLite-backed pool
# ---------------------------------------------------------------------------

tmp_dir = tempfile.mkdtemp(prefix="ai_knot_pool_")
storage = SQLiteStorage(str(Path(tmp_dir) / "shared.db"))

pool = SharedMemoryPool(storage=storage)
pool.register("planner")
pool.register("coder")
pool.register("reviewer")

planner = KnowledgeBase("planner", storage=storage)
coder = KnowledgeBase("coder", storage=storage)
reviewer = KnowledgeBase("reviewer", storage=storage)

print("=" * 60)
print("  ai-knot SharedMemoryPool demo")
print("=" * 60)


# ---------------------------------------------------------------------------
# Step 1: Each agent builds its private knowledge base
# ---------------------------------------------------------------------------

print("\n[planner] Recording project decisions...")
planner_facts = [
    slotted(
        "Primary database: PostgreSQL 16 on RDS",
        entity="stack",
        attribute="database",
        value_text="PostgreSQL 16",
        channel="architecture",
    ),
    slotted(
        "Backend language: Python 3.12 with strict mypy",
        entity="stack",
        attribute="language",
        value_text="Python 3.12",
        channel="architecture",
        importance=0.85,
    ),
    slotted(
        "Target deployment: Kubernetes on GKE",
        entity="stack",
        attribute="deployment",
        value_text="Kubernetes/GKE",
        channel="infrastructure",
    ),
]
planner.replace_facts(planner_facts)

print("\n[coder] Recording implementation standards...")
coder_facts = [
    slotted(
        "Auth: all endpoints require JWT Bearer tokens",
        entity="security",
        attribute="auth_scheme",
        value_text="JWT Bearer",
        channel="security",
        importance=0.95,
    ),
    slotted(
        "Testing: pytest + hypothesis, coverage ≥ 80%",
        entity="quality",
        attribute="testing",
        value_text="pytest+hypothesis",
        channel="engineering",
        type=MemoryType.PROCEDURAL,
    ),
]
coder.replace_facts(coder_facts)

print("\n[reviewer] Recording process rules...")
reviewer_facts = [
    slotted(
        "PRs require at least two approvals before merge",
        entity="process",
        attribute="pr_policy",
        value_text="2-approvals",
        channel="process",
        type=MemoryType.PROCEDURAL,
    ),
]
reviewer.replace_facts(reviewer_facts)


# ---------------------------------------------------------------------------
# Step 2: All agents publish to the shared pool
# ---------------------------------------------------------------------------

print("\n[pool] Publishing from all agents...")
for agent, kb, facts in [
    ("planner", planner, planner_facts),
    ("coder", coder, coder_facts),
    ("reviewer", reviewer, reviewer_facts),
]:
    published = pool.publish(agent, [f.id for f in facts], kb=kb, utility_threshold=0.5)
    print(f"  {agent}: {len(published)} fact(s) published")


# ---------------------------------------------------------------------------
# Step 3: Developer queries the pool
# ---------------------------------------------------------------------------

print("\n[coder] Querying pool for architecture context...")
results = pool.recall(
    "what database and language are we using?",
    "coder",
    top_k=3,
    topic_channel="architecture",
)
for fact, score in results:
    trust = pool.get_trust(fact.origin_agent_id)
    print(f"  [{score:.3f}] (from={fact.origin_agent_id}, trust={trust:.2f}) {fact.content}")


# ---------------------------------------------------------------------------
# Step 4: Slot supersession — planner updates the database decision
# ---------------------------------------------------------------------------

print("\n[planner] Updating database — switching to CockroachDB...")
new_db = slotted(
    "Database migrated to CockroachDB for horizontal scaling",
    entity="stack",
    attribute="database",  # same slot_key as original
    value_text="CockroachDB",
    channel="architecture",
    importance=0.95,
)
# Append the new fact to planner's KB (old version remains as history)
planner.replace_facts(planner.list_facts() + [new_db])
published_update = pool.publish("planner", [new_db.id], kb=planner, utility_threshold=0.5)

if published_update:
    mesi = published_update[0].mesi_state.value
    print(f"  Slot stack::database superseded (MESI state: {mesi})")
    print(f"  New value: {published_update[0].content}")


# ---------------------------------------------------------------------------
# Step 5: Reviewer syncs changes via slot deltas
# ---------------------------------------------------------------------------

print("\n[reviewer] Syncing changes via slot deltas...")
deltas = pool.sync_slot_deltas("reviewer")
if deltas:
    for delta in deltas:
        print(f"  op={delta.op!r}  slot={delta.slot_key!r}")
        print(f"    → {delta.content[:70]}")
else:
    print("  (no new deltas)")


# ---------------------------------------------------------------------------
# Step 6: Verify only one active fact per slot (no duplicates)
# ---------------------------------------------------------------------------

print("\n[pool] Active facts for slot stack::database:")
active = pool.recall("database technology", "reviewer", top_k=5, topic_channel="architecture")
db_facts = [f for f, _ in active if f.slot_key == "stack::database"]
print(f"  {len(db_facts)} active fact(s) — expected 1")
for f in db_facts:
    print(f"  v{f.version}: {f.content}")


# ---------------------------------------------------------------------------
# Step 7: Trust scores
# ---------------------------------------------------------------------------

print("\n[trust scores]  (higher = agent's facts are recalled more often)")
for agent in ["planner", "coder", "reviewer"]:
    print(f"  {agent}: {pool.get_trust(agent):.3f}")

print(f"\nDone. Temp files at: {tmp_dir}")
