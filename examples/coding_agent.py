"""Coding-agent memory workflow — multi-agent shared pool example.

Demonstrates the full ai-knot feature set for a coding-assistant scenario:
  - Two agents (architect + developer) with private knowledge bases
  - Slot-addressed facts (entity::attribute) for structured state
  - Shared pool with topic channels and publish gating
  - Auto-trust based on observed recall quality
  - Slot delta sync for efficient cross-agent updates

Run::

    python examples/coding_agent.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ai_knot.knowledge import KnowledgeBase, SharedMemoryPool
from ai_knot.storage.sqlite_storage import SQLiteStorage
from ai_knot.types import MemoryType

# ---------------------------------------------------------------------------
# Setup: two agents sharing a SQLite-backed pool
# ---------------------------------------------------------------------------

tmp_dir = tempfile.mkdtemp(prefix="ai_knot_example_")
db_path = str(Path(tmp_dir) / "shared.db")
storage = SQLiteStorage(db_path)

pool = SharedMemoryPool(storage=storage)
pool.register("architect")
pool.register("developer")

arch_kb = KnowledgeBase("architect", storage=storage)
dev_kb = KnowledgeBase("developer", storage=storage)

print("=" * 60)
print("  ai-knot coding-agent demo")
print("=" * 60)

# ---------------------------------------------------------------------------
# Architect records architectural decisions with slot addressing
# ---------------------------------------------------------------------------

print("\n[architect] Recording design decisions...")

db_fact = arch_kb.add(
    "Team uses PostgreSQL 16 as the primary database",
    importance=0.9,
)
db_fact.entity = "stack"
db_fact.attribute = "database"
db_fact.slot_key = "stack::database"
db_fact.value_text = "PostgreSQL 16"
db_fact.topic_channel = "architecture"

lang_fact = arch_kb.add(
    "Primary language is Python 3.12 with strict mypy",
    importance=0.85,
    type=MemoryType.PROCEDURAL,
)
lang_fact.entity = "stack"
lang_fact.attribute = "language"
lang_fact.slot_key = "stack::language"
lang_fact.value_text = "Python 3.12"
lang_fact.topic_channel = "architecture"
arch_kb.replace_facts([db_fact, lang_fact])

# Publish to shared pool with topic channel (gated by utility)
published = pool.publish(
    "architect",
    [db_fact.id, lang_fact.id],
    kb=arch_kb,
    utility_threshold=0.5,
)
print(f"  Published {len(published)} architecture facts to shared pool")

# ---------------------------------------------------------------------------
# Developer queries the shared pool
# ---------------------------------------------------------------------------

print("\n[developer] Querying shared pool for context...")

results = pool.recall("what database should I use?", "developer", top_k=3)
for fact, score in results:
    trust = pool.get_trust(fact.origin_agent_id)
    print(f"  [{score:.2f}] (trust={trust:.2f}) {fact.content}")

# ---------------------------------------------------------------------------
# Developer learns new info, publishes an update
# ---------------------------------------------------------------------------

print("\n[developer] Updating database decision (migration to CockroachDB)...")

new_db = dev_kb.add(
    "Team migrated to CockroachDB for horizontal scaling",
    importance=0.92,
)
new_db.entity = "stack"
new_db.attribute = "database"
new_db.slot_key = "stack::database"
new_db.value_text = "CockroachDB"
new_db.topic_channel = "architecture"
dev_kb.replace_facts([new_db])

published2 = pool.publish("developer", [new_db.id], kb=dev_kb, utility_threshold=0.5)
mesi = published2[0].mesi_state.value if published2 else "none"
print(f"  Published {len(published2)} update (MESI state: {mesi})")

# ---------------------------------------------------------------------------
# Architect syncs via slot deltas (lightweight)
# ---------------------------------------------------------------------------

print("\n[architect] Syncing changes via slot deltas...")

deltas = pool.sync_slot_deltas("architect")
for delta in deltas:
    print(f"  delta op={delta.op!r} slot={delta.slot_key!r}: {delta.content[:60]}")

# ---------------------------------------------------------------------------
# Final recall — shows updated state
# ---------------------------------------------------------------------------

print("\n[architect] Latest context after sync:")
final = pool.recall("database technology", "architect", top_k=3, topic_channel="architecture")
for fact, score in final:
    print(f"  [{score:.2f}] {fact.content} (v{fact.version})")

# ---------------------------------------------------------------------------
# Trust scores — auto-computed from publish + recall activity
# ---------------------------------------------------------------------------

print("\n[trust scores]")
for agent in ["architect", "developer"]:
    print(f"  {agent}: {pool.get_trust(agent):.3f}")

print("\nDone. Temp files at:", tmp_dir)
