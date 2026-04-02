"""ai-knot multilingual example — Russian + English in the same knowledge base."""

import shutil
from datetime import UTC, datetime

from ai_knot import KnowledgeBase, MemoryType

# Create a knowledge base with mixed-language facts.
kb = KnowledgeBase(agent_id="multilingual_demo")

# English facts.
kb.add("User is a senior backend developer at Acme Corp", importance=0.95)
kb.add("Team deploys everything in Docker Compose", importance=0.85)
kb.add("User prefers Python for all backend code", type=MemoryType.PROCEDURAL)

# Russian facts — the Cyrillic stemmer normalises morphological variants
# automatically (e.g. "предпочитает" and "предпочитать" → same stem).
kb.add(
    "Пользователь предпочитает PostgreSQL для хранения данных",
    importance=0.90,
)
kb.add(
    "Команда использует Kubernetes для оркестрации контейнеров",
    importance=0.80,
)
kb.add(
    "Релиз запланирован на пятницу",
    type=MemoryType.EPISODIC,
    importance=0.60,
)

# --- Queries in Russian ---
print("=== Query: 'какую базу данных использует пользователь?' ===")
context = kb.recall("какую базу данных использует пользователь?")
print(context)
print()

print("=== Query: 'развёртывание и оркестрация' ===")
context = kb.recall("развёртывание и оркестрация")
print(context)
print()

# --- Queries in English ---
print("=== Query: 'deployment infrastructure' ===")
context = kb.recall("deployment infrastructure")
print(context)
print()

# --- Mixed query (English keywords + Russian context) ---
print("=== Query: 'Docker Kubernetes контейнеры' ===")
context = kb.recall("Docker Kubernetes контейнеры")
print(context)
print()

# --- Clock injection: test how decay affects ranking over time ---
future = datetime(2027, 6, 1, tzinfo=UTC)
print("=== Same query, 1 year later (episodic facts decay faster) ===")
context_future = kb.recall("релиз пятница", now=future)
print(context_future or "(no relevant facts — decayed below threshold)")
print()

# Stats.
stats = kb.stats()
print(f"Total facts: {stats['total_facts']}")
print(f"By type: {stats['by_type']}")

# Clean up.
shutil.rmtree(".ai_knot", ignore_errors=True)
print("\nDemo complete. Cleaned up .ai_knot/")
