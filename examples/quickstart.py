"""ai-knot quickstart — minimal working example."""

import shutil
from datetime import UTC, datetime

from ai_knot import KnowledgeBase, MemoryType

# Create a knowledge base (stores in .ai_knot/ by default).
# Optional: rrf_weights tune the balance between BM25, importance, retention, recency.
kb = KnowledgeBase(agent_id="demo")

# Add facts manually.
kb.add("User is a senior backend developer at Acme Corp", importance=0.95)
kb.add("User prefers Python, dislikes async code", type=MemoryType.PROCEDURAL, importance=0.85)
kb.add("User deploys everything in Docker", importance=0.80)
kb.add("Deploy failed last Tuesday", type=MemoryType.EPISODIC, importance=0.40)

# Recall relevant facts for a query.
print("=== Query: 'how should I write this deployment script?' ===")
context = kb.recall("how should I write this deployment script?")
print(context)

print()

print("=== Query: 'where does the user work?' ===")
context = kb.recall("where does the user work?")
print(context)

print()

# Check stats.
stats = kb.stats()
print("=== Stats ===")
print(f"Total facts: {stats['total_facts']}")
print(f"By type: {stats['by_type']}")
print(f"Avg importance: {stats['avg_importance']:.2f}")
print(f"Avg retention: {stats['avg_retention']:.2f}")

# Apply decay (in real usage, this happens automatically on recall).
# The `now` parameter allows testing how facts look at a future point in time.
kb.decay()

# Clock injection: see how facts rank 6 months from now.
future = datetime(2026, 10, 1, tzinfo=UTC)
print("\n=== Same query, 6 months later (episodic facts decay faster) ===")
context = kb.recall("deployment", now=future)
print(context)

# Clean up demo data.
shutil.rmtree(".ai_knot", ignore_errors=True)
print("\nDemo complete. Cleaned up .ai_knot/")
