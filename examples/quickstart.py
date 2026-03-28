"""agentmemo quickstart — minimal working example."""

import shutil

from agentmemo import KnowledgeBase, MemoryType

# Create a knowledge base (stores in .agentmemo/ by default).
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
kb.decay()

# Clean up demo data.
shutil.rmtree(".agentmemo", ignore_errors=True)
print("\nDemo complete. Cleaned up .agentmemo/")
