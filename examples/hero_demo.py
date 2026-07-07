"""20-second hero demo for README / launch recordings.

Run::

    python examples/hero_demo.py

This is intentionally short and deterministic so it can be captured as a GIF,
terminal recording, or launch clip without any API keys or setup beyond the
package itself.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ai_knot import KnowledgeBase, MemoryType
from ai_knot.storage import SQLiteStorage


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="ai_knot_demo_") as tmp:
        db_path = Path(tmp) / "hero-demo.db"

        print("=== ai-knot: 20-second demo ===")
        print("Store facts once. Recall only what the next turn needs.\n")

        writer = KnowledgeBase(
            agent_id="assistant",
            storage=SQLiteStorage(db_path=str(db_path)),
            embed_url="",
        )
        writer.add("User is a senior backend developer at Acme Corp")
        writer.add("User prefers Python and dislikes async code", type=MemoryType.PROCEDURAL)
        writer.add("User deploys services with Docker and Kubernetes")
        print("Learned 3 facts into SQLite.\n")

        print("Fresh process. Same memory on disk. No transcript replay.\n")
        reader = KnowledgeBase(
            agent_id="assistant",
            storage=SQLiteStorage(db_path=str(db_path)),
            embed_url="",
        )
        print("Query: what stack and preferences should I use?\n")
        print(reader.recall("what stack and preferences should I use?"))


if __name__ == "__main__":
    main()
