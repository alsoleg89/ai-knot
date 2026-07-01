"""Browser inspector demo for the ai-knot HTTP sidecar.

Seeds a small deterministic knowledge base, then runs the HTTP sidecar so you
can inspect stored facts in a browser with no API keys or external services.

Run::

    pip install "ai-knot[server]"
    python examples/browser_inspector_demo.py

Then open::

    http://127.0.0.1:8000/inspect
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ai_knot import KnowledgeBase
from ai_knot.storage.sqlite_storage import SQLiteStorage


def build_demo_kb(
    *,
    base_dir: str | Path = ".ai_knot/browser-inspector-demo",
    agent_id: str = "browser-demo",
) -> KnowledgeBase:
    """Create an idempotent demo knowledge base with active + inactive facts."""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    kb = KnowledgeBase(
        agent_id=agent_id,
        storage=SQLiteStorage(db_path=str(base_path / "memory.db")),
        embed_url="",
    )
    kb.clear_all()

    kb.add(
        "User is a senior backend engineer working on a payment platform",
        importance=0.95,
        tags=["profile", "backend"],
    )
    kb.add(
        "Primary language is Python",
        importance=0.98,
        tags=["stack", "language"],
    )
    kb.add(
        "API framework is FastAPI",
        importance=0.96,
        tags=["stack", "framework"],
    )
    kb.add(
        "Primary database is PostgreSQL",
        importance=0.96,
        tags=["stack", "database"],
    )
    kb.add(
        "Deployments run through Docker",
        importance=0.94,
        tags=["stack", "ops"],
    )
    kb.add(
        "Deployments happen on Tuesdays after 15:00 UTC",
        type="procedural",
        importance=0.9,
        tags=["runbook", "deploy"],
    )
    kb.add(
        "A data migration is planned for 2099 and should stay hidden from live recall today",
        event_time=datetime(2099, 1, 1, tzinfo=UTC),
        tags=["future", "migration"],
    )
    return kb


def main() -> None:
    try:
        import uvicorn

        from ai_knot.server import create_app
    except ImportError as exc:  # pragma: no cover - exercised manually
        raise SystemExit(
            "Install the server extra first: pip install \"ai-knot[server]\""
        ) from exc

    kb = build_demo_kb()
    app = create_app(kb)
    print("ai-knot browser inspector demo")
    print("Open: http://127.0.0.1:8000/inspect")
    print("Press Ctrl-C to stop.")
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
