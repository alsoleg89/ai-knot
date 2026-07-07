"""Zero-network demo of the ai-knot HTTP sidecar memory surface.

This example does **not** bind a real port or call external services. It uses
FastAPI's in-process test client to exercise the exact JSON routes exposed by
``ai-knot serve``:

- ``GET /health``
- ``POST /v1/facts``
- ``POST /v1/search``
- ``GET /v1/facts``
- ``GET /v1/facts/{fact_id}``
- ``DELETE /v1/facts/{fact_id}``

Run::

    pip install "ai-knot[server]"
    python examples/http_sidecar_surface_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_knot import KnowledgeBase
from ai_knot.storage import SQLiteStorage


@dataclass
class DemoResult:
    health_status: str
    version: str
    added_fact_id: str
    search_context: str
    search_fact_contents: list[str]
    listed_fact_contents: list[str]
    fetched_content: str
    delete_status: int
    remaining_fact_contents: list[str]


def build_demo_result() -> DemoResult:
    try:
        from fastapi.testclient import TestClient

        from ai_knot.server import create_app
    except ImportError as exc:  # pragma: no cover - exercised manually
        raise SystemExit(
            'Install the server extra first: pip install "ai-knot[server]"'
        ) from exc

    with TemporaryDirectory(prefix="ai-knot-http-sidecar-") as tmpdir:
        kb = KnowledgeBase(
            agent_id="http-sidecar-surface-demo",
            storage=SQLiteStorage(db_path=str(Path(tmpdir) / "memory.db")),
            embed_url="",
        )
        client = TestClient(create_app(kb))

        health = client.get("/health")
        added = client.post(
            "/v1/facts",
            json={
                "content": "User deploys APIs with Docker Compose",
                "importance": 0.92,
                "tags": ["ops"],
            },
        )
        client.post(
            "/v1/facts",
            json={
                "content": "User prefers Python over Java",
                "importance": 0.88,
                "tags": ["stack"],
            },
        )

        added_body = added.json()
        fact_id = added_body["id"]
        searched = client.post(
            "/v1/search",
            json={"query": "what should the deployment checklist use?", "top_k": 5},
        ).json()
        listed = client.get("/v1/facts").json()["facts"]
        fetched = client.get(f"/v1/facts/{fact_id}").json()
        deleted = client.delete(f"/v1/facts/{fact_id}")
        remaining = client.get("/v1/facts").json()["facts"]

        return DemoResult(
            health_status=health.json()["status"],
            version=health.json()["version"],
            added_fact_id=fact_id,
            search_context=searched["context"],
            search_fact_contents=[fact["content"] for fact in searched["facts"]],
            listed_fact_contents=[fact["content"] for fact in listed],
            fetched_content=fetched["content"],
            delete_status=deleted.status_code,
            remaining_fact_contents=[fact["content"] for fact in remaining],
        )


def main() -> None:
    result = build_demo_result()

    print("=== HTTP sidecar memory surface (no real port) ===")
    print("This demo uses FastAPI TestClient against create_app(...).")
    print(
        "Serve the same JSON surface on a real socket with: "
        "ai-knot --storage sqlite serve assistant --port 8000"
    )
    print()
    print("Routes exercised:")
    print("  GET /health")
    print("  POST /v1/facts")
    print("  POST /v1/search")
    print("  GET /v1/facts")
    print("  GET /v1/facts/<fact_id>")
    print("  DELETE /v1/facts/<fact_id>")
    print()
    print("Health:")
    print(f"  status={result.health_status} version={result.version}")
    print()
    print("Added fact id:")
    print(f"  {result.added_fact_id}")
    print()
    print("Search context:")
    print(result.search_context)
    print()
    print("Search fact contents:")
    for content in result.search_fact_contents:
        print(f"  - {content}")
    print()
    print("Listed fact contents:")
    for content in result.listed_fact_contents:
        print(f"  - {content}")
    print()
    print("Fetched fact content:")
    print(f"  {result.fetched_content}")
    print()
    print("Delete status:")
    print(f"  {result.delete_status}")
    print()
    print("Remaining facts after delete:")
    for content in result.remaining_fact_contents:
        print(f"  - {content}")
    print("(No external services or real socket bind were used in this demo.)")


if __name__ == "__main__":
    main()
