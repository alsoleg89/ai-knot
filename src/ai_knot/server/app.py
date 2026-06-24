"""FastAPI application exposing the knowledge base over HTTP.

Endpoints (all JSON):

  ``GET  /health``      → liveness + version (unauthenticated)
  ``POST /v1/recall``   → point-in-time recall: context string + structured facts
  ``POST /v1/facts``    → add a fact
  ``GET  /v1/stats``    → knowledge-base statistics

When a bearer token is configured (``create_app(kb, token=...)`` or the
``AI_KNOT_SERVER_TOKEN`` env var via ``ai-knot serve``), every ``/v1/*`` route
requires ``Authorization: Bearer <token>``. ``/health`` is always open.

Recall never calls an LLM; add/recall are the same proven KnowledgeBase methods
used by the MCP server, so the HTTP surface inherits their behaviour (including
``now`` point-in-time semantics).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from ai_knot import __version__
from ai_knot.knowledge import KnowledgeBase
from ai_knot.types import Fact, MemoryType

_MAX_TOP_K = 200


class RecallRequest(BaseModel):
    """Body for ``POST /v1/recall``."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=_MAX_TOP_K)
    now: str | None = Field(
        default=None,
        description="ISO-8601 point-in-time anchor; facts not yet active at it are excluded.",
    )


class FactRequest(BaseModel):
    """Body for ``POST /v1/facts``."""

    content: str = Field(min_length=1)
    type: str = "semantic"
    importance: float = Field(default=0.8, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    event_time: str | None = Field(default=None, description="ISO-8601 time the knowledge held.")


def _parse_dt(value: str | None) -> datetime | None:
    """Parse an ISO-8601 string, or raise a 422 with a clear message."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"invalid ISO-8601 datetime: {value!r}"
        ) from exc
    # Naive inputs (e.g. a date-only "2023-05-08") are treated as UTC so they can
    # be compared against the store's timezone-aware validity bounds.
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _serialize_fact(fact: Fact) -> dict[str, Any]:
    """JSON-safe projection of a Fact (the fields a client needs)."""
    return {
        "id": fact.id,
        "content": fact.content,
        "type": fact.type.value,
        "importance": fact.importance,
        "tags": list(fact.tags),
        "created_at": fact.created_at.isoformat(),
        "event_time": fact.event_time.isoformat() if fact.event_time else None,
        "valid_from": fact.valid_from.isoformat() if fact.valid_from else None,
        "valid_until": fact.valid_until.isoformat() if fact.valid_until else None,
    }


def create_app(kb: KnowledgeBase, *, token: str | None = None) -> FastAPI:
    """Build the FastAPI app over *kb*.

    Args:
        kb: the knowledge base to serve.
        token: optional bearer token; when set, ``/v1/*`` routes require
            ``Authorization: Bearer <token>``.
    """
    app = FastAPI(title="ai-knot", version=__version__, summary="Agent knowledge layer over HTTP")

    def require_auth(authorization: str | None = Header(default=None)) -> None:
        if token is None:
            return
        if authorization != f"Bearer {token}":
            raise HTTPException(status_code=401, detail="missing or invalid bearer token")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.post("/v1/recall")
    def recall(req: RecallRequest, _: None = Depends(require_auth)) -> dict[str, Any]:
        now = _parse_dt(req.now)
        facts = kb.recall_facts(req.query, top_k=req.top_k, now=now)
        context = kb.recall(req.query, top_k=req.top_k, now=now)
        return {"context": context, "facts": [_serialize_fact(f) for f in facts]}

    @app.post("/v1/facts", status_code=201)
    def add_fact(req: FactRequest, _: None = Depends(require_auth)) -> dict[str, Any]:
        try:
            mtype = MemoryType(req.type)
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"invalid memory type {req.type!r}; expected one of "
                f"{[t.value for t in MemoryType]}",
            ) from exc
        fact = kb.add(
            req.content,
            type=mtype,
            importance=req.importance,
            tags=req.tags,
            event_time=_parse_dt(req.event_time),
        )
        return _serialize_fact(fact)

    @app.get("/v1/stats")
    def stats(_: None = Depends(require_auth)) -> dict[str, Any]:
        return kb.stats()

    return app
