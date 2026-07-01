"""FastAPI application exposing the knowledge base over HTTP.

Endpoints (all JSON):

  ``GET  /health``      → liveness + version (unauthenticated)
  ``GET  /v1/facts``    → list stored facts for inspection/debugging
  ``POST /v1/recall``   → point-in-time recall: context string + structured facts
  ``POST /v1/facts``    → add a fact
  ``GET  /v1/stats``    → knowledge-base statistics
  ``GET  /inspect``     → read-only browser inspector over the same knowledge base

When a bearer token is configured (``create_app(kb, token=...)`` or the
``AI_KNOT_SERVER_TOKEN`` env var via ``ai-knot serve``), every ``/v1/*`` route
and ``/inspect`` require ``Authorization: Bearer <token>``. ``/health`` is
always open.

Recall never calls an LLM; add/recall are the same proven KnowledgeBase methods
used by the MCP server, so the HTTP surface inherits their behaviour (including
``now`` point-in-time semantics).
"""

from __future__ import annotations

from datetime import UTC, datetime
from html import escape
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ai_knot import __version__
from ai_knot.knowledge import KnowledgeBase
from ai_knot.types import Fact, MemoryType

_MAX_TOP_K = 200
_DEFAULT_LIST_LIMIT = 100
_MAX_LIST_LIMIT = 500


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


def _serialize_fact(fact: Fact, *, active_at: datetime | None = None) -> dict[str, Any]:
    """JSON-safe projection of a Fact (the fields a client needs)."""
    payload = {
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
    if active_at is not None:
        payload["active"] = fact.is_active(active_at)
    return payload


def _matching_facts(
    kb: KnowledgeBase,
    *,
    now_dt: datetime,
    include_inactive: bool,
) -> list[Fact]:
    """Return facts filtered for an inspection/listing view."""
    facts = kb.list_facts()
    if not include_inactive:
        facts = [fact for fact in facts if fact.is_active(now_dt)]
    return sorted(facts, key=lambda fact: (fact.created_at, fact.id), reverse=True)


def _fmt_dt(value: datetime | None) -> str:
    """Format datetimes consistently for the HTML inspector."""
    if value is None:
        return "—"
    return value.isoformat(timespec="seconds")


def _render_inspector_html(
    *,
    agent_id: str,
    token_enabled: bool,
    query: str,
    top_k: int,
    now: str,
    include_inactive: bool,
    limit: int,
    stats: dict[str, Any],
    listed_facts: list[Fact],
    total_matching_facts: int,
    recalled_facts: list[Fact],
    recall_context: str,
    anchor: datetime,
) -> str:
    """Render a small, dependency-free browser inspector."""
    fact_rows = []
    for fact in listed_facts:
        tags = ", ".join(fact.tags) if fact.tags else "—"
        status = "active" if fact.is_active(anchor) else "inactive"
        fact_rows.append(
            "<tr>"
            f"<td>{escape(status)}</td>"
            f"<td>{escape(fact.type.value)}</td>"
            f"<td><code>{escape(fact.id)}</code></td>"
            f"<td>{escape(fact.content)}</td>"
            f"<td>{fact.importance:.2f}</td>"
            f"<td>{escape(tags)}</td>"
            f"<td>{escape(_fmt_dt(fact.created_at))}</td>"
            f"<td>{escape(_fmt_dt(fact.event_time))}</td>"
            f"<td>{escape(_fmt_dt(fact.valid_until))}</td>"
            "</tr>"
        )

    recall_block = ""
    if query:
        recall_rows = []
        for fact in recalled_facts:
            recall_rows.append(
                "<li>"
                f"<code>{escape(fact.id)}</code> "
                f"<strong>{escape(fact.type.value)}</strong> "
                f"{escape(fact.content)}"
                "</li>"
            )
        recall_list = "\n".join(recall_rows) if recall_rows else "<li>No matching facts.</li>"
        recall_block = f"""
<section>
  <h2>Recall Preview</h2>
  <p><strong>Query:</strong> <code>{escape(query)}</code></p>
  <pre>{escape(recall_context or "No matching facts.")}</pre>
  <ul>
    {recall_list}
  </ul>
</section>
"""

    token_note = (
        "Bearer token required"
        if token_enabled
        else "Open locally unless your reverse proxy adds auth"
    )
    checked = " checked" if include_inactive else ""
    rows_html = (
        "\n".join(fact_rows)
        if fact_rows
        else '<tr><td colspan="9">No facts match the current filter.</td></tr>'
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ai-knot inspector</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f1ea;
      --panel: #fffdf8;
      --ink: #1d1b18;
      --muted: #6f665d;
      --line: #d6cec3;
      --accent: #9b3d1b;
      --accent-soft: #f3e3dc;
      --code: #f0ece5;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      background: radial-gradient(circle at top left, #fff7ed, var(--bg) 38%, #efe9de 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 32px 20px 48px;
    }}
    header {{
      display: grid;
      gap: 12px;
      margin-bottom: 24px;
    }}
    h1, h2 {{ margin: 0; }}
    h1 {{
      font-size: clamp(2rem, 3vw, 3rem);
      letter-spacing: -0.03em;
    }}
    p {{ margin: 0; color: var(--muted); }}
    .hero {{
      display: grid;
      gap: 18px;
      padding: 24px;
      border: 1px solid var(--line);
      border-radius: 22px;
      background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(249,241,233,0.92));
      box-shadow: 0 18px 50px rgba(29, 27, 24, 0.08);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
    }}
    .card, section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 8px 24px rgba(29, 27, 24, 0.05);
    }}
    .card {{
      padding: 14px 16px;
    }}
    .card strong {{
      display: block;
      font-size: 1.5rem;
      color: var(--accent);
    }}
    section {{
      padding: 20px;
      margin-top: 18px;
    }}
    form {{
      display: grid;
      grid-template-columns: 2fr repeat(3, minmax(120px, 1fr)) auto auto;
      gap: 10px;
      align-items: end;
    }}
    label {{
      display: grid;
      gap: 6px;
      font-size: 0.95rem;
      color: var(--muted);
    }}
    input[type="text"],
    input[type="datetime-local"],
    input[type="number"] {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      background: #fff;
      color: var(--ink);
    }}
    input[type="checkbox"] {{
      margin-right: 6px;
    }}
    .checkbox {{
      display: flex;
      align-items: center;
      height: 42px;
      color: var(--muted);
    }}
    button, .link-button {{
      border: 0;
      border-radius: 999px;
      padding: 10px 16px;
      font: inherit;
      cursor: pointer;
      text-decoration: none;
      text-align: center;
    }}
    button {{
      background: var(--accent);
      color: #fffaf5;
    }}
    .link-button {{
      background: var(--accent-soft);
      color: var(--accent);
    }}
    pre, code {{
      font-family: "SFMono-Regular", "SF Mono", Menlo, Consolas, monospace;
      background: var(--code);
      border-radius: 10px;
    }}
    pre {{
      padding: 14px;
      overflow-x: auto;
      white-space: pre-wrap;
      margin: 12px 0 0;
      color: var(--ink);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 12px;
      border-top: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      font-size: 0.85rem;
      letter-spacing: 0.03em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    tbody tr:hover {{
      background: rgba(155, 61, 27, 0.04);
    }}
    ul {{
      margin: 12px 0 0;
      padding-left: 22px;
      color: var(--ink);
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      font-size: 0.95rem;
      color: var(--muted);
    }}
    @media (max-width: 900px) {{
      form {{
        grid-template-columns: 1fr;
      }}
      .checkbox {{
        height: auto;
      }}
    }}
  </style>
</head>
<body>
  <main>
    <header class="hero">
      <div>
        <p>ai-knot browser inspector</p>
        <h1>Deterministic memory, in a page.</h1>
      </div>
      <div class="meta">
        <span><strong>Agent:</strong> <code>{escape(agent_id)}</code></span>
        <span><strong>Version:</strong> <code>{escape(__version__)}</code></span>
        <span><strong>Auth:</strong> {escape(token_note)}</span>
        <span><strong>Anchor:</strong> <code>{escape(_fmt_dt(anchor))}</code></span>
      </div>
      <div class="stats">
        <div class="card"><span>Active facts</span><strong>{stats["total_facts"]}</strong></div>
        <div class="card"><span>Semantic</span><strong>{stats["by_type"]["semantic"]}</strong></div>
        <div class="card">
          <span>Procedural</span><strong>{stats["by_type"]["procedural"]}</strong>
        </div>
        <div class="card"><span>Episodic</span><strong>{stats["by_type"]["episodic"]}</strong></div>
      </div>
    </header>

    <section>
      <h2>Recall test</h2>
      <p>
        Use the same retrieval path your agent uses, then inspect the underlying stored facts
        below.
      </p>
      <form method="get" action="/inspect">
        <label>Query
          <input
            type="text"
            name="q"
            value="{escape(query)}"
            placeholder="what stack does the user use?"
          >
        </label>
        <label>Top k
          <input type="number" name="top_k" min="1" max="{_MAX_TOP_K}" value="{top_k}">
        </label>
        <label>Anchor
          <input
            type="text"
            name="now"
            value="{escape(now)}"
            placeholder="2026-07-01T12:00:00+00:00"
          >
        </label>
        <label>List limit
          <input type="number" name="limit" min="1" max="{_MAX_LIST_LIMIT}" value="{limit}">
        </label>
        <label class="checkbox">
          <input type="checkbox" name="include_inactive" value="true"{checked}>
          Include inactive facts
        </label>
        <button type="submit">Refresh</button>
        <a class="link-button" href="/inspect">Reset</a>
      </form>
    </section>

    {recall_block}

    <section>
      <h2>Stored facts</h2>
      <p>Showing {len(listed_facts)} of {total_matching_facts} matching facts, newest first.</p>
      <table>
        <thead>
          <tr>
            <th>Status</th>
            <th>Type</th>
            <th>ID</th>
            <th>Content</th>
            <th>Importance</th>
            <th>Tags</th>
            <th>Created</th>
            <th>Event time</th>
            <th>Valid until</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </section>
  </main>
</body>
</html>
"""


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

    @app.get("/v1/facts")
    def list_facts(
        limit: int = Query(default=_DEFAULT_LIST_LIMIT, ge=1, le=_MAX_LIST_LIMIT),
        include_inactive: bool = False,
        now: str | None = None,
        _: None = Depends(require_auth),
    ) -> dict[str, Any]:
        anchor = _parse_dt(now) or datetime.now(UTC)
        facts = _matching_facts(kb, now_dt=anchor, include_inactive=include_inactive)
        listed = facts[:limit]
        return {
            "facts": [_serialize_fact(fact, active_at=anchor) for fact in listed],
            "returned": len(listed),
            "total_matching": len(facts),
            "include_inactive": include_inactive,
            "now": anchor.isoformat(),
        }

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

    @app.get("/inspect", response_class=HTMLResponse)
    def inspect(
        q: str = "",
        top_k: int = Query(default=5, ge=1, le=_MAX_TOP_K),
        now: str = "",
        include_inactive: bool = False,
        limit: int = Query(default=_DEFAULT_LIST_LIMIT, ge=1, le=_MAX_LIST_LIMIT),
        _: None = Depends(require_auth),
    ) -> HTMLResponse:
        anchor = _parse_dt(now) or datetime.now(UTC)
        facts = _matching_facts(kb, now_dt=anchor, include_inactive=include_inactive)
        listed = facts[:limit]
        query = q.strip()
        recalled_facts = kb.recall_facts(query, top_k=top_k, now=anchor) if query else []
        recall_context = kb.recall(query, top_k=top_k, now=anchor) if query else ""
        return HTMLResponse(
            _render_inspector_html(
                agent_id=getattr(kb, "_agent_id", "unknown"),
                token_enabled=token is not None,
                query=query,
                top_k=top_k,
                now=now,
                include_inactive=include_inactive,
                limit=limit,
                stats=kb.stats(),
                listed_facts=listed,
                total_matching_facts=len(facts),
                recalled_facts=recalled_facts,
                recall_context=recall_context,
                anchor=anchor,
            )
        )

    return app
