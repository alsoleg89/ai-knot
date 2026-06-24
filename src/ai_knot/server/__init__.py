"""Optional HTTP sidecar for ai-knot.

A thin FastAPI surface over :class:`~ai_knot.knowledge.KnowledgeBase` for clients
that cannot speak MCP over stdio. Requires the ``server`` extra::

    pip install "ai-knot[server]"

Run it with ``ai-knot serve`` (see :mod:`ai_knot.cli`) or build the app yourself::

    from ai_knot import KnowledgeBase
    from ai_knot.server import create_app

    app = create_app(KnowledgeBase(agent_id="svc"))
"""

from __future__ import annotations

from ai_knot.server.app import create_app

__all__ = ["create_app"]
