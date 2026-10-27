"""Optional configured-owner always-on HTTP routes."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, Response

from ai_knot._always_on_tools import invoke_always_on
from ai_knot.always_on.codec import error_response, loads_strict

if TYPE_CHECKING:
    from ai_knot.always_on.memory import AlwaysOnMemory

ROUTES = (
    ("POST", "/intents", "create_intent", None),
    ("GET", "/intents", "list_intents", None),
    ("GET", "/intents/{intent_id}", "get_intent", "intent_id"),
    ("POST", "/intents/{intent_id}/claim", "claim", "intent_id"),
    ("POST", "/intents/{intent_id}/renew", "renew", "intent_id"),
    ("POST", "/intents/{intent_id}/checkpoint", "checkpoint", "intent_id"),
    ("POST", "/intents/{intent_id}/resume", "resume", "intent_id"),
    ("POST", "/subscriptions", "subscribe", None),
    ("POST", "/subscriptions/{subscription_id}/poll", "poll", "subscription_id"),
    ("POST", "/subscriptions/{subscription_id}/ack", "ack", "subscription_id"),
    ("DELETE", "/subscriptions/{subscription_id}", "unsubscribe", "subscription_id"),
    ("POST", "/actions", "prepare_action", None),
    ("POST", "/actions/{action_id}/begin", "begin_action", "action_id"),
    ("POST", "/actions/{action_id}/finish", "finish_action", "action_id"),
    ("POST", "/actions/{action_id}/reconcile", "reconcile_action", "action_id"),
    ("POST", "/evidence/capture", "capture_evidence", None),
    ("POST", "/freshness", "set_freshness", None),
    ("GET", "/freshness/due", "due_verifications", None),
    ("POST", "/freshness/record", "record_verification", None),
    ("POST", "/maintenance/run", "run_maintenance", None),
)


def build_always_on_router(memory: AlwaysOnMemory, require_auth: Callable[..., Any]) -> APIRouter:
    router = APIRouter(prefix="/v1/always-on", dependencies=[Depends(require_auth)])

    def endpoint(method: str, identifier: str | None) -> Callable[..., Any]:
        async def handle(request: Request) -> Response:
            try:
                if request.method == "GET":
                    query: dict[str, Any] = dict(request.query_params)
                    if "limit" in query:
                        query["limit"] = int(query["limit"])
                    payload: object = query
                else:
                    raw = await request.body()
                    payload = loads_strict(raw) if raw else {}
                if not isinstance(payload, dict):
                    raise ValueError("Expected object.")
                if identifier:
                    if identifier in payload:
                        raise ValueError("Identifier belongs in URL.")
                    payload[identifier] = request.path_params[identifier]
                result = await invoke_always_on(memory, method, payload)
                if method in ("ack", "unsubscribe"):
                    return Response(status_code=204)
                return JSONResponse(result)
            except (ValueError, TypeError) as exc:
                status, body = error_response(exc)
                return JSONResponse(body, status_code=status)
            except Exception as exc:
                status, body = error_response(exc)
                if status == 422:
                    raise
                return JSONResponse(body, status_code=status)

        return handle

    for verb, path, method, identifier in ROUTES:
        router.add_api_route(path, endpoint(method, identifier), methods=[verb], name=method)
    return router
