"""One typed invocation boundary for HTTP, MCP and CLI adapters."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .always_on.codec import decode_arguments, encode_always_on

if TYPE_CHECKING:
    from .always_on.memory import AlwaysOnMemory

TOOL_METHODS = {
    "intent_create": "create_intent",
    "intent_get": "get_intent",
    "intent_list": "list_intents",
    "intent_claim": "claim",
    "intent_renew": "renew",
    "intent_checkpoint": "checkpoint",
    "intent_resume": "resume",
    "memory_evidence_capture": "capture_evidence",
    "memory_subscribe": "subscribe",
    "memory_poll": "poll",
    "memory_ack": "ack",
    "memory_unsubscribe": "unsubscribe",
    "action_prepare": "prepare_action",
    "action_begin": "begin_action",
    "action_finish": "finish_action",
    "action_reconcile": "reconcile_action",
    "memory_freshness_set": "set_freshness",
    "memory_freshness_due": "due_verifications",
    "memory_freshness_record": "record_verification",
    "memory_maintenance": "run_maintenance",
}


async def invoke_always_on(memory: AlwaysOnMemory, method: str, payload: object) -> Any:
    if method == "run_maintenance":
        from .always_on.maintenance import MaintenanceBudget, run_maintenance

        def request(*, run_key: str, budget: MaintenanceBudget) -> None:
            pass

        request.__annotations__["budget"] = MaintenanceBudget
        args = decode_arguments(request, payload)
        return encode_always_on(await run_maintenance(memory, **args))
    function = getattr(memory, method)
    return encode_always_on(function(**decode_arguments(function, payload)))


def register_always_on_tools(app: Any, memory: AlwaysOnMemory) -> None:
    """Expose explicit tool schemas while retaining the strict shared decoder."""
    import inspect
    import json

    from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata

    from .always_on.codec import error_response

    class StrictMetadata(FuncMetadata):
        async def call_fn_with_arg_validation(
            self,
            fn: Any,
            fn_is_async: bool,
            arguments_to_validate: dict[str, Any],
            arguments_to_pass_directly: dict[str, Any] | None,
        ) -> Any:
            # MCP's default parser coerces strings and drops unknown arguments.
            # The shared boundary must see exactly what the host supplied.
            return await fn(**arguments_to_validate)

    def make_tool(method: str) -> Any:
        async def tool(**arguments: Any) -> str:
            try:
                result = await invoke_always_on(memory, method, arguments)
            except Exception as exc:
                status, result = error_response(exc)
                if status == 422 and not isinstance(exc, (ValueError, TypeError)):
                    raise
            return json.dumps(result, ensure_ascii=False, allow_nan=False)

        if method == "run_maintenance":
            parameters = [
                inspect.Parameter("run_key", inspect.Parameter.KEYWORD_ONLY, annotation=Any),
                inspect.Parameter("budget", inspect.Parameter.KEYWORD_ONLY, annotation=Any),
            ]
        else:
            parameters = [
                parameter.replace(annotation=Any)
                for parameter in inspect.signature(getattr(memory, method)).parameters.values()
            ]
        tool.__signature__ = inspect.Signature(parameters, return_annotation=str)  # type: ignore[attr-defined]
        return tool

    for name, method in TOOL_METHODS.items():
        function = make_tool(method)
        app.add_tool(
            function,
            name=name,
            description=f"Durable {method}; requires SQLite or PostgreSQL.",
            structured_output=False,
        )
        registered = app._tool_manager.get_tool(name)
        registered.fn_metadata = StrictMetadata(**registered.fn_metadata.model_dump())
