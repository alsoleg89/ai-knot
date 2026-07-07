"""OpenAI Agents SDK adapter.

ai-knot complements the OpenAI Agents SDK's short-term/session memory with a
distilled, long-term fact store. The adapter builds a ``RunConfig`` using the
SDK's ``call_model_input_filter`` hook, recalling relevant facts from
``KnowledgeBase`` and appending them to the model instructions immediately
before a model call.

There is **no hard dependency** on ``openai-agents``. Importing this module is
safe without the SDK installed; the import is required only when building a real
``RunConfig`` or when the returned filter is invoked by the SDK.

Example::

    from ai_knot import KnowledgeBase
    from ai_knot.integrations.openai_agents import AiKnotAgentsMemory

    kb = KnowledgeBase("assistant")
    kb.add("User prefers Python and deploys with Docker")

    memory = AiKnotAgentsMemory(kb)
    run_config = memory.build_run_config()

    # Then pass run_config into Runner.run(...) / Runner.run_sync(...).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ai_knot.knowledge import KnowledgeBase


def _text_from_node(node: object) -> str:
    """Extract text from a Responses-style content node."""
    if isinstance(node, str):
        return node
    if isinstance(node, dict):
        text = node.get("text")
        if isinstance(text, str):
            return text
        if isinstance(text, dict):
            for key in ("value", "text"):
                value = text.get(key)
                if isinstance(value, str):
                    return value
        content = node.get("content")
        if content is not None:
            return _text_from_content(content)
    return ""


def _text_from_content(content: object) -> str:
    """Extract text from a Responses-style message content payload."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [_text_from_node(item) for item in content]
        return " ".join(part.strip() for part in parts if part and part.strip()).strip()
    return ""


def _normalise_input_items(items: object) -> list[Any]:
    """Normalise model input items to a list for the SDK hook."""
    if items is None:
        return []
    if isinstance(items, list):
        return list(items)
    if isinstance(items, tuple):
        return list(items)
    return [items]


class AiKnotAgentsMemory:
    """Long-term memory adapter for the OpenAI Agents SDK.

    The adapter does not replace the SDK's own session/history primitives. It
    adds a *second* layer: distilled facts recalled from ai-knot and injected
    into model instructions immediately before each model call.

    Args:
        knowledge_base: The KnowledgeBase to query for long-term memory.
        top_k: Maximum number of facts to recall per turn.
        heading: Markdown heading used for the injected memory block.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        *,
        top_k: int = 5,
        heading: str = "## Agent Memory",
    ) -> None:
        self._kb = knowledge_base
        self._top_k = top_k
        self._heading = heading

    def extract_query(self, input_items: Sequence[object]) -> str:
        """Return the latest user text from Responses-style input items."""
        for item in reversed(input_items):
            if not isinstance(item, dict):
                continue
            if item.get("role") != "user":
                continue
            text = _text_from_content(item.get("content"))
            if text:
                return text
        return ""

    def augment_instructions(self, instructions: str | None, query: str) -> str | None:
        """Append recalled memory to *instructions* when query-relevant facts exist."""
        if not query:
            return instructions
        context = self._kb.recall(query, top_k=self._top_k)
        if not context:
            return instructions
        memory_block = f"{self._heading}\n{context}"
        if instructions and instructions.strip():
            return f"{instructions.rstrip()}\n\n{memory_block}"
        return memory_block

    def build_call_model_input_filter(
        self,
        existing_filter: Callable[[Any], Any] | None = None,
    ) -> Callable[[Any], Any]:
        """Return a ``call_model_input_filter`` for the OpenAI Agents SDK.

        If *existing_filter* is provided, it runs first; ai-knot then appends
        memory to the resulting instructions while preserving the filtered input.
        """

        def filter_fn(data: Any) -> Any:
            try:
                from agents.run import ModelInputData
            except ImportError as exc:  # pragma: no cover - exercised via explicit import test
                raise ImportError(
                    'OpenAI Agents SDK is required. Install with: pip install "ai-knot[agents]" '
                    "(or pip install openai-agents)"
                ) from exc

            model_data = existing_filter(data) if existing_filter else data.model_data
            input_items = _normalise_input_items(getattr(model_data, "input", None))
            instructions = getattr(model_data, "instructions", None)
            query = self.extract_query(input_items)
            return ModelInputData(
                input=input_items,
                instructions=self.augment_instructions(instructions, query),
            )

        return filter_fn

    def build_run_config(self, **kwargs: Any) -> Any:
        """Return an SDK ``RunConfig`` wired with ai-knot memory injection."""
        try:
            from agents import RunConfig
        except ImportError as exc:  # pragma: no cover - exercised via explicit import test
            raise ImportError(
                'OpenAI Agents SDK is required. Install with: pip install "ai-knot[agents]" '
                "(or pip install openai-agents)"
            ) from exc

        existing_filter = kwargs.pop("call_model_input_filter", None)
        return RunConfig(
            call_model_input_filter=self.build_call_model_input_filter(existing_filter),
            **kwargs,
        )
