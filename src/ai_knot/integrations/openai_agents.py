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

from ai_knot.integrations._task_context import task_query, user_texts
from ai_knot.knowledge import KnowledgeBase


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

    def extract_query(
        self, input_items: Sequence[object], *, task_context: str | None = None
    ) -> str:
        """Build a bounded user-task query; standalone latest tasks win."""
        return task_query(user_texts(input_items), task_context=task_context)

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
        *,
        task_context: str | None = None,
    ) -> Callable[[Any], Any]:
        """Return a ``call_model_input_filter`` for the OpenAI Agents SDK.

        If *existing_filter* is provided, it runs first; ai-knot then appends
        memory to the resulting instructions while preserving the filtered input.
        ``task_context`` is an optional per-run retrieval hint for follow-ups;
        build a fresh filter when the host task changes.
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
            query = self.extract_query(input_items, task_context=task_context)
            return ModelInputData(
                input=input_items,
                instructions=self.augment_instructions(instructions, query),
            )

        return filter_fn

    def build_run_config(self, *, task_context: str | None = None, **kwargs: Any) -> Any:
        """Build a run config, consuming the optional host ``task_context`` hint.

        The hint is captured by this config's filter, not stored on the adapter
        or forwarded as an SDK field. Build a fresh config for a different task.
        """
        try:
            from agents import RunConfig
        except ImportError as exc:  # pragma: no cover - exercised via explicit import test
            raise ImportError(
                'OpenAI Agents SDK is required. Install with: pip install "ai-knot[agents]" '
                "(or pip install openai-agents)"
            ) from exc

        existing_filter = kwargs.pop("call_model_input_filter", None)
        return RunConfig(
            call_model_input_filter=self.build_call_model_input_filter(
                existing_filter, task_context=task_context
            ),
            **kwargs,
        )
