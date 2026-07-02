"""PydanticAI adapter.

ai-knot complements PydanticAI's per-run/chat history with a distilled,
long-term fact store. The adapter uses **runtime instructions** on each run so
the recalled memory stays query-aware: only the facts relevant to the current
user prompt are appended.

There is **no hard dependency** on ``pydantic-ai``. Importing this module is
safe without the framework installed; the adapter only expects an object with
``run`` / ``run_sync`` / ``run_stream``-style methods that accept an
``instructions=...`` keyword argument, which matches PydanticAI's official API.

Example::

    from pydantic_ai import Agent

    from ai_knot import KnowledgeBase
    from ai_knot.integrations.pydanticai import AiKnotPydanticAIMemory

    kb = KnowledgeBase("assistant")
    kb.add("User prefers Python")
    kb.add("User deploys with Docker Compose")

    agent = Agent(
        "openai:gpt-5.2",
        instructions="You are a concise staff engineer.",
    )
    memory = AiKnotPydanticAIMemory(kb)

    result = memory.run_sync(agent, "Write a local deployment checklist.")
    print(result.output)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, TypeVar

from ai_knot.knowledge import KnowledgeBase

_T_co = TypeVar("_T_co", covariant=True)
RuntimeInstructions = str | Sequence[str] | None


class _SupportsRunSync(Protocol[_T_co]):
    def run_sync(self, user_prompt: str, /, **kwargs: Any) -> _T_co: ...


class _SupportsRun(Protocol[_T_co]):
    async def run(self, user_prompt: str, /, **kwargs: Any) -> _T_co: ...


class _SupportsRunStream(Protocol[_T_co]):
    def run_stream(self, user_prompt: str, /, **kwargs: Any) -> _T_co: ...


class _SupportsRunStreamSync(Protocol[_T_co]):
    def run_stream_sync(self, user_prompt: str, /, **kwargs: Any) -> _T_co: ...


class AiKnotPydanticAIMemory:
    """Long-term memory adapter for PydanticAI agents.

    The adapter does not replace PydanticAI's own message history. It adds a
    second layer: long-term facts recalled from ai-knot and appended through the
    framework's per-run ``instructions=...`` surface.

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

    def augment_instructions(
        self,
        instructions: RuntimeInstructions,
        query: str,
        *,
        top_k: int | None = None,
        now: Any = None,
    ) -> RuntimeInstructions:
        """Append recalled memory to a runtime ``instructions=...`` payload."""
        if not query.strip():
            return instructions

        context = self._kb.recall(query, top_k=top_k or self._top_k, now=now)
        if not context:
            return instructions

        memory_block = f"{self._heading}\n{context}"
        if instructions is None:
            return memory_block
        if isinstance(instructions, str):
            return f"{instructions.rstrip()}\n\n{memory_block}"
        return [*instructions, memory_block]

    def run_sync(
        self,
        agent: _SupportsRunSync[_T_co],
        user_prompt: str,
        /,
        *,
        instructions: RuntimeInstructions = None,
        top_k: int | None = None,
        now: Any = None,
        **kwargs: Any,
    ) -> _T_co:
        """Call ``agent.run_sync(...)`` with ai-knot memory in runtime instructions."""
        return agent.run_sync(
            user_prompt,
            instructions=self.augment_instructions(
                instructions,
                user_prompt,
                top_k=top_k,
                now=now,
            ),
            **kwargs,
        )

    async def run(
        self,
        agent: _SupportsRun[_T_co],
        user_prompt: str,
        /,
        *,
        instructions: RuntimeInstructions = None,
        top_k: int | None = None,
        now: Any = None,
        **kwargs: Any,
    ) -> _T_co:
        """Call ``agent.run(...)`` with ai-knot memory in runtime instructions."""
        return await agent.run(
            user_prompt,
            instructions=self.augment_instructions(
                instructions,
                user_prompt,
                top_k=top_k,
                now=now,
            ),
            **kwargs,
        )

    def run_stream(
        self,
        agent: _SupportsRunStream[_T_co],
        user_prompt: str,
        /,
        *,
        instructions: RuntimeInstructions = None,
        top_k: int | None = None,
        now: Any = None,
        **kwargs: Any,
    ) -> _T_co:
        """Call ``agent.run_stream(...)`` with ai-knot memory in runtime instructions."""
        return agent.run_stream(
            user_prompt,
            instructions=self.augment_instructions(
                instructions,
                user_prompt,
                top_k=top_k,
                now=now,
            ),
            **kwargs,
        )

    def run_stream_sync(
        self,
        agent: _SupportsRunStreamSync[_T_co],
        user_prompt: str,
        /,
        *,
        instructions: RuntimeInstructions = None,
        top_k: int | None = None,
        now: Any = None,
        **kwargs: Any,
    ) -> _T_co:
        """Call ``agent.run_stream_sync(...)`` with ai-knot memory in runtime instructions."""
        return agent.run_stream_sync(
            user_prompt,
            instructions=self.augment_instructions(
                instructions,
                user_prompt,
                top_k=top_k,
                now=now,
            ),
            **kwargs,
        )
