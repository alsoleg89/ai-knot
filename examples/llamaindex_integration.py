"""LlamaIndex integration example.

Shows how to add ai-knot long-term memory to a real LlamaIndex chat-engine run
through the native ``memory=...`` seam. Requires ``llama-index-core`` plus an
LLM integration package for an actual networked run.

Run::

    pip install "ai-knot[llamaindex]" "llama-index-llms-openai"
    OPENAI_API_KEY=... python examples/llamaindex_integration.py
"""

from __future__ import annotations

import os
from tempfile import TemporaryDirectory
from typing import Any

from ai_knot import KnowledgeBase
from ai_knot.integrations.llamaindex import AiKnotLlamaIndexMemory
from ai_knot.storage import YAMLStorage

_INSTALL_HINT = (
    'Install LlamaIndex first: pip install "ai-knot[llamaindex]" '
    '"llama-index-llms-openai"'
)


def build_chat_engine(tmpdir: str) -> tuple[Any, KnowledgeBase]:
    try:
        from llama_index.core.chat_engine import SimpleChatEngine
        from llama_index.llms.openai import OpenAI
    except ImportError as exc:  # pragma: no cover - example only
        raise SystemExit(_INSTALL_HINT) from exc

    kb = KnowledgeBase(
        agent_id="llamaindex-demo",
        storage=YAMLStorage(base_dir=tmpdir),
    )
    kb.add("User prefers Python over Java")
    kb.add("User deploys APIs with Docker Compose")
    kb.add("Release notes should stay concise and numbered")

    memory = AiKnotLlamaIndexMemory.from_defaults(knowledge_base=kb, top_k=4)
    llm = OpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    chat_engine = SimpleChatEngine.from_defaults(
        llm=llm,
        memory=memory,
        system_prompt="You are a concise senior backend engineer.",
    )
    return chat_engine, kb


def main() -> None:
    with TemporaryDirectory(prefix="ai-knot-llamaindex-real-") as tmpdir:
        chat_engine, kb = build_chat_engine(tmpdir)
        response = chat_engine.chat("Write a local deployment checklist for my API stack.")

        print("=== LlamaIndex + ai-knot ===")
        print(response)
        print()
        print("Stored facts:")
        for fact in kb.list():
            print(f"  - [{fact.id}] {fact.content}")


if __name__ == "__main__":
    main()
