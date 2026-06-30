"""LangChain / LangGraph memory — ai-knot as a retriever and as chat memory.

Shows the two thin adapters in ``ai_knot.integrations.langchain``. Neither
requires LangChain to be installed: with ``langchain_core`` present the retriever
yields real ``Document`` objects, otherwise a shim with the same attributes — so
this example runs offline with only ai-knot installed, and recall never calls an LLM.

Run::

    python examples/langchain_integration.py
"""

from __future__ import annotations

import shutil

from ai_knot import KnowledgeBase
from ai_knot.integrations.langchain import AiKnotChatMemory, AiKnotRetriever


def retriever_demo() -> None:
    kb = KnowledgeBase(agent_id="lc_retriever_demo")
    kb.add("User is a senior backend developer at Acme Corp")
    kb.add("User writes Go and avoids Java")
    kb.add("User deploys services with Docker and Kubernetes")
    kb.add("Team standup is at 10am")

    retriever = AiKnotRetriever(kb, top_k=3)

    print("=== As a LangChain retriever ===")
    print("Query: 'what language and tooling does the user use?'\n")
    for doc in retriever.invoke("what language and tooling does the user use?"):
        print(f"  • {doc.page_content}   (score={doc.metadata['score']})")
    print("\n(The standup fact is left behind — only relevant facts are retrieved.)\n")

    shutil.rmtree(".ai_knot/lc_retriever_demo", ignore_errors=True)


def chat_memory_demo() -> None:
    kb = KnowledgeBase(agent_id="lc_memory_demo")
    memory = AiKnotChatMemory(kb)  # memory_key="history" by default

    print("=== As drop-in conversational memory (BaseChatMemory shape) ===")
    # Each user turn is distilled into a stored fact.
    memory.save_context({"input": "I deploy everything in Docker"}, {"output": "Noted."})
    memory.save_context({"input": "I prefer Python over Java"}, {"output": "Got it."})

    # A later turn recalls only the facts relevant to the current input.
    variables = memory.load_memory_variables({"input": "how should I write the deploy script?"})
    print("Injected into prompt under 'history':")
    print(variables["history"] or "  (nothing relevant)")
    print()

    shutil.rmtree(".ai_knot/lc_memory_demo", ignore_errors=True)


if __name__ == "__main__":
    retriever_demo()
    chat_memory_demo()
    print("Demo complete.")
