"""Third-party integrations for ai_knot."""

from __future__ import annotations

from ai_knot.integrations.autogen import AiKnotAutoGenMemory
from ai_knot.integrations.crewai import AiKnotCrewAIMemory
from ai_knot.integrations.langchain import (
    AiKnotChatMemory,
    AiKnotRetriever,
    create_add_memory_function,
    create_add_memory_tool,
    create_basic_memory_functions,
    create_basic_memory_tools,
    create_delete_memory_function,
    create_delete_memory_tool,
    create_get_memory_function,
    create_get_memory_tool,
    create_list_memory_function,
    create_list_memory_tool,
    create_manage_memory_function,
    create_manage_memory_tool,
    create_search_memory_function,
    create_search_memory_tool,
)
from ai_knot.integrations.llamaindex import AiKnotLlamaIndexMemory
from ai_knot.integrations.openai import MemoryEnabledOpenAI
from ai_knot.integrations.openai_agents import AiKnotAgentsMemory
from ai_knot.integrations.pydanticai import AiKnotPydanticAIMemory
from ai_knot.integrations.semantic_resolver_llm import LLMSemanticConflictResolver

__all__ = [
    "AiKnotAutoGenMemory",
    "AiKnotAgentsMemory",
    "AiKnotChatMemory",
    "AiKnotCrewAIMemory",
    "AiKnotLlamaIndexMemory",
    "AiKnotPydanticAIMemory",
    "AiKnotRetriever",
    "LLMSemanticConflictResolver",
    "MemoryEnabledOpenAI",
    "create_add_memory_function",
    "create_add_memory_tool",
    "create_basic_memory_functions",
    "create_basic_memory_tools",
    "create_delete_memory_function",
    "create_delete_memory_tool",
    "create_get_memory_function",
    "create_get_memory_tool",
    "create_list_memory_function",
    "create_list_memory_tool",
    "create_manage_memory_function",
    "create_manage_memory_tool",
    "create_search_memory_function",
    "create_search_memory_tool",
]
