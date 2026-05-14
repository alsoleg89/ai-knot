from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from deep_research.corpus import Corpus
from deep_research.llm import LLMClient

if TYPE_CHECKING:
    from deep_research.memory import SemanticMemory


@dataclass
class RoleContext:
    tick: int
    focus: str
    corpus: Corpus
    phase: str = "generate"
    semantic: SemanticMemory | None = field(default=None, repr=False)

    def recall(self, query: str, k: int = 5, stream: str | None = None) -> list[dict[str, Any]]:
        if self.semantic is None:
            return []
        return self.semantic.recall(query, k=k, stream=stream)


@dataclass
class RoleOutput:
    role_name: str
    summary: str
    tokens_used: int
    data: dict[str, Any]


class BaseRole(ABC):
    name: str = ""

    def __init__(self, llm: LLMClient) -> None:
        self.llm: LLMClient = llm

    @abstractmethod
    def run(self, ctx: RoleContext) -> RoleOutput: ...
