"""Core data types for agentmemo."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4


class MemoryType(str, Enum):
    """Classification of stored knowledge.

    SEMANTIC — facts about the world or user ("works at Sber").
    PROCEDURAL — how the user wants things done ("always use type hints").
    EPISODIC — specific past events ("deploy failed last Tuesday").
    """

    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    EPISODIC = "episodic"


@dataclass
class Fact:
    """A single unit of knowledge extracted from or added to an agent's memory.

    Attributes:
        content: The human-readable knowledge string.
        type: Classification — semantic, procedural, or episodic.
        importance: How critical this fact is (0.0-1.0). Higher = remembered longer.
        retention_score: Current memory strength after Ebbinghaus decay (0.0-1.0).
        access_count: Number of times this fact was retrieved via recall().
        tags: Optional labels for organization.
        id: Unique 8-char hex identifier.
        created_at: When the fact was first stored (UTC).
        last_accessed: When the fact was last retrieved (UTC).
    """

    content: str
    type: MemoryType = MemoryType.SEMANTIC
    importance: float = 0.8
    retention_score: float = 1.0
    access_count: int = 0
    tags: list[str] = field(default_factory=list)
    id: str = field(default_factory=lambda: uuid4().hex[:8])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ConversationTurn:
    """A single message in a conversation.

    Attributes:
        role: "user" or "assistant".
        content: The message text.
    """

    role: str
    content: str
