"""LOCOMO benchmark adapter for ai-knot v2.

Bridges the existing aiknotbench TypeScript runner with the v2 MemoryAPI.
This module provides the Python-side adapter that can be called from
the MCP server or directly.

Usage (Python):
    adapter = LocomoAdapter()
    results = adapter.run_conversation(conv_data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LocomoTurn:
    speaker: str  # "human" | "ai"
    text: str
    timestamp: int = 0


@dataclass
class LocomoQuestion:
    question_id: str
    question: str
    gold_answer: str
    category: int  # 1-4: single-hop, multi-hop, temporal, adversarial
    evidence_turn_ids: list[str] = field(default_factory=list)


@dataclass
class LocomoConversation:
    conv_id: str
    turns: list[LocomoTurn]
    questions: list[LocomoQuestion]


@dataclass
class LocomoResult:
    question_id: str
    question: str
    gold_answer: str
    predicted_answer: str
    correct: bool
    category: int
    atoms_used: int
    evidence_pack_id: str


class LocomoAdapter:
    """Adapter between LOCOMO conversation format and ai-knot v2 MemoryAPI.

    Handles:
    - Converting LOCOMO turns to RawEpisodes
    - Answering LOCOMO questions via recall
    - Scoring results

    Note: Answer generation (LLM call) is outside this module.
    This module handles memory operations only.
    """

    def __init__(self, api: Any | None = None) -> None:
        if api is None:
            from ai_knot_v2.api.product import MemoryAPI

            self._api = MemoryAPI(db_path=":memory:")
        else:
            self._api = api

    def ingest_conversation(self, conv: LocomoConversation) -> dict[str, Any]:
        """Learn all turns in a LOCOMO conversation. Returns LearnResponse dict."""
        from ai_knot_v2.api.sdk import EpisodeIn, LearnRequest

        episodes = []
        for i, turn in enumerate(conv.turns):
            episodes.append(
                EpisodeIn(
                    text=turn.text,
                    speaker="user" if turn.speaker == "human" else "agent",
                    session_id=conv.conv_id,
                    timestamp=turn.timestamp or (1_700_000_000 + i * 60),
                )
            )

        resp = self._api.learn(LearnRequest(episodes=episodes))
        return resp.model_dump()

    def answer_question(self, question: LocomoQuestion) -> dict[str, Any]:
        """Recall relevant atoms for a LOCOMO question. Returns recall data."""
        from ai_knot_v2.api.sdk import RecallRequest

        resp = self._api.recall(RecallRequest(query=question.question))
        return {
            "question_id": question.question_id,
            "atoms": [a.model_dump() for a in resp.atoms],
            "evidence_pack_id": resp.evidence_pack_id,
            "intervention_variable": resp.intervention_variable,
        }

    def run_conversation(
        self,
        conv: LocomoConversation,
    ) -> list[dict[str, Any]]:
        """Full pipeline: ingest all turns, answer all questions."""
        self.ingest_conversation(conv)
        results = []
        for q in conv.questions:
            recall_data = self.answer_question(q)
            results.append(recall_data)
        return results

    def reset(self) -> None:
        """Reset the API (clear memory) for a fresh conversation."""
        from ai_knot_v2.api.product import MemoryAPI

        self._api = MemoryAPI(db_path=":memory:")


def parse_locomo_jsonl(jsonl_path: str) -> list[LocomoConversation]:
    """Parse LOCOMO JSONL format into LocomoConversation objects.

    Expected format per line: {conv_id, turns: [{speaker, text}], questions: [{...}]}
    """
    import json
    from pathlib import Path

    conversations: list[LocomoConversation] = []
    path = Path(jsonl_path)
    if not path.exists():
        return conversations

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            turns = [
                LocomoTurn(
                    speaker=t.get("speaker", "human"),
                    text=t.get("text", ""),
                    timestamp=t.get("timestamp", 0),
                )
                for t in data.get("turns", [])
            ]
            questions = [
                LocomoQuestion(
                    question_id=q.get("id", str(i)),
                    question=q.get("question", ""),
                    gold_answer=q.get("answer", ""),
                    category=q.get("category", 1),
                    evidence_turn_ids=q.get("evidence_turns", []),
                )
                for i, q in enumerate(data.get("questions", []))
            ]
            conversations.append(
                LocomoConversation(
                    conv_id=data.get("conv_id", str(len(conversations))),
                    turns=turns,
                    questions=questions,
                )
            )

    return conversations
