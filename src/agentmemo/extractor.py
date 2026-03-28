"""LLM-based fact extraction from conversations.

Supports OpenAI and Anthropic providers via httpx.
The LLM is instructed to return structured JSON with extracted facts.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import time
from typing import Any

import httpx

from agentmemo.types import ConversationTurn, Fact, MemoryType

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
# HTTP status codes that warrant a retry with backoff.
_RETRY_STATUS_CODES = {429, 500, 502, 503}

_EXTRACTION_SYSTEM_PROMPT = """You are a knowledge extraction engine.
Given a conversation, extract ONLY meaningful facts worth remembering.

Rules:
- Skip greetings, thanks, filler ("ok", "got it", "thanks").
- Each fact must be a single, self-contained statement.
- Classify each fact as: semantic (about user/world), procedural (preferences/how-to),
  episodic (specific events).
- Rate importance from 0.0 to 1.0.

Return a JSON array. Example:
[
  {"content": "User works at Sber", "type": "semantic", "importance": 0.9},
  {"content": "User prefers Python over Java", "type": "procedural", "importance": 0.85}
]

If no meaningful facts exist, return an empty array: []
Return ONLY the JSON array, no other text."""


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two strings (word-level)."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def deduplicate_facts(facts: list[Fact], *, threshold: float = 0.8) -> list[Fact]:
    """Remove near-duplicate facts by Jaccard word similarity.

    Args:
        facts: List of facts to deduplicate.
        threshold: Similarity threshold above which facts are considered duplicates.

    Returns:
        Deduplicated list, keeping the first occurrence.
    """
    if not facts:
        return []

    unique: list[Fact] = []
    for fact in facts:
        is_dup = False
        for existing in unique:
            if _jaccard_similarity(fact.content, existing.content) >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(fact)
    return unique


class Extractor:
    """Extract structured facts from conversations using an LLM.

    Args:
        api_key: API key for the LLM provider.
        provider: "openai" or "anthropic".
        model: Model name (defaults to provider-appropriate model).
    """

    def __init__(
        self,
        api_key: str,
        *,
        provider: str = "openai",
        model: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._provider = provider
        self._model = model or self._default_model()

    def _default_model(self) -> str:
        if self._provider == "anthropic":
            return "claude-haiku-4-5-20251001"
        return "gpt-4o-mini"

    def extract(self, turns: list[ConversationTurn]) -> list[Fact]:
        """Extract facts from a conversation.

        Args:
            turns: List of conversation messages.

        Returns:
            List of extracted Facts. Returns [] on any LLM error.
        """
        if not turns:
            return []

        raw_facts = self._call_llm(turns)
        facts = [self._parse_fact(entry) for entry in raw_facts if isinstance(entry, dict)]
        return deduplicate_facts(facts)

    def _call_llm(self, turns: list[ConversationTurn]) -> list[dict[str, Any]]:
        """Call the LLM to extract facts. Returns parsed JSON array."""
        conversation_text = "\n".join(f"{t.role}: {t.content}" for t in turns)

        if self._provider == "anthropic":
            return self._call_anthropic(conversation_text)
        return self._call_openai(conversation_text)

    def _call_openai(self, conversation_text: str) -> list[dict[str, Any]]:
        """Call OpenAI-compatible API with retry on transient errors."""
        for attempt in range(_MAX_RETRIES):
            try:
                response = httpx.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "messages": [
                            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
                            {"role": "user", "content": conversation_text},
                        ],
                        "temperature": 0.0,
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                return self._parse_json_response(content)
            except httpx.TimeoutException:
                logger.warning(
                    "OpenAI request timed out (attempt %d/%d)", attempt + 1, _MAX_RETRIES
                )
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in _RETRY_STATUS_CODES:
                    logger.warning(
                        "HTTP %d from OpenAI, will retry (attempt %d/%d)",
                        status,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                else:
                    logger.warning("HTTP %d from OpenAI — aborting extraction", status)
                    return []
            except httpx.RequestError as exc:
                logger.warning("OpenAI network error: %s", exc)
            except (KeyError, ValueError) as exc:
                logger.warning("Unexpected OpenAI response format: %s", exc)
                return []

            if attempt < _MAX_RETRIES - 1:
                wait = 2**attempt
                logger.info("Retrying in %ds…", wait)
                time.sleep(wait)

        logger.warning("OpenAI extraction failed after %d attempts", _MAX_RETRIES)
        return []

    def _call_anthropic(self, conversation_text: str) -> list[dict[str, Any]]:
        """Call Anthropic API with retry on transient errors."""
        for attempt in range(_MAX_RETRIES):
            try:
                response = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self._api_key,
                        "anthropic-version": "2023-06-01",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self._model,
                        "max_tokens": 2048,
                        "system": _EXTRACTION_SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": conversation_text}],
                    },
                    timeout=30.0,
                )
                response.raise_for_status()
                content = response.json()["content"][0]["text"]
                return self._parse_json_response(content)
            except httpx.TimeoutException:
                logger.warning(
                    "Anthropic request timed out (attempt %d/%d)", attempt + 1, _MAX_RETRIES
                )
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in _RETRY_STATUS_CODES:
                    logger.warning(
                        "HTTP %d from Anthropic, will retry (attempt %d/%d)",
                        status,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                else:
                    logger.warning("HTTP %d from Anthropic — aborting extraction", status)
                    return []
            except httpx.RequestError as exc:
                logger.warning("Anthropic network error: %s", exc)
            except (KeyError, ValueError) as exc:
                logger.warning("Unexpected Anthropic response format: %s", exc)
                return []

            if attempt < _MAX_RETRIES - 1:
                wait = 2**attempt
                logger.info("Retrying in %ds…", wait)
                time.sleep(wait)

        logger.warning("Anthropic extraction failed after %d attempts", _MAX_RETRIES)
        return []

    @staticmethod
    def _parse_json_response(content: str) -> list[dict[str, Any]]:
        """Parse a JSON array from LLM response, handling markdown fences."""
        content = content.strip()
        # Strip markdown code fences if present.
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.DOTALL)
        if match:
            content = match.group(1)
        try:
            parsed = json.loads(content)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse LLM response as JSON: %s", content[:200])
        return []

    @staticmethod
    def _parse_fact(entry: dict[str, Any]) -> Fact:
        """Convert a raw dict from LLM output into a Fact."""
        memory_type = MemoryType.SEMANTIC
        raw_type = entry.get("type", "semantic")
        with contextlib.suppress(ValueError):
            memory_type = MemoryType(raw_type)

        # Clamp importance to valid range regardless of what LLM returned.
        importance = max(0.0, min(1.0, float(entry.get("importance", 0.8))))

        return Fact(
            content=str(entry.get("content", "")),
            type=memory_type,
            importance=importance,
        )
