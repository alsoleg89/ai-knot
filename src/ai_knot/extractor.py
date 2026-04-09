"""LLM-based fact extraction from conversations.

Supports any LLM provider via the :class:`LLMProvider` protocol.
The LLM is instructed to return structured JSON with extracted facts.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from typing import Any

from ai_knot._resolve import (
    deduplicate_facts,
    resolve_against_existing,
    resolve_by_slot,
    resolve_structured,
)
from ai_knot.providers import LLMProvider, call_with_retry, create_provider
from ai_knot.tokenizer import tokenize
from ai_knot.types import ConversationTurn, Fact, MemoryOp, MemoryType

logger = logging.getLogger(__name__)

_EXTRACTION_SYSTEM_PROMPT = """You are a knowledge extraction engine.
Given a conversation, extract ONLY meaningful facts worth remembering.

Rules:
- Skip greetings, thanks, filler ("ok", "got it", "thanks").
- Each fact must be a single, self-contained statement.
- Classify each fact as: semantic (about user/world), procedural (preferences/how-to),
  episodic (specific events).
- Rate importance from 0.0 to 1.0.
- Include 1-3 short domain tags per fact (lowercase, single words).
- Add "canonical": a short, normalised paraphrase for deduplication and search
  (remove names/pronouns where possible; keep the core proposition).
- Add "witness": copy the KEY PHRASE or rule EXACTLY as it appears in the source
  text. This is the verbatim evidence span used for grounding and audit.
  Omit only when there is no distinct quotable phrase.
- For facts about a person, organization, or object: add "entity" (the subject,
  e.g. "Alex Chen") and "attribute" (the property, e.g. "salary", "job_title",
  "employer"), and "value" (the specific value, e.g. "95000").
  Add "qualifiers" as a JSON object for temporal or conditional context,
  e.g. {"since": "2024-01", "currency": "USD"}.
  Omit entity/attribute/value/qualifiers for general statements where they don't apply.
- When messages have timestamps (shown as [DATE] prefixes), resolve relative
  temporal references to absolute dates in the extracted fact content.
  E.g. if the timestamp is [8 May, 2023] and the speaker says "yesterday",
  write "7 May 2023" in the fact content.  Keep the original phrasing in
  the "witness" field (verbatim evidence).
- Add "op" to signal extraction intent (string, one of: "add", "update", "delete", "noop"):
  "add"    — new fact not previously known (default when omitted).
  "update" — conversation explicitly corrects an existing value
             (e.g. "actually my salary is now $120k", "I switched to TypeScript").
  "delete" — conversation explicitly removes knowledge
             (e.g. "forget that I work at Acme", "I no longer use Redis").
  "noop"   — conversation merely confirms already-known information; nothing new.

Return a JSON array. Example:
[
  {"content": "Alex Chen works at FinServe Capital as Senior PM", "type": "semantic",
   "importance": 0.9, "tags": ["employer", "role"],
   "canonical": "person works at company as role",
   "witness": "Alex Chen, Senior Product Manager at FinServe Capital",
   "entity": "Alex Chen", "attribute": "job_title", "value": "Senior PM",
   "qualifiers": {}, "op": "add"},
  {"content": "User prefers Python over Java",
   "type": "procedural", "importance": 0.85,
   "tags": ["python", "preferences"],
   "canonical": "prefers Python over Java",
   "witness": "I prefer Python over Java", "op": "add"}
]

If no meaningful facts exist, return an empty array: []
Return ONLY the JSON array, no other text."""


def _format_turn(turn: ConversationTurn) -> str:
    """Format a conversation turn for the LLM prompt.

    Prepends a ``[date]`` prefix when the turn carries a timestamp so the
    LLM can resolve relative temporal references ("yesterday", "last week")
    to absolute dates.
    """
    if turn.timestamp is not None:
        date_str = turn.timestamp.strftime("%-d %B, %Y")
        return f"[{date_str}] {turn.role}: {turn.content}"
    return f"{turn.role}: {turn.content}"


def _atc_score(snippet: str, source: str) -> float:
    """Asymmetric Token Containment: fraction of snippet tokens found in source.

    ATC = |tokens(snippet) ∩ tokens(source)| / |tokens(snippet)|

    Returns 1.0 if snippet is empty (vacuously supported).
    Inspired by Broder (1997) similarity estimation.
    """
    s_tokens = set(tokenize(snippet))
    if not s_tokens:
        return 1.0
    src_tokens = set(tokenize(source))
    return len(s_tokens & src_tokens) / len(s_tokens)


def _verify_facts_atc(
    facts: list[Fact],
    source_text: str,
    *,
    threshold: float = 0.6,
) -> list[Fact]:
    """Verify facts against source text using ATC and annotate each fact.

    For each fact, computes the ATC score of ``fact.content`` against
    ``source_text``, then sets:
    - ``fact.supported``: ``True`` if score >= threshold.
    - ``fact.support_confidence``: the ATC score.
    - ``fact.verification_source``: ``"atc"``.

    Args:
        facts: Facts to verify (modified in place).
        source_text: The original text that the facts were extracted from.
        threshold: Minimum ATC score to consider a fact supported.

    Returns:
        The same list, modified in place.
    """
    for fact in facts:
        score = _atc_score(fact.content, source_text)
        if fact.witness_surface:
            score = max(score, _atc_score(fact.witness_surface, source_text))
        fact.supported = score >= threshold
        fact.support_confidence = score
        fact.verification_source = "atc"
    return facts


def _populate_source_snippets(
    facts: list[Fact],
    turn_texts: list[str],
    *,
    max_snippets: int = 3,
    min_atc: float = 0.4,
) -> None:
    """Populate fact.source_snippets from original conversation turns.

    For each fact, scores every turn by ATC(fact.content, turn) and by
    ATC(fact.witness_surface, turn) (if non-empty), then stores the top
    ``max_snippets`` turns with score >= ``min_atc``.  Modifies in place.
    """
    for fact in facts:
        scored: dict[str, float] = {}
        for turn in turn_texts:
            score = _atc_score(fact.content, turn)
            if fact.witness_surface:
                score = max(score, _atc_score(fact.witness_surface, turn))
            if score >= min_atc and turn not in scored:
                scored[turn] = score
        fact.source_snippets = [
            t for t, _ in sorted(scored.items(), key=lambda x: x[1], reverse=True)
        ][:max_snippets]


# Re-export resolver functions so existing imports from ai_knot.extractor still work.
__all__ = [
    "Extractor",
    "deduplicate_facts",
    "resolve_against_existing",
    "resolve_by_slot",
    "resolve_structured",
]


class Extractor:
    """Extract structured facts from conversations using an LLM.

    Args:
        provider: An LLM provider instance, or a provider name string.
            If a string, ``api_key`` is required.
        api_key: API key (used only when ``provider`` is a string).
        model: Model name (defaults to provider's default model).
        timeout: Per-request timeout in seconds. ``None`` uses the provider default.
        batch_size: Maximum number of conversation turns per LLM call. Long
            conversations are automatically split into chunks of this size and
            results are merged. Prevents JSON truncation on large inputs.
    """

    def __init__(
        self,
        provider: LLMProvider | str = "openai",
        *,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        batch_size: int = 20,
        **provider_kwargs: str,
    ) -> None:
        if isinstance(provider, str):
            self._provider = create_provider(provider, api_key, **provider_kwargs)
        else:
            self._provider = provider
        self._model = model or self._provider.default_model
        self._timeout = timeout
        self._batch_size = batch_size

    def extract(self, turns: list[ConversationTurn]) -> list[Fact]:
        """Extract facts from a conversation.

        Long conversations are automatically split into chunks of ``batch_size``
        turns so that the LLM never receives more than that at once, preventing
        silent fact loss due to JSON truncation.

        Args:
            turns: List of conversation messages.

        Returns:
            List of extracted Facts. Returns [] on any LLM error.
        """
        if not turns:
            return []

        # ATC source includes timestamp prefixes so resolved dates
        # ("7 May 2023" from "yesterday" + [8 May, 2023]) pass verification.
        source_text = "\n".join(_format_turn(t) for t in turns)

        all_raw: list[dict[str, Any]] = []
        for i in range(0, len(turns), self._batch_size):
            chunk = turns[i : i + self._batch_size]
            all_raw.extend(self._call_llm(chunk))

        facts = [self._parse_fact(entry) for entry in all_raw if isinstance(entry, dict)]
        facts = deduplicate_facts(facts)
        _verify_facts_atc(facts, source_text)
        # Source snippets use timestamped turn text so evidence contains the date
        # context and BM25 can match temporal query tokens.
        turn_texts = [_format_turn(t) for t in turns]
        _populate_source_snippets(facts, turn_texts)
        return facts

    def _call_llm(self, turns: list[ConversationTurn]) -> list[dict[str, Any]]:
        """Call the LLM to extract facts. Returns parsed JSON array."""
        conversation_text = "\n".join(_format_turn(t) for t in turns)
        content = call_with_retry(
            self._provider,
            _EXTRACTION_SYSTEM_PROMPT,
            conversation_text,
            self._model,
            timeout=self._timeout,
        )
        if not content:
            return []
        return self._parse_json_response(content)

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
            if isinstance(parsed, dict) and isinstance(parsed.get("facts"), list):
                inner: list[dict[str, Any]] = parsed["facts"]
                return inner
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
        try:
            importance = max(0.0, min(1.0, float(entry.get("importance", 0.8))))
        except (ValueError, TypeError):
            importance = 0.8

        # Parse tags from LLM response (graceful degradation if missing).
        raw_tags = entry.get("tags", [])
        tags: list[str] = []
        if isinstance(raw_tags, list):
            tags = [str(t).lower().strip() for t in raw_tags if isinstance(t, str)][:5]

        # Parse qualifiers dict (graceful degradation if malformed).
        raw_qualifiers = entry.get("qualifiers", {})
        qualifiers: dict[str, str] = {}
        if isinstance(raw_qualifiers, dict):
            qualifiers = {str(k): str(v) for k, v in raw_qualifiers.items()}

        entity = str(entry.get("entity", "")).strip().lower()
        attribute = str(entry.get("attribute", "")).strip().lower()
        # Derive deterministic slot key when entity+attribute are both present.
        slot_key = f"{entity}::{attribute}" if entity and attribute else ""

        witness = str(entry.get("witness", "") or entry.get("verbatim", ""))
        canonical = str(entry.get("canonical", ""))

        # Parse LLM-signalled extraction intent (defaults to ADD when absent/invalid).
        op = MemoryOp.ADD
        raw_op = entry.get("op", "add")
        with contextlib.suppress(ValueError):
            op = MemoryOp(str(raw_op).lower())

        return Fact(
            content=str(entry.get("content", "")),
            type=memory_type,
            importance=importance,
            tags=tags,
            source_verbatim=witness,
            entity=entity,
            attribute=attribute,
            canonical_surface=canonical,
            witness_surface=witness,
            value_text=str(entry.get("value", "")),
            qualifiers=qualifiers,
            slot_key=slot_key,
            op=op,
        )
