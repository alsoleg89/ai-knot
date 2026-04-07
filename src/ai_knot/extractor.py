"""LLM-based fact extraction from conversations.

Supports any LLM provider via the :class:`LLMProvider` protocol.
The LLM is instructed to return structured JSON with extracted facts.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from datetime import UTC, datetime
from typing import Any

from ai_knot.providers import LLMProvider, call_with_retry, create_provider
from ai_knot.tokenizer import tokenize
from ai_knot.types import ConversationTurn, Fact, MemoryType

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

Return a JSON array. Example:
[
  {"content": "Alex Chen works at FinServe Capital as Senior PM", "type": "semantic",
   "importance": 0.9, "tags": ["employer", "role"],
   "canonical": "person works at company as role",
   "witness": "Alex Chen, Senior Product Manager at FinServe Capital",
   "entity": "Alex Chen", "attribute": "job_title", "value": "Senior PM",
   "qualifiers": {}},
  {"content": "User prefers Python over Java",
   "type": "procedural", "importance": 0.85,
   "tags": ["python", "preferences"],
   "canonical": "prefers Python over Java",
   "witness": "I prefer Python over Java"}
]

If no meaningful facts exist, return an empty array: []
Return ONLY the JSON array, no other text."""


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
        fact.supported = score >= threshold
        fact.support_confidence = score
        fact.verification_source = "atc"
    return facts


def _jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard similarity between two strings using stemmed tokens.

    Uses the shared stemmed tokenizer (Broder 1997) so that morphological
    variants like "deployed"/"deploying" are recognized as identical.
    This keeps deduplication consistent with the retriever's BM25F scoring.
    """
    words_a = set(tokenize(a))
    words_b = set(tokenize(b))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _containment_similarity(a: str, b: str) -> float:
    """Max asymmetric token containment between two strings.

    containment(A, B) = |A ∩ B| / min(|A|, |B|)

    Catches cases where a short fact is a subset of a longer one, or
    two facts express the same idea with different amounts of detail.
    Jaccard penalises length asymmetry; containment does not.
    """
    words_a = set(tokenize(a))
    words_b = set(tokenize(b))
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    return intersection / min(len(words_a), len(words_b))


def _dedup_similarity(a: str, b: str) -> float:
    """Combined dedup similarity: max(jaccard, containment).

    Using the max of both metrics catches both symmetric overlap
    (Jaccard) and asymmetric subset relationships (containment).
    """
    return max(_jaccard_similarity(a, b), _containment_similarity(a, b))


# Unified dedup threshold for both within-batch and cross-store deduplication.
# 0.85 avoids false-positive merges of short but semantically distinct rules
# (e.g. "keep posts under 300 words" vs "use fenced code blocks"), which occur
# at the original 0.7 threshold when containment similarity exceeds min-token ratio.
_DEDUP_THRESHOLD: float = 0.85


def deduplicate_facts(facts: list[Fact], *, threshold: float = _DEDUP_THRESHOLD) -> list[Fact]:
    """Remove near-duplicate facts by combined similarity (Jaccard + containment).

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
            if _dedup_similarity(fact.content, existing.content) >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(fact)
    return unique


def resolve_against_existing(
    new_facts: list[Fact],
    existing: list[Fact],
    *,
    threshold: float = _DEDUP_THRESHOLD,
) -> tuple[list[Fact], list[Fact]]:
    """Separate new facts into inserts and temporal closes relative to existing facts.

    For each new fact, if a sufficiently similar existing fact is found
    (combined similarity >= threshold), the old fact is **temporally closed**
    (``valid_until`` set to now) and the new fact is queued for insertion with
    bumped importance. This preserves the full fact history instead of mutating
    it in place.

    If no match is found, the new fact is inserted unchanged.

    Args:
        new_facts: Facts extracted from the latest conversation.
        existing: Active facts already stored for this agent.
        threshold: Combined similarity threshold to consider two facts duplicates.

    Returns:
        A 2-tuple ``(to_insert, closed_existing)`` where:
        - ``to_insert``: facts to insert (both genuinely new and new-version replacements).
        - ``closed_existing``: old facts that were temporally closed (``valid_until`` set).
    """
    to_insert: list[Fact] = []
    closed: list[Fact] = []
    now = datetime.now(UTC)

    for new in new_facts:
        matched: Fact | None = None
        best_sim = 0.0
        for old in existing:
            sim = _dedup_similarity(new.content, old.content)
            if sim >= threshold and sim > best_sim:
                best_sim = sim
                matched = old
        if matched is not None:
            # Temporal close: seal the old version without mutating its content.
            matched.valid_until = now
            closed.append(matched)
            # Carry forward importance and version from the old fact.
            new.importance = min(1.0, matched.importance + 0.05)
            new.version = matched.version + 1
            to_insert.append(new)
        else:
            to_insert.append(new)

    return to_insert, closed


_PRONOUNS = frozenset({"he", "she", "it", "they", "i", "we", "you", "him", "her", "them"})


def entity_match(a: str, b: str) -> bool:
    """Return True if two entity strings refer to the same real-world entity.

    Uses containment (substring in both directions) and Jaccard similarity.
    Guards against pronoun-based false matches (e.g. "he" != "Alex Chen").
    """
    a, b = a.lower().strip(), b.lower().strip()
    if not a or not b:
        return False
    if a in _PRONOUNS or b in _PRONOUNS:
        return False
    if a in b or b in a:
        return True
    return _jaccard_similarity(a, b) > 0.5


def resolve_structured(
    new_fact: Fact,
    existing: list[Fact],
) -> Fact | None:
    """Entity-addressed dedup: find existing fact with same entity+attribute.

    Returns the existing Fact if a match is found, or None if no match
    (meaning this fact should be treated as a new insert).

    Only fires when both new_fact.entity and new_fact.attribute are non-empty.
    Falls back to cosine-based dedup otherwise.
    """
    if not new_fact.entity or not new_fact.attribute:
        return None
    for existing_fact in existing:
        if not existing_fact.entity or not existing_fact.attribute:
            continue
        if (
            entity_match(new_fact.entity, existing_fact.entity)
            and new_fact.attribute.lower().strip() == existing_fact.attribute.lower().strip()
            and existing_fact.is_active()
        ):
            return existing_fact
    return None


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

        source_text = "\n".join(f"{t.role}: {t.content}" for t in turns)

        all_raw: list[dict[str, Any]] = []
        for i in range(0, len(turns), self._batch_size):
            chunk = turns[i : i + self._batch_size]
            all_raw.extend(self._call_llm(chunk))

        facts = [self._parse_fact(entry) for entry in all_raw if isinstance(entry, dict)]
        facts = deduplicate_facts(facts)
        _verify_facts_atc(facts, source_text)
        return facts

    def _call_llm(self, turns: list[ConversationTurn]) -> list[dict[str, Any]]:
        """Call the LLM to extract facts. Returns parsed JSON array."""
        conversation_text = "\n".join(f"{t.role}: {t.content}" for t in turns)
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

        entity = str(entry.get("entity", ""))
        attribute = str(entry.get("attribute", ""))
        # Derive deterministic slot key when entity+attribute are both present.
        slot_key = f"{entity}::{attribute}" if entity and attribute else ""

        witness = str(entry.get("witness", "") or entry.get("verbatim", ""))
        canonical = str(entry.get("canonical", ""))

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
        )
