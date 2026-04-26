"""LLM-based fact extraction from conversations.

Supports any LLM provider via the :class:`LLMProvider` protocol.
The LLM is instructed to return structured JSON with extracted facts.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import re
from typing import Any
from uuid import uuid4

from ai_knot._date_enrichment import enrich_date_tags
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
Given a conversation, extract ALL meaningful facts worth remembering.

Rules:
- Extract every factual statement: names, places, dates, events,
  preferences, plans, relationships, opinions.
- Skip ONLY greetings, thanks, and pure filler ("ok", "got it", "thanks").
- Each fact must be a single, self-contained statement.
- When messages have timestamps (shown as [DATE] prefixes), resolve relative
  temporal references to absolute dates in the extracted fact content.
  E.g. if the timestamp is [8 May, 2023] and the speaker says "yesterday",
  write "7 May 2023" in the fact content.
- For facts about a person, organization, or object: add "entity" (the subject,
  e.g. "Alex Chen") and "attribute" (the property, e.g. "job_title",
  "employer"), and "value" (the specific value, e.g. "Senior PM").
  Omit entity/attribute/value for general statements where they don't apply.

Return a JSON array. Each element needs only these fields:
  "content" (string, required) — the fact
  "type" (string) — "semantic", "procedural", or "episodic"
  "importance" (number) — 0.0 to 1.0
  "tags" (array of strings) — 1-3 domain labels
  "entity", "attribute", "value" (strings, optional) — for structured facts

Example:
[
  {"content": "Alex Chen works at FinServe Capital as Senior PM", "type": "semantic",
   "importance": 0.9, "tags": ["employer", "role"],
   "entity": "Alex Chen", "attribute": "job_title", "value": "Senior PM"},
  {"content": "User prefers Python over Java",
   "type": "procedural", "importance": 0.85, "tags": ["python", "preferences"]}
]

If no meaningful facts exist, return an empty array: []
Return ONLY the JSON array, no other text."""

# ---------------------------------------------------------------------------
# C6b — Enumeration split helpers
# ---------------------------------------------------------------------------

# Comma-separated: "A, B, C" or "A, B, and C" — ≥ 3 items.
# First captured group is a SINGLE word so the regex anchors at the first list
# item rather than absorbing the preceding verb into the match.
_ENUM_PATTERN_COMMA = re.compile(
    r"\b(\w+)(?:,\s*\w+(?:\s+\w+)?){2,}"
    r"(?:,?\s+(?:and|or)\s+\w+(?:\s+\w+)?)?",
    re.I,
)
# Semicolon-separated: "A; B; C" or "A; B; and C" — ≥ 3 items.
_ENUM_PATTERN_SEMI = re.compile(
    r"\b(\w+)(?:;\s*\w+(?:\s+\w+)?){2,}"
    r"(?:;?\s+(?:and|or)\s+\w+(?:\s+\w+)?)?",
    re.I,
)
_ITEM_SPLIT_COMMA = re.compile(r",\s*", re.I)
_ITEM_SPLIT_SEMI = re.compile(r";\s*", re.I)
_LEADING_CONJ = re.compile(r"^(?:and|or)\s+", re.I)
_MAX_ITEM_LEN = 20  # chars per item; guards against splitting long phrases

# Matches a leading "[date]" bracket prefix (e.g. "[27 June, 2023] ").
_DATE_PREFIX_RE = re.compile(r"^\s*\[[^\]]+\]\s*")


def _extract_enum_items(match_text: str, splitter: re.Pattern[str]) -> list[str]:
    """Split an enumeration match into individual items, filtered by length.

    Handles the common pattern "X, Y, and Z" where the separator before "and"
    may leave "and Z" as the last raw chunk — we strip the leading conjunction.
    """
    raw = splitter.split(match_text.strip())
    items = [_LEADING_CONJ.sub("", t).strip(" .!?-") for t in raw if t.strip()]
    return [it for it in items if 1 < len(it) <= _MAX_ITEM_LEN]


def _build_verb_prefix(content: str, match_start: int) -> str:
    """Extract a clean verb prefix from *content* up to *match_start*.

    For plain facts the prefix is simply everything before the enumeration
    match.  For dated-window content that contains ``[date]`` prefix and
    ``/``-separated turns, we:

    1. Strip and re-attach the leading ``[date]`` bracket so children still
       carry the date (required for ``enrich_date_tags`` to work on them).
    2. Trim the body prefix to the nearest turn separator (``/``) or sentence
       boundary (``". "``) so we don't drag previous turns into child content.

    Examples::

        "[2023-06-27] Alice: I love pottery, camping, swimming"
        → "[2023-06-27] Alice: I love"   (clean; no prior-turn noise)

        "[2023-06-27] Bob: hi / Alice: I love pottery, camping, swimming"
        → "[2023-06-27] Alice: I love"   (trimmed at "/" — prior turn dropped)

        "Melanie enjoys pottery, camping, swimming"
        → "Melanie enjoys"               (unchanged — no date prefix or "/" sep)
    """
    date_m = _DATE_PREFIX_RE.match(content)
    date_prefix = date_m.group(0) if date_m else ""
    body = content[len(date_prefix) :]
    body_before = body[: match_start - len(date_prefix)] if date_m else content[:match_start]

    # Trim to the latest turn-separator or sentence boundary.
    cut = max(body_before.rfind("/"), body_before.rfind(". "))
    verb_body = body_before[cut + 1 :] if cut >= 0 else body_before

    # Strip trailing whitespace and list-intro punctuation (; — -) but NOT
    # colons, so that "Hobbies:" is preserved as a valid verb prefix.
    return (date_prefix + verb_body).rstrip(" ;—-")


def split_enumerations(facts: list[Fact]) -> list[Fact]:
    """Emit one child Fact per enumerated item alongside the original list-fact.

    Detects comma-separated **or** semicolon-separated lists with ≥ 3 short
    items (≤ 20 chars each) and emits additional atomic facts for each item
    while keeping the original list-fact intact for deduplication downstream.

    Works on **all** ingest modes:
    - ``learn`` / ``dated-learn``: called inside ``Extractor.extract()`` before
      ``deduplicate_facts``.
    - ``raw`` / ``dated``: called inside ``KnowledgeBase.add()`` after
      ``enrich_date_tags``.

    For dated-window content the leading ``[date]`` bracket is preserved on
    child facts so ``enrich_date_tags`` can annotate them with temporal tags.
    Turn-separator (``/``) noise from prior windows is trimmed from the prefix.

    Structured fields (``entity``, ``attribute``, ``tags``, ``type``,
    ``slot_key``) are copied to derived facts; ``importance`` is reduced by
    0.05 to rank derived facts slightly below the source.
    ``source_snippets`` are reset to ``[]`` so ``_populate_source_snippets``
    can re-populate them based on the narrower item content.
    """
    extras: list[Fact] = []
    for f in facts:
        if not f.content:
            continue

        # Try comma first, then semicolon.
        m = _ENUM_PATTERN_COMMA.search(f.content)
        splitter = _ITEM_SPLIT_COMMA
        if m is None:
            m = _ENUM_PATTERN_SEMI.search(f.content)
            splitter = _ITEM_SPLIT_SEMI
        if m is None:
            continue

        items = _extract_enum_items(m.group(0), splitter)
        if len(items) < 3:
            continue

        verb_prefix = _build_verb_prefix(f.content, m.start())
        for item in items:
            new_content = f"{verb_prefix} {item}".strip() if verb_prefix else item
            extras.append(
                dataclasses.replace(
                    f,
                    id=uuid4().hex[:8],
                    content=new_content,
                    importance=max(0.0, f.importance - 0.05),
                    tags=list(f.tags),  # copy — don't share the mutable list
                    source_snippets=[],  # reset; ATC re-populates for this content
                    access_intervals=[],
                )
            )
    return facts + extras


# Keep old private name as alias so any internal callers still work.
_split_enumerations = split_enumerations


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
    "split_enumerations",
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
        # C6b: expand list-facts into per-item atomic facts before dedup.
        facts = _split_enumerations(facts)
        facts = deduplicate_facts(facts)
        _verify_facts_atc(facts, source_text)
        # Source snippets use timestamped turn text so evidence contains the date
        # context and BM25 can match temporal query tokens.
        turn_texts = [_format_turn(t) for t in turns]
        _populate_source_snippets(facts, turn_texts)
        # C6c: inject canonical date tags so temporal queries match via BM25F.
        for f in facts:
            enrich_date_tags(f)
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

        # Accept witness/canonical/op from LLM if provided, otherwise derive.
        witness = str(entry.get("witness", "") or entry.get("verbatim", ""))
        content = str(entry.get("content", ""))
        canonical = str(entry.get("canonical", ""))
        if not canonical and content:
            # Auto-generate canonical: lowercase, strip leading articles.
            canonical = re.sub(r"^(the|a|an)\s+", "", content.lower()).strip()

        # Parse LLM-signalled extraction intent (defaults to ADD when absent/invalid).
        op = MemoryOp.ADD
        raw_op = entry.get("op", "add")
        with contextlib.suppress(ValueError):
            op = MemoryOp(str(raw_op).lower())

        return Fact(
            content=content,
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
