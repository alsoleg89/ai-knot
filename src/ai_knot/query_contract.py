"""Query contract derivation — geometry-based routing without surface-keyword policy.

Design rules:
  * No routing based on surface keywords like "would", "likely", "might", "when".
  * Signals (lexical hints) are weak features only — they never override evidence geometry.
  * BOOL answer_space defaults to DIRECT truth_mode, not HYPOTHESIS.
    Only choose_strategy() in query_runtime may switch to HYPOTHESIS after seeing
    EvidenceProfile (i.e., when direct evidence is absent).

Enforced by: scripts/check_query_runtime_isolation.py (no storage access)
             tests/test_query_contract.py (product queries, no LoCoMo branching)
"""

from __future__ import annotations

import re

from ai_knot.query_types import (
    AnswerContract,
    AnswerSpace,
    EvidenceRegime,
    QueryFrame,
    TimeAxis,
    TruthMode,
)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def analyze_query(question: str) -> QueryFrame:
    """Derive a QueryFrame from a natural-language question.

    Geometry is derived from structural features of the question, not from
    benchmark-category labels or surface-keyword policy rules.
    """
    tokens = _tokenize_lower(question)
    entities = _extract_focus_entities(question)
    answer_space = _detect_geometry(question, tokens)
    temporal_scope = _detect_temporal_scope(question, tokens)
    locality = _detect_locality(question, entities)
    epistemic_mode = _detect_epistemic_mode(question, tokens, answer_space)
    evidence_regime = _derive_evidence_regime(answer_space)
    focus_relation = _extract_focus_relation(question, entities)
    target_kind = _derive_target_kind(answer_space, tokens, temporal_scope)

    return QueryFrame(
        focus_entities=tuple(entities),
        target_kind=target_kind,
        answer_space=answer_space,
        temporal_scope=temporal_scope,
        epistemic_mode=epistemic_mode,
        locality=locality,
        evidence_regime=evidence_regime,
        focus_relation=focus_relation,
    )


def derive_answer_contract(frame: QueryFrame) -> AnswerContract:
    """Map a QueryFrame to an AnswerContract.

    Note: truth_mode here is a preliminary suggestion.  The actual operator
    selection happens in choose_strategy() after EvidenceProfile is computed.
    BOOL does NOT automatically map to HYPOTHESIS — that only happens when
    choose_strategy() finds no direct evidence.
    """
    if frame.answer_space is AnswerSpace.SET or frame.temporal_scope in ("interval",):
        truth_mode = TruthMode.RECONSTRUCT
    else:
        truth_mode = TruthMode.DIRECT  # default for all spaces, including BOOL

    time_axis = _temporal_scope_to_axis(frame.temporal_scope)

    return AnswerContract(
        answer_space=frame.answer_space,
        truth_mode=truth_mode,
        time_axis=time_axis,
        locality=frame.locality,
        evidence_regime=frame.evidence_regime,
    )


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

# Tokens that signal a set/aggregation question.
_SET_SIGNALS: frozenset[str] = frozenset(
    {
        "what",
        "which",
        "list",
        "all",
        "every",
        "each",
        "name",
        "mention",
        "types",
        "kinds",
        "examples",
        "activities",
        "hobbies",
        "books",
        "movies",
        "places",
        "people",
        "friends",
        "members",
        "items",
    }
)

# Tokens that signal a boolean question.
_BOOL_SIGNALS: frozenset[str] = frozenset(
    {
        "is",
        "are",
        "was",
        "were",
        "did",
        "does",
        "do",
        "has",
        "have",
        "had",
        "can",
        "could",
        "should",
        "would",
        "will",
    }
)

# Tokens that signal temporal scope.
_CURRENT_SIGNALS: frozenset[str] = frozenset(
    {"now", "current", "currently", "today", "present", "still", "latest"}
)
_HISTORICAL_SIGNALS: frozenset[str] = frozenset(
    {
        "before",
        "ago",
        "previously",
        "used to",
        "once",
        "past",
        "earlier",
        "back then",
        "at the time",
        "last year",
        "then",
        "formerly",
    }
)
_INTERVAL_SIGNALS: frozenset[str] = frozenset(
    {"between", "from", "during", "throughout", "period", "span", "range", "over", "across"}
)
_EVENT_TIME_SIGNALS: frozenset[str] = frozenset(
    {
        "when",
        "date",
        "time",
        "moment",
        "occasion",
        "which day",
        "what day",
        "what time",
        "what year",
        "what month",
    }
)

# Named entity detection — one or more capitalized consecutive words.
_NAMED_ENTITY_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b")

# Question-opener words that are capitalized but are not entity names.
_STOP_CAPS: frozenset[str] = frozenset(
    {
        "What",
        "When",
        "Who",
        "Where",
        "How",
        "Why",
        "Which",
        "Is",
        "Are",
        "Was",
        "Were",
        "Did",
        "Do",
        "Does",
        "Has",
        "Have",
        "Had",
        "Will",
        "Should",
        "Could",
        "Would",
        "Can",
        "Tell",
        "List",
        "Name",
        "Describe",
    }
)

# Verb-like focus relation signals.
_RELATION_VERBS: frozenset[str] = frozenset(
    {
        "work",
        "live",
        "study",
        "know",
        "meet",
        "love",
        "marry",
        "play",
        "attend",
        "visit",
        "read",
        "watch",
        "drink",
        "eat",
        "drive",
        "use",
    }
)


def _tokenize_lower(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _extract_focus_entities(question: str) -> list[str]:
    """Extract named entities from the question (proper-name heuristic).

    Handles single-token names (Alice, Bob) and strips possessive suffixes
    (Alice's → Alice).  Question-opener words (What, When, …) are excluded.
    """
    seen: set[str] = set()
    entities: list[str] = []
    for m in _NAMED_ENTITY_RE.finditer(question):
        raw = m.group(1)
        # Strip possessive suffix.
        if raw.endswith("'s"):
            e = raw[:-2]
        elif raw.endswith("'"):
            e = raw[:-1]
        else:
            e = raw
        if not e or e in _STOP_CAPS:
            continue
        if e not in seen:
            seen.add(e)
            entities.append(e)
    return entities


def _detect_geometry(question: str, tokens: list[str]) -> AnswerSpace:
    """Classify the expected answer shape from question structure."""
    # Starts with "how many" / "how much" → scalar
    q_lower = question.lower().strip()
    if q_lower.startswith(("how many", "how much", "how often", "how long")):
        return AnswerSpace.SCALAR

    # Question begins with "what" + noun (likely set or description).
    token_set = set(tokens[:8])  # look at first 8 tokens

    # Structural SET signals — avoids hardcoded noun lists.
    if tokens and tokens[0] == "what":
        tail = tokens[1:4]
        # Plural-noun heuristic: ends in 's' (not 'ss'), length > 3, no apostrophe
        # (apostrophe indicates possessive, not plural).
        if any(
            t.endswith("s") and not t.endswith("ss") and len(t) > 3 and "'" not in t for t in tail
        ):
            return AnswerSpace.SET
        if "all" in tail or "list" in tail:
            return AnswerSpace.SET

    # "list / enumerate / name all / what are all" → SET
    if token_set & {"list", "enumerate"}:
        return AnswerSpace.SET
    if "all" in token_set and "what" in token_set:
        return AnswerSpace.SET

    # "who" → ENTITY
    if tokens and tokens[0] == "who":
        return AnswerSpace.ENTITY

    # Yes/no question patterns: starts with auxiliary or "is/are/was/were/did/does".
    first = tokens[0] if tokens else ""
    if first in _BOOL_SIGNALS:
        return AnswerSpace.BOOL

    # Inversion: "Did X...?", "Has X...?", "Are X...?"
    if q_lower.endswith("?") and first in _BOOL_SIGNALS:
        return AnswerSpace.BOOL

    # "when" → scalar (a date) — but mark as SCALAR not SET
    if tokens and tokens[0] == "when":
        return AnswerSpace.SCALAR

    # "where" → ENTITY (a place)
    if tokens and tokens[0] == "where":
        return AnswerSpace.ENTITY

    # Default: description
    return AnswerSpace.DESCRIPTION


def _detect_temporal_scope(question: str, tokens: list[str]) -> str:
    """Classify temporal scope: current / historical / interval / none."""
    token_set = set(tokens)
    if token_set & _CURRENT_SIGNALS:
        return "current"
    if token_set & _INTERVAL_SIGNALS:
        return "interval"
    if token_set & _EVENT_TIME_SIGNALS:
        return "historical"  # asking for a specific past event time
    if token_set & _HISTORICAL_SIGNALS:
        return "historical"
    # "what is X's Y" (present tense, no time signal) → current
    if tokens and tokens[0] in ("what", "who", "where") and "is" in token_set:
        return "current"
    return "none"


def _detect_locality(question: str, entities: list[str]) -> str:
    """Estimate the locality dimension of the query."""
    if len(entities) >= 2:
        return "cross_entity"
    if len(entities) == 1:
        q_lower = question.lower()
        if any(w in q_lower for w in ("everything", "all about", "tell me about")):
            return "entity_scope"
        return "point"
    return "event_neighborhood"


def _detect_epistemic_mode(
    question: str, tokens: list[str], answer_space: AnswerSpace
) -> TruthMode:
    """Return a weak epistemic hint — NOT a hard routing decision.

    DIRECT is the default.  Only choose_strategy() in query_runtime may
    switch to HYPOTHESIS or RANKED after examining EvidenceProfile.
    """
    # For SET/SCALAR/DESCRIPTION, RECONSTRUCT or DIRECT are both fine here.
    # The contract's truth_mode overrides this anyway.
    return TruthMode.DIRECT


def _derive_evidence_regime(answer_space: AnswerSpace) -> EvidenceRegime:
    if answer_space is AnswerSpace.BOOL:
        return EvidenceRegime.SUPPORT_VS_CONTRA
    if answer_space is AnswerSpace.SET:
        return EvidenceRegime.AGGREGATE
    return EvidenceRegime.SINGLE


def _extract_focus_relation(question: str, entities: list[str]) -> str | None:
    """Try to extract a verb-like focus relation from the question."""
    tokens = _tokenize_lower(question)
    for t in tokens:
        if t in _RELATION_VERBS:
            return t
    return None


def _derive_target_kind(answer_space: AnswerSpace, tokens: list[str], temporal_scope: str) -> str:
    token_set = set(tokens[:8])
    if temporal_scope in ("historical", "interval"):
        return "event"
    if answer_space is AnswerSpace.ENTITY:
        if "where" in token_set:
            return "location"
        if "who" in token_set:
            return "identity"
        return "entity"
    if answer_space is AnswerSpace.SET:
        return "set"
    if answer_space is AnswerSpace.SCALAR:
        return "scalar"
    if answer_space is AnswerSpace.BOOL:
        return "state"
    return "description"


def _temporal_scope_to_axis(temporal_scope: str) -> TimeAxis:
    return {
        "current": TimeAxis.CURRENT,
        "historical": TimeAxis.EVENT,
        "interval": TimeAxis.INTERVAL,
        "none": TimeAxis.NONE,
    }.get(temporal_scope, TimeAxis.NONE)
