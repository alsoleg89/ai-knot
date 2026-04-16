"""Query contract derivation — geometry-based routing without surface-keyword policy.

Design rules:
  * No routing based on surface keywords like "would", "likely", "might", "when".
  * Signals (lexical hints) are weak features only — they never override evidence geometry.
  * BOOL answer_space defaults to DIRECT truth_mode, not HYPOTHESIS.
    Only choose_strategy() in query_runtime may switch to HYPOTHESIS after seeing
    EvidenceProfile (i.e., when direct evidence is absent).

Enforced by: scripts/check_query_runtime_isolation.py (no storage access)
             tests/test_query_contract.py (product queries, no benchmark-label branching)
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

# Aggregation noun heads that make "what/which + has/have/does + head" a SET question.
_SET_NOUN_HEADS: frozenset[str] = frozenset(
    {
        "books",
        "movies",
        "films",
        "shows",
        "places",
        "countries",
        "cities",
        "languages",
        "hobbies",
        "activities",
        "sports",
        "instruments",
        "skills",
        "jobs",
        "courses",
        "classes",
        "events",
        "awards",
        "friends",
        "pets",
        "children",
        "kids",
    }
)

# Verb-like focus relation signals (canonical lemma forms).
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
        "research",
        "restore",
        "pursue",
        "find",
        "pass",
        "satisfy",
        "grow",
        "build",
        "buy",
        "sell",
        "like",
        "enjoy",
        "hate",
        "prefer",
        "move",
        "join",
        "leave",
        "start",
        "stop",
        "retire",
        "die",
    }
)

# Inflected verb → canonical lemma.  Checked before _RELATION_VERBS lookup so
# that question tokens like "drives", "restoring" can be normalized to their
# infinitive form before becoming focus_relation.
_VERB_LEMMA_MAP: dict[str, str] = {
    "drives": "drive",
    "driving": "drive",
    "driven": "drive",
    "researches": "research",
    "researching": "research",
    "researched": "research",
    "restores": "restore",
    "restoring": "restore",
    "restored": "restore",
    "pursues": "pursue",
    "pursuing": "pursue",
    "pursued": "pursue",
    "finds": "find",
    "finding": "find",
    "found": "find",
    "passes": "pass",
    "passing": "pass",
    "passed": "pass",
    "satisfies": "satisfy",
    "satisfying": "satisfy",
    "satisfied": "satisfy",
    "grows": "grow",
    "growing": "grow",
    "grew": "grow",
    "grown": "grow",
    "builds": "build",
    "building": "build",
    "built": "build",
    "buys": "buy",
    "buying": "buy",
    "bought": "buy",
    "sells": "sell",
    "selling": "sell",
    "sold": "sell",
    "likes": "like",
    "liking": "like",
    "liked": "like",
    "enjoys": "enjoy",
    "enjoying": "enjoy",
    "enjoyed": "enjoy",
    "hates": "hate",
    "hating": "hate",
    "hated": "hate",
    "prefers": "prefer",
    "preferring": "prefer",
    "preferred": "prefer",
    "moves": "move",
    "moving": "move",
    "moved": "move",
    "joins": "join",
    "joining": "join",
    "joined": "join",
    "leaves": "leave",
    "leaving": "leave",
    "left": "leave",
    "starts": "start",
    "starting": "start",
    "started": "start",
    "stops": "stop",
    "stopping": "stop",
    "stopped": "stop",
    "retires": "retire",
    "retiring": "retire",
    "retired": "retire",
    "dies": "die",
    "dying": "die",
    "died": "die",
    "works": "work",
    "working": "work",
    "worked": "work",
    "lives": "live",
    "living": "live",
    "lived": "live",
    "studies": "study",
    "studying": "study",
    "studied": "study",
    "knows": "know",
    "knowing": "know",
    "knew": "know",
    "known": "know",
    "meets": "meet",
    "meeting": "meet",
    "met": "meet",
    "loves": "love",
    "loving": "love",
    "loved": "love",
    "marries": "marry",
    "marrying": "marry",
    "married": "marry",
    "plays": "play",
    "playing": "play",
    "played": "play",
    "attends": "attend",
    "attending": "attend",
    "attended": "attend",
    "visits": "visit",
    "visiting": "visit",
    "visited": "visit",
    "reads": "read",
    "reading": "read",
    "watches": "watch",
    "watching": "watch",
    "watched": "watch",
    "drinks": "drink",
    "drinking": "drink",
    "drank": "drink",
    "drunk": "drink",
    "eats": "eat",
    "eating": "eat",
    "ate": "eat",
    "eaten": "eat",
    "uses": "use",
    "using": "use",
    "used": "use",
}

# Compound-phrase patterns that map a multi-token question fragment to the
# materializer's compound relation name.  Checked before single-token lemma
# lookup so that "find … satisfying" wins over plain "find".
_COMPOUND_RELATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:find|finds|found|finding)\b.*\bsatisfying\b", re.I), "finds_satisfying"),
    (re.compile(r"\b(?:move|moves|moved|moving|relocate|relocated)\b\s+to\b", re.I), "moved_to"),
    (re.compile(r"\b(?:work|works|worked|working)\b\s+as\b", re.I), "role"),
    (re.compile(r"\b(?:pass|passes|passed|passing)\b\s+away\b", re.I), "passed_away"),
]


def _tokenize_lower(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _extract_focus_entities(question: str) -> list[str]:
    """Extract named entities from the question (proper-name heuristic).

    Handles single-token names (Alice, Bob) and strips possessive suffixes
    (Alice's → Alice).  Question-opener words (What, When, Would, Did, …)
    are excluded even when they appear as the first word in a multi-word span
    (e.g. "Would Caroline" → entity "Caroline").
    """
    seen: set[str] = set()
    entities: list[str] = []
    for m in _NAMED_ENTITY_RE.finditer(question):
        raw = m.group(1)
        # Strip possessive suffix.
        if raw.endswith("'s"):
            raw = raw[:-2]
        elif raw.endswith("'"):
            raw = raw[:-1]
        # Strip leading stop-cap words (e.g. "Would Caroline" → "Caroline").
        words = raw.split()
        while words and words[0] in _STOP_CAPS:
            words = words[1:]
        e = " ".join(words)
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

    # Structural SET signals — explicit aggregation cues only, no plural-noun heuristic.
    if tokens and tokens[0] == "what":
        tail = tokens[1:4]
        if "all" in tail or "list" in tail or "enumerate" in tail:
            return AnswerSpace.SET

    # Imperative aggregation: "name every …", "list …", "enumerate …"
    if tokens and tokens[0] in {"name", "list", "enumerate"}:
        return AnswerSpace.SET

    # "what are all …" — explicit all-enumeration phrasing
    if "all" in token_set and "what" in token_set:
        return AnswerSpace.SET

    # "which are …" phrase
    if ("which are" in q_lower or "what are" in q_lower) and (
        "all" in token_set or "list" in token_set or "enumerate" in token_set
    ):
        return AnswerSpace.SET

    # Implicit SET: conservative — only fire when a known aggregation noun head
    # appears in "what/which + has/have/does/do" structure.
    # e.g. "What books has X read?" / "What hobbies does Alice have?"
    if tokens and tokens[0] in {"what", "which"}:
        early = set(tokens[1:5])
        if early & {"has", "have", "does", "do"}:
            rest_tokens = tokens[1:]
            if any(noun in rest_tokens for noun in _SET_NOUN_HEADS):
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
    """Try to extract a verb-like focus relation from the question.

    Checks compound-phrase patterns first (e.g. "find … satisfying" → "finds_satisfying"),
    then normalizes inflected verb forms via _VERB_LEMMA_MAP, then falls back to
    _RELATION_VERBS for single-token lemmas.
    Returns the canonical compound or lemma form.
    """
    if re.search(r"\bwhat(?:'s| is| was)\b.+\blike\b\s*\??\s*$", question, re.I):
        return None
    for pattern, compound_relation in _COMPOUND_RELATION_PATTERNS:
        if pattern.search(question):
            return compound_relation
    tokens = _tokenize_lower(question)
    for t in tokens:
        lemma = _VERB_LEMMA_MAP.get(t)
        if lemma:
            return lemma
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
