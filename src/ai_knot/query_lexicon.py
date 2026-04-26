"""Frame Lexical Bridge — intent-aware query expansion without LLM."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FrameDef:
    """One lexical frame: a cluster of semantically related terms with weights."""

    terms: dict[str, float]  # term → weight (all < 1.0)
    intents: frozenset[str]  # RecallIntent values this frame applies to


# LEXICON: 6 frame families
# Rules:
#   - ALL weights MUST be < 1.0
#   - NO LOCOMO names, gold answers, or dataset-specific terms
#   - NAVIGATIONAL intent never gets any expansion (enforced in expand_query_lexically)
#   - Terms must be generic (applicable to any personal memory domain)
LEXICON: dict[str, FrameDef] = {
    "activity_sport": FrameDef(
        terms={
            "practice": 0.7,
            "play": 0.7,
            "train": 0.6,
            "compete": 0.5,
            "game": 0.6,
            "match": 0.5,
            "exercise": 0.5,
            "workout": 0.4,
        },
        intents=frozenset({"factual", "aggregational", "exploratory"}),
    ),
    "preference_opinion": FrameDef(
        terms={
            "prefer": 0.7,
            "like": 0.6,
            "enjoy": 0.6,
            "love": 0.7,
            "favorite": 0.8,
            "dislike": 0.5,
            "hate": 0.5,
            "want": 0.5,
        },
        intents=frozenset({"factual", "aggregational"}),
    ),
    "location_place": FrameDef(
        terms={
            "visit": 0.6,
            "go": 0.5,
            "travel": 0.6,
            "live": 0.7,
            "stay": 0.5,
            "move": 0.5,
            "relocate": 0.6,
        },
        intents=frozenset({"factual", "aggregational", "exploratory"}),
    ),
    "work_career": FrameDef(
        terms={
            "work": 0.7,
            "job": 0.7,
            "career": 0.7,
            "hire": 0.6,
            "employ": 0.6,
            "role": 0.6,
            "position": 0.5,
            "company": 0.6,
        },
        intents=frozenset({"factual", "aggregational"}),
    ),
    "event_temporal": FrameDef(
        terms={
            "happen": 0.6,
            "occur": 0.6,
            "attend": 0.7,
            "celebrate": 0.6,
            "meet": 0.6,
            "visit": 0.5,
            "join": 0.5,
        },
        intents=frozenset({"factual", "exploratory"}),
    ),
    "relationship_social": FrameDef(
        terms={
            "friend": 0.7,
            "family": 0.7,
            "partner": 0.6,
            "colleague": 0.5,
            "meet": 0.5,
            "date": 0.5,
            "marry": 0.7,
            "divorce": 0.7,
        },
        intents=frozenset({"factual", "aggregational", "exploratory"}),
    ),
}


@dataclass
class LexicalExpansion:
    """Result of query expansion."""

    original_query: str
    intent: str
    expansion_weights: dict[str, float] = field(default_factory=dict)
    frames_applied: list[str] = field(default_factory=list)
    terms_added: int = 0


def expand_query_lexically(
    query: str,
    intent: str,
    *,
    max_terms_per_intent: int = 8,
) -> LexicalExpansion:
    """Expand query with frame terms relevant to the given intent.

    NAVIGATIONAL intent always returns empty expansion (no terms added).
    All returned weights are < 1.0.

    Args:
        query: Original query string.
        intent: RecallIntent value as string.
        max_terms_per_intent: Cap on total expansion terms.

    Returns:
        LexicalExpansion with expansion_weights dict.
    """
    # NAVIGATIONAL: never expand (would add noise to artifact search)
    if intent == "navigational":
        return LexicalExpansion(
            original_query=query,
            intent=intent,
            expansion_weights={},
            frames_applied=[],
            terms_added=0,
        )

    q_lower = query.lower()
    expansion_weights: dict[str, float] = {}
    frames_applied: list[str] = []

    for frame_name, frame in LEXICON.items():
        if intent not in frame.intents:
            continue
        # Only apply frame if query token overlaps with frame terms or frame name tokens.
        frame_name_tokens = set(frame_name.replace("_", " ").split())
        query_tokens = set(q_lower.split())
        frame_term_set = set(frame.terms.keys())
        if (
            not (query_tokens & frame_term_set)
            and not (query_tokens & frame_name_tokens)
            and not any(t in q_lower for t in frame.terms)
        ):
            continue
        frames_applied.append(frame_name)
        for term, weight in frame.terms.items():
            if term not in q_lower:  # don't re-add terms already in query
                expansion_weights[term] = max(expansion_weights.get(term, 0.0), weight)

    # Cap total terms by keeping highest-weight terms
    if len(expansion_weights) > max_terms_per_intent:
        sorted_terms = sorted(expansion_weights.items(), key=lambda x: -x[1])
        expansion_weights = dict(sorted_terms[:max_terms_per_intent])

    return LexicalExpansion(
        original_query=query,
        intent=intent,
        expansion_weights=expansion_weights,
        frames_applied=frames_applied,
        terms_added=len(expansion_weights),
    )
