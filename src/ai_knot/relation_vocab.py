"""Canonical relation vocabulary for ai-knot.

Single source of truth for relation names, lemma normalization, aliases,
and compound relation detection.  Consumed by query_contract, support_retrieval,
and query_operators — never import private dicts from those modules directly.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Canonical relation verbs (infinitive/base form).
# These are the forms stored as claim.relation by the materializer.
# ---------------------------------------------------------------------------

RELATION_VERBS: frozenset[str] = frozenset(
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
        # first-person event relations added in v5
        "attended",
        "joined",
        "signed_up_for",
        "applied_to",
        "visited",
        "met_with",
        "bought",
        "ran",
        "created",
        "spoke_at",
    }
)

# ---------------------------------------------------------------------------
# Inflected form → canonical lemma.
# ---------------------------------------------------------------------------

VERB_LEMMA_MAP: dict[str, str] = {
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
    # first-person event inflections
    "signs": "sign",
    "signing": "sign",
    "signed": "sign",
    "applies": "apply",
    "applying": "apply",
    "applied": "apply",
    "runs": "run",
    "running": "run",
    "ran": "run",
    "creates": "create",
    "creating": "create",
    "created": "create",
    "speaks": "speak",
    "speaking": "speak",
    "spoke": "speak",
}

# ---------------------------------------------------------------------------
# Compound-phrase patterns: question fragment → materializer compound relation.
# Checked before single-token lemma so "find … satisfying" wins over "find".
# ---------------------------------------------------------------------------

COMPOUND_RELATION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:find|finds|found|finding)\b.*\bsatisfying\b", re.I), "finds_satisfying"),
    (re.compile(r"\b(?:move|moves|moved|moving|relocate|relocated)\b\s+to\b", re.I), "moved_to"),
    (re.compile(r"\b(?:work|works|worked|working)\b\s+as\b", re.I), "role"),
    (re.compile(r"\b(?:pass|passes|passed|passing)\b\s+away\b", re.I), "passed_away"),
    (re.compile(r"\b(?:sign|signs|signed|signing)\b.*\b(?:up\s+for|up)\b", re.I), "signed_up_for"),
    (re.compile(r"\b(?:apply|applies|applied|applying)\b.*\bto\b", re.I), "applied_to"),
    (re.compile(r"\b(?:meet|meets|met|meeting)\b.*\bwith\b", re.I), "met_with"),
    (re.compile(r"\b(?:speak|speaks|spoke|speaking)\b.*\bat\b", re.I), "spoke_at"),
]

# ---------------------------------------------------------------------------
# Aliases: canonical query lemma → compound relation names stored by materializer.
# Used by retrieval (topic fan-out) and operators (relevance matching).
# ---------------------------------------------------------------------------

RELATION_ALIASES: dict[str, tuple[str, ...]] = {
    "find": ("finds_satisfying",),
    "like": ("likes",),
    "love": ("likes",),
    "enjoy": ("likes",),
    "hate": ("dislikes",),
    "dislike": ("dislikes",),
    "drive": ("drives",),
    "move": ("moved_to",),
    "work": ("works_at", "works_as"),
    "pass": ("passed_away",),
    "sign": ("signed_up_for",),
    "apply": ("applied_to",),
    "meet": ("met_with",),
    "speak": ("spoke_at",),
    "attend": ("attended",),
    "visit": ("visited",),
    "buy": ("bought",),
    "join": ("joined",),
    "read": ("read",),
    "run": ("ran",),
    "create": ("created",),
}

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def canonical_relation_for_token(token: str) -> str | None:
    """Return the canonical relation for a single token, or None.

    Checks VERB_LEMMA_MAP first (inflected forms), then compound patterns
    (single token can't trigger those), then RELATION_VERBS directly.
    """
    # Direct lemma lookup
    lemma = VERB_LEMMA_MAP.get(token.lower())
    if lemma:
        return lemma
    lower = token.lower()
    if lower in RELATION_VERBS:
        return lower
    return None


def canonical_relation_for_phrase(phrase: str) -> str | None:
    """Return the canonical relation for a multi-token phrase, or None.

    Checks compound patterns first, then falls back to per-token lookup.
    """
    for pattern, compound in COMPOUND_RELATION_PATTERNS:
        if pattern.search(phrase):
            return compound
    tokens = re.findall(r"[a-z']+", phrase.lower())
    for t in tokens:
        result = canonical_relation_for_token(t)
        if result:
            return result
    return None


def alias_relations(canonical: str) -> tuple[str, ...]:
    """Return compound aliases for a canonical relation (empty tuple if none)."""
    return RELATION_ALIASES.get(canonical, ())


def matches_relation(claim_relation: str | None, focus_relation: str | None) -> bool:
    """Return True if claim_relation matches focus_relation (with alias expansion).

    Handles:
    - Exact match: claim_relation == focus_relation
    - Alias match: claim_relation in alias_relations(focus_relation)
    - Reverse alias: focus_relation in alias_relations(claim_relation)
    - None on either side → False
    """
    if not claim_relation or not focus_relation:
        return False
    if claim_relation == focus_relation:
        return True
    if claim_relation in RELATION_ALIASES.get(focus_relation, ()):
        return True
    return focus_relation in RELATION_ALIASES.get(claim_relation, ())
