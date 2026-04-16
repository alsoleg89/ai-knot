"""Shared text-level guards for discourse noise detection.

These functions identify low-signal text patterns that should not be
materialized as facts or should receive a penalty in ranking.
"""

from __future__ import annotations

import re

# Only the three deictic pronouns — nothing broader.
_DEICTIC_SUBJECTS: frozenset[str] = frozenset({"that", "it", "this"})

# Short evaluative predicates pattern.  Fires only on a tight surface form.
_EVALUATIVE_RE = re.compile(
    r"^(?:so |really |truly |absolutely )?"
    r"(?:amazing|wonderful|cool|awesome|nice|great|fantastic|brilliant"
    r"|terrible|horrible|awful|bad|good|interesting|exciting|fun)\b"
    r"[\s!.]*$",
    re.IGNORECASE,
)


def is_deictic_subject(subject: str) -> bool:
    """Return True if subject is a bare deictic pronoun (that/it/this)."""
    return subject.strip().lower() in _DEICTIC_SUBJECTS


def is_evaluative_predicate(value: str) -> bool:
    """Return True if value matches a short evaluative adjective pattern."""
    return bool(_EVALUATIVE_RE.match(value.strip()))
