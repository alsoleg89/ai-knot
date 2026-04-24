"""EntityGroupoid — entity identity and canonical orbit resolution.

Sprint 3: canonical string-based identity.
Sprint 3b: holonomy detection for identity merge loops.
"""

from __future__ import annotations

import re
import unicodedata


def _normalize_str(s: str) -> str:
    """Lowercase, strip articles, collapse whitespace, remove non-word chars."""
    s = unicodedata.normalize("NFKC", s).lower().strip()
    s = re.sub(r"\b(the|a|an)\b\s*", "", s)
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", "_", s.strip())


class EntityGroupoid:
    """Maps surface entity strings to canonical orbit IDs.

    Sprint 3: exact string matching after normalization.
    Sprint 3b: holonomy detection (detect identity loops / closed-loop merges).
    """

    def __init__(self) -> None:
        self._orbits: dict[str, str] = {}  # normalized_form -> orbit_id
        # Merge edges: orbit_id → orbit_id (directed graph for holonomy check)
        self._merge_edges: dict[str, str] = {}

    def resolve(self, surface: str) -> str:
        """Return canonical orbit_id for the given surface form.

        Creates a new orbit if the surface has not been seen before.
        """
        norm = _normalize_str(surface)
        if norm not in self._orbits:
            self._orbits[norm] = f"entity:{norm}"
        return self._orbits[norm]

    def merge(self, surface_a: str, surface_b: str) -> str:
        """Declare that two surface forms refer to the same entity.

        Returns the canonical orbit_id (surface_a wins).
        Records the merge edge for holonomy detection.
        """
        orbit_a = self.resolve(surface_a)
        orbit_b_old = self.resolve(surface_b)
        norm_b = _normalize_str(surface_b)
        self._orbits[norm_b] = orbit_a
        # Record merge edge only if the orbits differ (avoid self-loops)
        if orbit_b_old != orbit_a:
            self._merge_edges[orbit_b_old] = orbit_a
        return orbit_a

    def has_holonomy(self) -> bool:
        """Return True if any merge chain contains a closed loop.

        A holonomy (identity loop) means merges A→B→C→…→A, which indicates
        contradictory identity resolution that requires manual disambiguation.
        """
        # Floyd's cycle detection on the merge edge graph
        for start in self._merge_edges:
            visited: set[str] = set()
            current = start
            while current in self._merge_edges:
                if current in visited:
                    return True
                visited.add(current)
                current = self._merge_edges[current]
        return False

    def holonomy_orbits(self) -> list[str]:
        """Return list of orbit IDs that participate in merge loops."""
        cycles: list[str] = []
        for start in self._merge_edges:
            visited: set[str] = set()
            current = start
            chain: list[str] = []
            while current in self._merge_edges:
                if current in visited:
                    cycles.append(current)
                    break
                visited.add(current)
                chain.append(current)
                current = self._merge_edges[current]
        return cycles

    def known_orbits(self) -> set[str]:
        return set(self._orbits.values())


def resolve_speaker_entity(speaker: str, user_id: str | None, agent_id: str) -> str:
    """Map first-person pronouns to the appropriate entity orbit."""
    if speaker == "user":
        return f"entity:{user_id}" if user_id else "entity:unknown_user"
    if speaker == "agent":
        return f"entity:{agent_id}"
    return "entity:system"


_FIRST_PERSON = re.compile(r"^(i|me|my|myself|mine|we|us|our|ours)$", re.I)
_THIRD_PERSON_MASC = re.compile(r"^(he|him|his|himself)$", re.I)
_THIRD_PERSON_FEM = re.compile(r"^(she|her|hers|herself)$", re.I)
_THIRD_PERSON_NEUT = re.compile(r"^(they|them|their|theirs|themselves|it|its|itself)$", re.I)


def pronoun_entity(pronoun: str, speaker_orbit: str, last_mentioned: str | None) -> str:
    """Resolve a pronoun to an entity orbit (best-effort, Sprint 3 level)."""
    if _FIRST_PERSON.match(pronoun):
        return speaker_orbit
    if _THIRD_PERSON_MASC.match(pronoun) or _THIRD_PERSON_FEM.match(pronoun):
        return last_mentioned or "entity:unknown"
    if _THIRD_PERSON_NEUT.match(pronoun):
        return last_mentioned or "entity:unknown"
    return "entity:unknown"
