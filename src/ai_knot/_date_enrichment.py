"""Date-tag enrichment for Fact objects.

Parses dates from fact.content using pure-regex patterns (no external deps)
and injects canonical forms into fact.tags so BM25F can match temporal queries.

The tag field has weight _W_TAGS = 2.0 in InvertedIndex — date tags therefore
receive a direct boost when recall queries contain date tokens.

Exported symbols
----------------
enrich_date_tags : in-place enrichment of a single Fact (public for tests).
"""

from __future__ import annotations

import contextlib
import re

from ai_knot.types import Fact

# ---------------------------------------------------------------------------
# Month lookup (abbreviations + full names)
# ---------------------------------------------------------------------------

_MONTH_MAP: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

# Ordered list for month-number → name conversion.
_MONTH_NAMES: list[str] = [
    "",
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]

# ---------------------------------------------------------------------------
# Date patterns — multi-component only to prevent false positives
# (a bare "2023" from "Room 2023" won't match any of these)
# ---------------------------------------------------------------------------

# "[27 June, 2023]" or "27 June, 2023"  → groups: day, month_name, year
_DATE_DMY = re.compile(r"\[?\s*(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})\s*\]?")

# "June 27, 2023"  → groups: month_name, day, year
_DATE_MDY = re.compile(r"\b([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})\b")

# ISO "2023-06-27"  → groups: year, month_num, day
_DATE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

# "June 2023" (month-year, no day)  → groups: month_name, year
# Requires ≥ 3-letter month name to avoid common words ("a 2023").
_DATE_MY = re.compile(r"\b([A-Za-z]{3,})\s+(\d{4})\b")

# Maximum number of tags after enrichment (defensive cap).
_MAX_TAGS = 10


def enrich_date_tags(fact: Fact) -> Fact:
    """Add canonical date forms to fact.tags in-place.

    Mode-agnostic: works on ``fact.content`` regardless of ingest path (raw /
    dated / learn).  Tags are indexed by BM25F at weight 2.0, so date-queries
    are matched more reliably for temporal and multi-hop (Cat2) questions.

    Canonical forms injected for each parsed date:
    - ISO day  ``"2023-06-27"``     (only when day is available)
    - Month-year ``"june 2023"``
    - Month only ``"june"``
    - Year only  ``"2023"``

    If the fact already has ``_MAX_TAGS`` tags, no new date tags are added.

    Returns:
        The same ``Fact`` object (modified in place).
    """
    if not fact.content:
        return fact

    sources = [fact.content]
    if fact.witness_surface:
        sources.append(fact.witness_surface)

    # Collected: (year, month_int, day_int | None)
    parsed: list[tuple[int, int, int | None]] = []

    for text in sources:
        # "[27 June, 2023]" style (most common in dated ingest)
        for m in _DATE_DMY.finditer(text):
            month_num = _MONTH_MAP.get(m.group(2).lower())
            if month_num:
                with contextlib.suppress(ValueError):
                    parsed.append((int(m.group(3)), month_num, int(m.group(1))))

        # "June 27, 2023" style
        for m in _DATE_MDY.finditer(text):
            month_num = _MONTH_MAP.get(m.group(1).lower())
            if month_num:
                with contextlib.suppress(ValueError):
                    parsed.append((int(m.group(3)), month_num, int(m.group(2))))

        # ISO "2023-06-27"
        for m in _DATE_ISO.finditer(text):
            with contextlib.suppress(ValueError):
                y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if 1 <= mo <= 12 and 1 <= d <= 31:
                    parsed.append((y, mo, d))

        # "June 2023" (month-year only)
        for m in _DATE_MY.finditer(text):
            month_num = _MONTH_MAP.get(m.group(1).lower())
            if month_num:
                with contextlib.suppress(ValueError):
                    parsed.append((int(m.group(2)), month_num, None))

    if not parsed:
        return fact

    new_tags: list[str] = list(fact.tags)
    seen: set[str] = set(new_tags)

    for year, month, day in parsed:
        month_name = _MONTH_NAMES[month]  # e.g. "june"
        my = f"{month_name} {year}"  # e.g. "june 2023"
        year_str = str(year)

        candidates: list[str] = [my, month_name, year_str]
        if day is not None:
            iso = f"{year:04d}-{month:02d}-{day:02d}"
            candidates = [iso] + candidates

        for t in candidates:
            if t not in seen and len(new_tags) < _MAX_TAGS:
                new_tags.append(t)
                seen.add(t)

    fact.tags = new_tags
    return fact
