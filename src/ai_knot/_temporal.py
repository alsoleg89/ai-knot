"""General relative-time resolution for temporal memory.

Given a span of text and the timestamp at which the memory was formed (its
``event_time`` anchor), resolve relative temporal expressions — "yesterday",
"last week", "two months ago", "next Friday", "this month" — into absolute
calendar dates.

This is a *general* English temporal-grammar resolver (the same class of
capability as ``dateparser`` / Duckling), NOT tied to any benchmark.  In
production the anchor is ``now()`` at ingestion time; for historical import
(e.g. replaying old chat logs) it is each message's original timestamp.

The resolver is deliberately stdlib-only (no third-party date library) to keep
the package's dependency surface minimal.

Exported symbols
----------------
ResolvedDate         : a single resolved (date, granularity, phrase) triple.
resolve_event_dates  : resolve all relative expressions in a text span.
format_event_date    : human-facing rendering of a resolved date.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

# ---------------------------------------------------------------------------
# Vocabulary tables (general English relative-time grammar)
# ---------------------------------------------------------------------------

_WEEKDAYS: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
    # common abbreviations
    "mon": 0,
    "tue": 1,
    "tues": 1,
    "wed": 2,
    "thu": 3,
    "thur": 3,
    "thurs": 3,
    "fri": 4,
    "sat": 5,
    "sun": 6,
}

# unit name -> number of base units (days) where fixed, else handled specially
_NUMBER_WORDS: dict[str, int] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "couple": 2,
    "few": 3,
    "several": 3,
}

_GRAN_ORDER = {"day": 0, "week": 1, "month": 2, "year": 3}


@dataclass(frozen=True)
class ResolvedDate:
    """A relative expression resolved against an anchor.

    Attributes:
        value: The absolute calendar date the expression resolves to.
        granularity: Precision of the answer — "day", "week", "month", "year".
        phrase: The matched relative expression (for provenance/debugging).
        confidence: 0.0-1.0; explicit offsets score high, bare/implicit lower.
    """

    value: date
    granularity: str
    phrase: str
    confidence: float


# ---------------------------------------------------------------------------
# stdlib calendar arithmetic (avoids a dateutil dependency)
# ---------------------------------------------------------------------------


def _shift_months(d: date, months: int) -> date:
    """Return *d* shifted by *months* (may be negative), clamping the day."""
    total = d.month - 1 + months
    year = d.year + total // 12
    month = total % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _shift_years(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # Feb 29 -> non-leap year
        return d.replace(year=d.year + years, day=28)


def _prev_weekday(anchor: date, weekday: int, weeks_back: int = 1) -> date:
    """Most recent past occurrence of *weekday* (0=Mon), going *weeks_back* back."""
    diff = (anchor.weekday() - weekday) % 7
    diff = 7 if diff == 0 else diff
    return anchor - timedelta(days=diff + 7 * (weeks_back - 1))


def _next_weekday(anchor: date, weekday: int) -> date:
    diff = (weekday - anchor.weekday()) % 7
    diff = 7 if diff == 0 else diff
    return anchor + timedelta(days=diff)


def _num(token: str | None) -> int:
    if token is None:
        return 1
    token = token.lower()
    return int(token) if token.isdigit() else _NUMBER_WORDS.get(token, 1)


def _as_date(anchor: datetime | date) -> date:
    return anchor.date() if isinstance(anchor, datetime) else anchor


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

_RE_N_AGO = re.compile(
    r"\b(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|couple|few|several)\s+"
    r"(day|week|month|year)s?\s+(ago|before|earlier|back|prior)\b"
)
_RE_IN_N = re.compile(
    r"\b(?:in|after)\s+(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten|couple|few)\s+"
    r"(day|week|month|year)s?\b"
)
_RE_DURATION_NOW = re.compile(
    r"\b(?:for\s+|since\s+)?(\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(year|month|week|day)s?\s+(?:now|already|so far)\b"
)
_RE_PRESENT = re.compile(
    r"\b(just|recently|today|right now|currently|these days|"
    r"this (?:morning|afternoon|evening))\b"
)
_NUM_OR_WORD = r"(\d+|a|an|one|two|three|four|couple|few)"


def resolve_event_dates(text: str, anchor: datetime | date) -> list[ResolvedDate]:
    """Resolve all relative temporal expressions in *text* against *anchor*.

    Args:
        text: The memory content (may contain 0..N relative expressions).
        anchor: The memory's event_time — when it was recorded/uttered.

    Returns:
        ResolvedDate list, ordered by descending confidence then specificity.
        Empty if no relative expression is present.
    """
    base = _as_date(anchor)
    t = text.lower()
    out: list[ResolvedDate] = []

    def add(value: date, gran: str, phrase: str, conf: float) -> None:
        out.append(ResolvedDate(value, gran, phrase, conf))

    # --- day-grained absolutes -------------------------------------------
    if re.search(r"\bday before yesterday\b", t):
        add(base - timedelta(days=2), "day", "day before yesterday", 0.95)
    if re.search(r"(?<!day before )\byesterday\b", t):
        add(base - timedelta(days=1), "day", "yesterday", 0.97)
    if re.search(r"\btomorrow\b", t):
        add(base + timedelta(days=1), "day", "tomorrow", 0.95)

    # --- "N <unit> ago/before" -------------------------------------------
    for m in _RE_N_AGO.finditer(t):
        unit = m.group(2)
        n = _num(m.group(1))
        add(_shift(base, unit, -n), unit, m.group(0), 0.9)

    # --- "in N <unit>" (future) ------------------------------------------
    for m in _RE_IN_N.finditer(t):
        unit = m.group(2)
        n = _num(m.group(1))
        add(_shift(base, unit, n), unit, m.group(0), 0.85)

    # --- duration-to-start: "seven years now" => anchor - 7y -------------
    for m in _RE_DURATION_NOW.finditer(t):
        unit = m.group(2)
        n = _num(m.group(1))
        add(_shift(base, unit, -n), unit, m.group(0), 0.8)

    # --- last/next/this <week|month|year> --------------------------------
    for kw, delta in (("last", -1), ("past", -1), ("next", 1), ("this", 0)):
        for unit in ("week", "month", "year"):
            if re.search(rf"\b{kw}\s+{unit}\b", t):
                add(_shift(base, unit, delta), unit, f"{kw} {unit}", 0.85)

    # --- weekend ----------------------------------------------------------
    if re.search(r"\b(last|past)\s+weekend\b", t):
        add(_prev_weekday(base, 5, 1), "week", "last weekend", 0.8)
    for m in re.finditer(rf"\b{_NUM_OR_WORD}\s+weekends?\s+(?:ago|before|back|earlier)\b", t):
        add(_prev_weekday(base, 5, _num(m.group(1))), "week", m.group(0), 0.78)
    if re.search(r"\b(the\s+)?weekend before\b", t):
        add(_prev_weekday(base, 5, 1), "week", "the weekend before", 0.75)

    # --- weekdays: last/past/this/next/bare ------------------------------
    for name, wi in _WEEKDAYS.items():
        if re.search(rf"\b(?:last|past|this)\s+{name}\b", t):
            add(_prev_weekday(base, wi, 1), "day", f"last {name}", 0.85)
        elif re.search(rf"\bnext\s+{name}\b", t):
            add(_next_weekday(base, wi), "day", f"next {name}", 0.82)
        elif re.search(rf"\b{name}\b", t):
            # bare weekday: assume most recent past occurrence
            add(_prev_weekday(base, wi, 1), "day", name, 0.55)

    # --- present-tense / "just" => the event is happening at the anchor ---
    if _RE_PRESENT.search(t):
        add(base, "day", "present-tense", 0.6)

    # de-duplicate (same value+granularity) keeping highest confidence,
    # then order by confidence desc, finer granularity first.
    best: dict[tuple[date, str], ResolvedDate] = {}
    for r in out:
        key = (r.value, r.granularity)
        if key not in best or r.confidence > best[key].confidence:
            best[key] = r
    return sorted(
        best.values(),
        key=lambda r: (-r.confidence, _GRAN_ORDER.get(r.granularity, 9)),
    )


def _shift(base: date, unit: str, amount: int) -> date:
    if unit == "day":
        return base + timedelta(days=amount)
    if unit == "week":
        return base + timedelta(weeks=amount)
    if unit == "month":
        return _shift_months(base, amount)
    if unit == "year":
        return _shift_years(base, amount)
    raise ValueError(f"unknown unit: {unit}")


def format_event_date(r: ResolvedDate) -> str:
    """Render a resolved date for prompt injection at its granularity."""
    if r.granularity == "year":
        return r.value.strftime("%Y")
    if r.granularity == "month":
        return r.value.strftime("%B %Y")
    return r.value.strftime("%-d %B %Y")
