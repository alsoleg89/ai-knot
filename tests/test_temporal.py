"""Tests for the general relative-time resolver (_temporal.py)."""

from __future__ import annotations

from datetime import date, datetime

from ai_knot._temporal import (
    ResolvedDate,
    format_event_date,
    resolve_event_dates,
)

# 8 May 2023 is a Monday — used as the common anchor below.
ANCHOR = date(2023, 5, 8)


def _top(text: str, anchor: date = ANCHOR) -> ResolvedDate | None:
    res = resolve_event_dates(text, anchor)
    return res[0] if res else None


class TestDayGrained:
    def test_yesterday(self) -> None:
        assert _top("I went to the gym yesterday").value == date(2023, 5, 7)

    def test_tomorrow(self) -> None:
        assert _top("the show is tomorrow").value == date(2023, 5, 9)

    def test_day_before_yesterday(self) -> None:
        assert _top("we met the day before yesterday").value == date(2023, 5, 6)

    def test_n_days_ago(self) -> None:
        assert _top("it happened 3 days ago").value == date(2023, 5, 5)


class TestWeekMonthYear:
    def test_last_week(self) -> None:
        r = _top("I gave a speech at school last week")
        assert r is not None and r.value == date(2023, 5, 1) and r.granularity == "week"

    def test_next_month(self) -> None:
        r = _top("we're going camping next month")
        assert r is not None and r.value == date(2023, 6, 8) and r.granularity == "month"

    def test_this_month(self) -> None:
        r = _top("I lost my job this month")
        assert r is not None and r.granularity == "month" and r.value.month == 5

    def test_two_weeks_ago(self) -> None:
        assert _top("lost my job two weeks ago").value == date(2023, 4, 24)

    def test_three_months_ago_word(self) -> None:
        assert _top("started three months ago").value == date(2023, 2, 8)

    def test_in_two_weeks_future(self) -> None:
        assert _top("the event is in two weeks").value == date(2023, 5, 22)


class TestWeekdays:
    def test_last_saturday(self) -> None:
        # Anchor Monday 8 May -> previous Saturday = 6 May
        assert _top("ran a charity race last Saturday").value == date(2023, 5, 6)

    def test_weekday_abbreviation(self) -> None:
        # "last Fri" -> previous Friday = 5 May
        assert _top("I finally took a pottery class last Fri").value == date(2023, 5, 5)

    def test_next_friday(self) -> None:
        assert _top("the deadline is next Friday").value == date(2023, 5, 12)


class TestPresentAndDuration:
    def test_present_tense_just(self) -> None:
        r = _top("I just signed up for a pottery class")
        assert r is not None and r.value == ANCHOR

    def test_duration_now_to_start(self) -> None:
        # "seven years now" -> started 7 years before anchor
        assert _top("Seven years now and I love it").value == date(2016, 5, 8)


class TestNoExpression:
    def test_empty_when_no_relative_time(self) -> None:
        assert resolve_event_dates("I love hiking and painting", ANCHOR) == []

    def test_accepts_datetime_anchor(self) -> None:
        r = _top("yesterday", datetime(2023, 5, 8, 13, 56))  # type: ignore[arg-type]
        assert r is not None and r.value == date(2023, 5, 7)


class TestFormatting:
    def test_format_day(self) -> None:
        r = ResolvedDate(date(2023, 5, 7), "day", "yesterday", 0.97)
        assert format_event_date(r) == "7 May 2023"

    def test_format_month(self) -> None:
        r = ResolvedDate(date(2023, 6, 1), "month", "next month", 0.85)
        assert format_event_date(r) == "June 2023"

    def test_format_year(self) -> None:
        r = ResolvedDate(date(2016, 5, 8), "year", "seven years now", 0.8)
        assert format_event_date(r) == "2016"


class TestConfidenceOrdering:
    def test_explicit_beats_bare_weekday(self) -> None:
        # "yesterday" (0.97) should outrank a bare weekday mention (0.55)
        res = resolve_event_dates("yesterday we talked about Friday plans", ANCHOR)
        assert res[0].phrase == "yesterday"
