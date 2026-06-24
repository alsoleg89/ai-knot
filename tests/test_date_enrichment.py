"""Unit tests for ai_knot._date_enrichment — date-tag injection for BM25F."""

from __future__ import annotations

import pytest

from ai_knot._date_enrichment import enrich_date_tags
from ai_knot.types import Fact


def _tags(content: str, *, witness_surface: str = "") -> list[str]:
    fact = Fact(content=content, witness_surface=witness_surface)
    enrich_date_tags(fact)
    return fact.tags


class TestDMYStyle:
    """Bracket / no-bracket day-month-year — the dated-mode prefix format."""

    def test_bracket_dmy_with_full_month(self) -> None:
        tags = _tags("[27 June, 2023] Caroline went jogging")
        assert "2023-06-27" in tags
        assert "june 2023" in tags
        assert "june" in tags
        assert "2023" in tags

    def test_dmy_short_month_abbrev(self) -> None:
        tags = _tags("8 Jan, 2024 update on the project")
        assert "2024-01-08" in tags
        assert "january 2024" in tags
        assert "january" in tags
        assert "2024" in tags

    def test_dmy_no_comma(self) -> None:
        tags = _tags("[3 March 2022] event")
        assert "2022-03-03" in tags


class TestMDYStyle:
    """American month-day-year style."""

    def test_mdy_full_month(self) -> None:
        tags = _tags("Project started June 27, 2023 with kickoff")
        assert "2023-06-27" in tags
        assert "june 2023" in tags

    def test_mdy_short_month(self) -> None:
        tags = _tags("Sep 5, 2021 release shipped")
        assert "2021-09-05" in tags
        assert "september 2021" in tags


class TestISOStyle:
    """ISO 8601 yyyy-mm-dd style."""

    def test_iso_basic(self) -> None:
        tags = _tags("Deployment 2024-03-15 succeeded")
        assert "2024-03-15" in tags
        assert "march 2024" in tags
        assert "march" in tags
        assert "2024" in tags

    def test_iso_invalid_month_ignored(self) -> None:
        # 2024-13-01 is not a valid date; pattern matches but range guard rejects.
        tags = _tags("invoice 2024-13-01 pending")
        assert "2024-13-01" not in tags


class TestMonthYearOnly:
    """Month-year style without an explicit day."""

    def test_my_full_month(self) -> None:
        tags = _tags("hired in October 2022 as engineer")
        assert "october 2022" in tags
        assert "october" in tags
        assert "2022" in tags

    def test_my_does_not_emit_iso_when_no_day(self) -> None:
        tags = _tags("hired in October 2022")
        # No day means we cannot synthesise an ISO yyyy-mm-dd tag.
        assert not any(t.startswith("2022-") for t in tags)


class TestEdgeCases:
    """Defensive paths in the enrichment helper."""

    def test_empty_content_no_tags_added(self) -> None:
        fact = Fact(content="")
        enrich_date_tags(fact)
        assert fact.tags == []

    def test_text_without_dates_unchanged(self) -> None:
        fact = Fact(content="just some text without anything date-like")
        enrich_date_tags(fact)
        assert fact.tags == []

    def test_bare_year_in_room_number_does_not_match(self) -> None:
        # `_DATE_DMY` etc require multi-component matches — a bare "2023" in
        # "Room 2023" must not produce date tags (false-positive guard).
        tags = _tags("meet at Room 2023 tomorrow")
        assert "2023" not in tags

    def test_witness_surface_is_also_scanned(self) -> None:
        tags = _tags("plain content", witness_surface="event on July 4, 2023")
        assert "2023-07-04" in tags

    def test_existing_tags_preserved(self) -> None:
        fact = Fact(content="meeting on March 1, 2024", tags=["work", "calendar"])
        enrich_date_tags(fact)
        assert "work" in fact.tags
        assert "calendar" in fact.tags
        assert "2024-03-01" in fact.tags

    def test_tag_cap_respects_max_tags(self) -> None:
        # Pre-fill close to the _MAX_TAGS cap (10); enrichment must not overflow.
        many_tags = [f"pre{i}" for i in range(9)]
        fact = Fact(content="meeting on March 1, 2024", tags=list(many_tags))
        enrich_date_tags(fact)
        assert len(fact.tags) <= 10

    def test_returns_same_fact_object(self) -> None:
        fact = Fact(content="event on April 5, 2025")
        result = enrich_date_tags(fact)
        assert result is fact


@pytest.mark.parametrize(
    "text,expected_iso",
    [
        ("[1 January, 2020] kickoff", "2020-01-01"),
        ("December 31, 1999 Y2K eve", "1999-12-31"),
        ("logged 2025-07-22 outage", "2025-07-22"),
    ],
)
def test_canonical_iso_for_each_style(text: str, expected_iso: str) -> None:
    assert expected_iso in _tags(text)
