"""Regression tests for materialization v6 generic English frames.

Tests 3 new frame patterns added in v6:
1. 'got/received' → acquired (EVENT via _FP_EVENT_PATTERNS)
2. 'I've been X-ing' → activity_ongoing (STATE)
3. 'I became X' → became (TRANSITION)
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_knot.materialization import MATERIALIZATION_VERSION, materialize_episode
from ai_knot.query_types import ClaimKind, RawEpisode
from ai_knot.relation_vocab import RELATION_VERBS


def _episode(text: str, speaker: str = "Alice") -> RawEpisode:
    # Materializer extracts speaker from "Name: text" prefix in raw_text.
    now = datetime(2024, 6, 1, tzinfo=UTC)
    return RawEpisode(
        id="ep1",
        agent_id="a",
        session_id="s",
        turn_id="t",
        speaker=speaker,
        observed_at=now,
        raw_text=f"{speaker}: {text}",
        session_date=now,
        source_meta={},
        parent_episode_id=None,
    )


def test_version_is_6() -> None:
    assert MATERIALIZATION_VERSION == 6


@pytest.mark.parametrize(
    "text,expected_value_substr",
    [
        ("I got a promotion", "promotion"),
        ("I received an award for best design", "award"),
        ("I was given a scholarship", "scholarship"),
    ],
)
def test_acquired_frame(text: str, expected_value_substr: str) -> None:
    claims = materialize_episode(_episode(text))
    acquired = [c for c in claims if c.relation == "acquired"]
    assert acquired, (
        f"No 'acquired' claim from '{text}'; got {[(c.relation, c.value_text) for c in claims]}"
    )
    assert expected_value_substr.lower() in acquired[0].value_text.lower()
    assert acquired[0].kind == ClaimKind.EVENT


@pytest.mark.parametrize(
    "text",
    [
        "I've been running lately",
        "I have been cooking a lot",
        "I've been studying for exams",
    ],
)
def test_activity_ongoing_frame(text: str) -> None:
    claims = materialize_episode(_episode(text))
    activity = [c for c in claims if c.relation == "activity_ongoing"]
    assert activity, (
        f"No 'activity_ongoing' from '{text}'; got {[(c.relation, c.value_text) for c in claims]}"
    )
    assert activity[0].kind == ClaimKind.STATE


@pytest.mark.parametrize(
    "text,expected_value_substr",
    [
        ("I became a parent", "parent"),
        ("I became an engineer last year", "engineer"),
        ("I turned into a morning person", "morning person"),
    ],
)
def test_became_frame(text: str, expected_value_substr: str) -> None:
    claims = materialize_episode(_episode(text))
    became = [c for c in claims if c.relation == "became"]
    assert became, (
        f"No 'became' claim from '{text}'; got {[(c.relation, c.value_text) for c in claims]}"
    )
    assert expected_value_substr.lower() in became[0].value_text.lower()
    assert became[0].kind == ClaimKind.TRANSITION


# ---------------------------------------------------------------------------
# Relation reachability (BENCH_POLICY rule 3)
# Every relation emitted by a _FP_*_PATTERNS entry must exist in RELATION_VERBS.
# ---------------------------------------------------------------------------


def test_all_fp_event_relations_reachable() -> None:
    """All v6 relations must be in RELATION_VERBS for retrieval to work."""
    from ai_knot.materialization import _FP_EVENT_PATTERNS  # type: ignore[attr-defined]

    missing = [rel for rel, _ in _FP_EVENT_PATTERNS if rel not in RELATION_VERBS]
    assert not missing, (
        f"FP event relations not in RELATION_VERBS (unreachable via retrieval): {missing}"
    )


# ---------------------------------------------------------------------------
# No closed-noun-list check for visited / ran (BENCH_POLICY rule 1)
# Patterns must match generic phrases, not only a hardcoded allowlist.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "I visited my old neighbourhood",
        "I visited a friend in another city",
        "I stopped by the office to drop something off",
    ],
)
def test_visited_generic(text: str) -> None:
    claims = materialize_episode(_episode(text))
    visited = [c for c in claims if c.relation == "visited"]
    assert visited, (
        f"No 'visited' claim from '{text}'; got {[(c.relation, c.value_text) for c in claims]}"
    )


@pytest.mark.parametrize(
    "text",
    [
        "I ran the local event last weekend",
        "I completed a race this morning with a friend",
        "I ran a course through the mountains",
    ],
)
def test_ran_generic(text: str) -> None:
    claims = materialize_episode(_episode(text))
    ran = [c for c in claims if c.relation == "ran"]
    assert ran, f"No 'ran' claim from '{text}'; got {[(c.relation, c.value_text) for c in claims]}"
