"""Tests for Phase 3 contract-driven event retrieval helpers."""

# Import helpers from query_runtime
from ai_knot.query_runtime import (
    _expand_window_n_turns,
    _extract_explicit_date_from_question,
)


def test_extract_explicit_date_iso():
    d = _extract_explicit_date_from_question("When did this happen on 2023-01-19?")
    assert d is not None
    assert d.year == 2023 and d.month == 1 and d.day == 19


def test_extract_explicit_date_month_name():
    d = _extract_explicit_date_from_question("What happened on January 19, 2023?")
    assert d is not None


def test_relative_date_returns_none():
    assert _extract_explicit_date_from_question("What happened yesterday?") is None
    assert _extract_explicit_date_from_question("What about last Friday?") is None
    assert _extract_explicit_date_from_question("What will happen next month?") is None


def test_expand_window_n1():
    # 3-turn window
    from dataclasses import dataclass

    @dataclass
    class Ep:
        id: str
        prev_id: str | None = None
        next_id: str | None = None

    ep = Ep("c", prev_id="p", next_id="n")
    result = _expand_window_n_turns([ep], storage=None, agent_id="a", n=1)
    assert result == ["p", "c", "n"]


def test_expand_window_n2():
    # 5-turn window
    from dataclasses import dataclass

    @dataclass
    class Ep:
        id: str
        prev_id: str | None = None
        next_id: str | None = None

    ep_pp = Ep("pp")
    ep_p = Ep("p", prev_id="pp", next_id="c")
    ep_c = Ep("c", prev_id="p", next_id="n")
    ep_n = Ep("n", prev_id="c", next_id="nn")
    ep_nn = Ep("nn", prev_id="n")
    eps_map = {"pp": ep_pp, "p": ep_p, "c": ep_c, "n": ep_n, "nn": ep_nn}

    class FakeStorage:
        def get_episode(self, agent_id: str, eid: str) -> object:
            return eps_map.get(eid)

    result = _expand_window_n_turns([ep_c], storage=FakeStorage(), agent_id="a", n=2)
    assert "pp" in result and "p" in result and "c" in result and "n" in result and "nn" in result
    assert len(result) == 5
