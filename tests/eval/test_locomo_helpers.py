"""Unit tests for LoCoMo scenario helper functions."""

from __future__ import annotations

from tests.eval.benchmark.scenarios.s_locomo import _best_f1_against, _iter_turns


class TestBestF1Against:
    def test_exact_match(self) -> None:
        assert _best_f1_against(["the answer is 42"], "the answer is 42") == 1.0

    def test_partial_match(self) -> None:
        score = _best_f1_against(["the answer is 42"], "the answer is unknown")
        assert 0.0 < score < 1.0

    def test_no_match(self) -> None:
        assert _best_f1_against(["hello world"], "xyz abc") == 0.0

    def test_empty_retrieved(self) -> None:
        assert _best_f1_against([], "some answer") == 0.0

    def test_empty_gold(self) -> None:
        assert _best_f1_against(["some text"], "") == 0.0

    def test_best_of_multiple(self) -> None:
        # Second text is a better match
        score = _best_f1_against(
            ["irrelevant noise", "the answer is 42"],
            "the answer is 42",
        )
        assert score == 1.0


class TestIterTurns:
    def test_basic_turns(self) -> None:
        conv = {
            "session_1": [
                {"speaker": "Alice", "text": "Hello"},
                {"speaker": "Bob", "text": "Hi there"},
            ]
        }
        turns = _iter_turns(conv)
        assert turns == ["Alice: Hello", "Bob: Hi there"]

    def test_skips_date_keys(self) -> None:
        conv = {
            "session_1_date": "2024-01-01",
            "session_1": [{"speaker": "Alice", "text": "No date"}],
        }
        turns = _iter_turns(conv)
        assert turns == ["Alice: No date"]

    def test_skips_non_session_keys(self) -> None:
        conv = {
            "qa": [{"question": "What?", "answer": "42"}],
            "session_1": [{"speaker": "Alice", "text": "Real turn"}],
        }
        turns = _iter_turns(conv)
        assert turns == ["Alice: Real turn"]

    def test_default_speaker(self) -> None:
        conv = {
            "session_1": [{"text": "No speaker field"}],
        }
        turns = _iter_turns(conv)
        assert turns == ["speaker: No speaker field"]

    def test_multiple_sessions(self) -> None:
        conv = {
            "session_1": [{"speaker": "A", "text": "one"}],
            "session_2": [{"speaker": "B", "text": "two"}],
        }
        turns = _iter_turns(conv)
        assert len(turns) == 2
