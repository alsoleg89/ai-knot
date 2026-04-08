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
    """Tests use the real LoCoMo10 JSON schema.

    Each sample has:
      - ``conversation``: dict with ``speaker_a``, ``speaker_b``,
        ``session_N`` (list of turns), ``session_N_date_time`` (str).
      - ``qa``: list of QA dicts.
    """

    @staticmethod
    def _make_sample(conversation: dict[str, object]) -> dict[str, object]:
        """Wrap a conversation dict in a full LoCoMo sample structure."""
        return {"conversation": conversation, "qa": []}

    def test_basic_turns_from_conversation(self) -> None:
        sample = self._make_sample(
            {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "2:00 pm on 1 Jan, 2024",
                "session_1": [
                    {"speaker": "Alice", "text": "Hello", "dia_id": "D1:1"},
                    {"speaker": "Bob", "text": "Hi there", "dia_id": "D1:2"},
                ],
            }
        )
        turns = _iter_turns(sample)
        assert turns == ["Alice: Hello", "Bob: Hi there"]

    def test_skips_date_time_keys(self) -> None:
        sample = self._make_sample(
            {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "2024-01-01T14:00:00",
                "session_1": [{"speaker": "Alice", "text": "No date", "dia_id": "D1:1"}],
            }
        )
        turns = _iter_turns(sample)
        assert turns == ["Alice: No date"]

    def test_skips_non_session_keys(self) -> None:
        sample = {
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1": [{"speaker": "Alice", "text": "Real turn", "dia_id": "D1:1"}],
            },
            "qa": [{"question": "What?", "answer": "42"}],
        }
        turns = _iter_turns(sample)
        assert turns == ["Alice: Real turn"]

    def test_default_speaker(self) -> None:
        sample = self._make_sample(
            {
                "session_1": [{"text": "No speaker field", "dia_id": "D1:1"}],
            }
        )
        turns = _iter_turns(sample)
        assert turns == ["speaker: No speaker field"]

    def test_multiple_sessions_sorted(self) -> None:
        """Sessions must be sorted by number, not by dict iteration order."""
        sample = self._make_sample(
            {
                "speaker_a": "A",
                "speaker_b": "B",
                "session_3": [{"speaker": "A", "text": "three", "dia_id": "D3:1"}],
                "session_3_date_time": "3pm",
                "session_1": [{"speaker": "A", "text": "one", "dia_id": "D1:1"}],
                "session_1_date_time": "1pm",
                "session_2": [{"speaker": "B", "text": "two", "dia_id": "D2:1"}],
                "session_2_date_time": "2pm",
            }
        )
        turns = _iter_turns(sample)
        assert turns == ["A: one", "B: two", "A: three"]

    def test_empty_conversation(self) -> None:
        sample = self._make_sample({"speaker_a": "A", "speaker_b": "B"})
        assert _iter_turns(sample) == []

    def test_fallback_flat_sample(self) -> None:
        """Graceful fallback: if sample has no 'conversation' key, iterate top-level."""
        flat = {
            "session_1": [{"speaker": "X", "text": "hi", "dia_id": "D1:1"}],
        }
        turns = _iter_turns(flat)
        assert turns == ["X: hi"]
