"""Unit tests for LoCoMo scenario helper functions."""

from __future__ import annotations

from tests.eval.benchmark.scenarios.s_locomo import (
    _best_f1_against,
    _evidence_recall_at_k,
    _iter_turns,
)


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
        turns, dia_map = _iter_turns(sample)
        assert turns == ["Alice: Hello", "Bob: Hi there"]
        assert dia_map == {"D1:1": "Alice: Hello", "D1:2": "Bob: Hi there"}

    def test_skips_date_time_keys(self) -> None:
        sample = self._make_sample(
            {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1_date_time": "2024-01-01T14:00:00",
                "session_1": [{"speaker": "Alice", "text": "No date", "dia_id": "D1:1"}],
            }
        )
        turns, dia_map = _iter_turns(sample)
        assert turns == ["Alice: No date"]
        assert dia_map == {"D1:1": "Alice: No date"}

    def test_skips_non_session_keys(self) -> None:
        sample = {
            "conversation": {
                "speaker_a": "Alice",
                "speaker_b": "Bob",
                "session_1": [{"speaker": "Alice", "text": "Real turn", "dia_id": "D1:1"}],
            },
            "qa": [{"question": "What?", "answer": "42"}],
        }
        turns, dia_map = _iter_turns(sample)
        assert turns == ["Alice: Real turn"]
        assert dia_map == {"D1:1": "Alice: Real turn"}

    def test_default_speaker(self) -> None:
        sample = self._make_sample(
            {
                "session_1": [{"text": "No speaker field", "dia_id": "D1:1"}],
            }
        )
        turns, dia_map = _iter_turns(sample)
        assert turns == ["speaker: No speaker field"]
        assert dia_map == {"D1:1": "speaker: No speaker field"}

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
        turns, dia_map = _iter_turns(sample)
        assert turns == ["A: one", "B: two", "A: three"]
        assert dia_map == {"D1:1": "A: one", "D2:1": "B: two", "D3:1": "A: three"}

    def test_empty_conversation(self) -> None:
        sample = self._make_sample({"speaker_a": "A", "speaker_b": "B"})
        turns, dia_map = _iter_turns(sample)
        assert turns == []
        assert dia_map == {}

    def test_fallback_flat_sample(self) -> None:
        """Graceful fallback: if sample has no 'conversation' key, iterate top-level."""
        flat = {
            "session_1": [{"speaker": "X", "text": "hi", "dia_id": "D1:1"}],
        }
        turns, dia_map = _iter_turns(flat)
        assert turns == ["X: hi"]
        assert dia_map == {"D1:1": "X: hi"}

    def test_turn_without_dia_id(self) -> None:
        """Turns missing dia_id should still appear in turns but not in dia_map."""
        sample = self._make_sample(
            {
                "session_1": [
                    {"speaker": "A", "text": "has id", "dia_id": "D1:1"},
                    {"speaker": "B", "text": "no id"},
                ],
            }
        )
        turns, dia_map = _iter_turns(sample)
        assert turns == ["A: has id", "B: no id"]
        assert dia_map == {"D1:1": "A: has id"}


class TestEvidenceRecallAtK:
    def test_all_evidence_found(self) -> None:
        dia_map = {"D1:1": "Alice: Hello there", "D1:2": "Bob: Hi friend"}
        retrieved = ["Alice: Hello there", "Bob: Hi friend", "noise"]
        assert _evidence_recall_at_k(retrieved, ["D1:1", "D1:2"], dia_map) == 1.0

    def test_partial_evidence(self) -> None:
        dia_map = {"D1:1": "Alice: Hello there", "D1:2": "Bob: Hi friend"}
        retrieved = ["Alice: Hello there", "unrelated stuff"]
        score = _evidence_recall_at_k(retrieved, ["D1:1", "D1:2"], dia_map)
        assert score == 0.5

    def test_no_evidence_found(self) -> None:
        dia_map = {"D1:1": "Alice: very specific unique topic xyz"}
        retrieved = ["completely unrelated text about something else"]
        score = _evidence_recall_at_k(retrieved, ["D1:1"], dia_map)
        assert score == 0.0

    def test_empty_evidence_ids(self) -> None:
        assert _evidence_recall_at_k(["text"], [], {}) == 0.0

    def test_missing_dia_id_skipped(self) -> None:
        """Evidence IDs not in dia_map are skipped; recall is over resolved only."""
        dia_map = {"D1:1": "Alice: Hello"}
        # D1:2 not in dia_map → only D1:1 counts, and it's found
        score = _evidence_recall_at_k(["Alice: Hello"], ["D1:1", "D1:2"], dia_map)
        assert score == 1.0

    def test_all_ids_missing(self) -> None:
        """If no evidence IDs resolve, return 0.0."""
        assert _evidence_recall_at_k(["text"], ["D99:1"], {}) == 0.0

    def test_empty_retrieved(self) -> None:
        dia_map = {"D1:1": "Alice: Hello"}
        assert _evidence_recall_at_k([], ["D1:1"], dia_map) == 0.0
