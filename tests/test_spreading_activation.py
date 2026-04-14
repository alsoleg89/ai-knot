"""Unit tests for _spreading_activation.spreading_activation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ai_knot._inverted_index import InvertedIndex
from ai_knot._spreading_activation import spreading_activation
from ai_knot.types import Fact


def _make_fact(
    id: str,
    content: str,
    *,
    tags: list[str] | None = None,
    entity: str = "",
    value_text: str = "",
    slot_key: str = "",
    created_at: datetime | None = None,
) -> Fact:
    return Fact(
        id=id,
        content=content,
        tags=tags or [],
        entity=entity,
        value_text=value_text,
        slot_key=slot_key,
        created_at=created_at or datetime.now(UTC),
    )


class TestSpreadingActivationBasics:
    def test_empty_seeds_returns_empty(self) -> None:
        idx = InvertedIndex([_make_fact("1", "foo")])
        assert spreading_activation([], idx, topk=5) == []

    def test_seeds_preserved_in_output(self) -> None:
        f1 = _make_fact("1", "seed fact", tags=["solo"])
        idx = InvertedIndex([f1])
        result = spreading_activation([(f1, 1.0)], idx, topk=5)
        assert result[0][0].id == "1"

    def test_no_activation_when_no_shared_tags(self) -> None:
        f1 = _make_fact("1", "seed", tags=["alpha"])
        f2 = _make_fact("2", "other", tags=["beta"])
        idx = InvertedIndex([f1, f2])
        # f2 has no shared tag → not activated (temporal may fire; test with large window disabled)
        result_no_temporal = spreading_activation([(f1, 1.0)], idx, topk=5, temporal_window_sec=0)
        ids = {f.id for f, _ in result_no_temporal}
        assert "2" not in ids


class TestTagHop:
    def test_tag_hop_activates_neighbour(self) -> None:
        f1 = _make_fact("1", "Melanie plays violin", tags=["hobby", "music"])
        f2 = _make_fact("2", "Melanie does pottery", tags=["hobby", "art"])
        idx = InvertedIndex([f1, f2])
        result = spreading_activation([(f1, 1.0)], idx, topk=5, temporal_window_sec=0)
        ids = {f.id for f, _ in result}
        assert "1" in ids
        assert "2" in ids

    def test_activated_scores_below_seed(self) -> None:
        f1 = _make_fact("1", "seed", tags=["hobby"])
        f2 = _make_fact("2", "neighbour", tags=["hobby"])
        idx = InvertedIndex([f1, f2])
        result = spreading_activation([(f1, 1.0)], idx, topk=5, temporal_window_sec=0)
        score_map = {f.id: s for f, s in result}
        assert score_map["2"] < score_map["1"]

    def test_tag_hop_ignores_seed_itself(self) -> None:
        f1 = _make_fact("1", "seed", tags=["hobby"])
        idx = InvertedIndex([f1])
        result = spreading_activation([(f1, 1.0)], idx, topk=5)
        # Only f1 itself; no duplicates.
        assert len(result) == 1

    def test_multiple_seeds_same_tag(self) -> None:
        f1 = _make_fact("1", "violin", tags=["hobby"])
        f2 = _make_fact("2", "pottery", tags=["hobby"])
        f3 = _make_fact("3", "camping", tags=["hobby"])
        idx = InvertedIndex([f1, f2, f3])
        result = spreading_activation([(f1, 1.0), (f2, 0.9)], idx, topk=5, temporal_window_sec=0)
        ids = {f.id for f, _ in result}
        assert "3" in ids


class TestEntityHop:
    def test_entity_hop_via_value_text(self) -> None:
        f1 = _make_fact(
            "1",
            "X is friends with Melanie",
            entity="user",
            slot_key="user::friend",
            value_text="Melanie",
        )
        f2 = _make_fact("2", "Melanie went camping last week")
        idx = InvertedIndex([f1, f2])
        result = spreading_activation([(f1, 1.0)], idx, topk=5, temporal_window_sec=0)
        assert "2" in {f.id for f, _ in result}

    def test_entity_hop_via_entity_field(self) -> None:
        f1 = _make_fact("1", "salary record", entity="alex chen", slot_key="alex chen::salary")
        f2 = _make_fact("2", "Alex Chen joined the company in 2020")
        idx = InvertedIndex([f1, f2])
        result = spreading_activation([(f1, 1.0)], idx, topk=5, temporal_window_sec=0)
        assert "2" in {f.id for f, _ in result}

    def test_entity_hop_no_op_when_entity_empty(self) -> None:
        f1 = _make_fact("1", "dated raw turn")  # no entity, no value_text
        f2 = _make_fact("2", "another raw turn")
        idx = InvertedIndex([f1, f2])
        # Without temporal window, should only activate via tags (none here) → f2 not included
        result = spreading_activation([(f1, 1.0)], idx, topk=5, temporal_window_sec=0)
        assert "2" not in {f.id for f, _ in result}


class TestSlotCluster:
    def test_slot_cluster_requires_two_hot_seeds(self) -> None:
        """Single seed with slot_key → no hot slot → third fact not activated via slot."""
        f1 = _make_fact("1", "violin", slot_key="melanie::hobby", tags=["music"])
        f2 = _make_fact("2", "unrelated noise", slot_key="user::role", tags=[])
        f3 = _make_fact("3", "pottery", slot_key="melanie::hobby", tags=["art"])
        idx = InvertedIndex([f1, f2, f3])
        # Only f1 is seed → slot_counts = {"melanie::hobby": 1} → no hot slot.
        result = spreading_activation([(f1, 1.0)], idx, topk=5, temporal_window_sec=0)
        ids = {f.id for f, _ in result}
        # f3 has different tags ("art" vs "music") and no entity hop → should not appear.
        assert "3" not in ids

    def test_slot_cluster_activates_on_two_matching_seeds(self) -> None:
        f1 = _make_fact("1", "violin", slot_key="melanie::hobby")
        f2 = _make_fact("2", "dance", slot_key="melanie::hobby")
        f3 = _make_fact("3", "pottery", slot_key="melanie::hobby")
        idx = InvertedIndex([f1, f2, f3])
        result = spreading_activation([(f1, 1.0), (f2, 0.9)], idx, topk=5, temporal_window_sec=0)
        assert "3" in {f.id for f, _ in result}


class TestTemporalWindow:
    def test_temporal_window_activates_nearby(self) -> None:
        t0 = datetime.now(UTC)
        f1 = _make_fact("1", "event A", created_at=t0)
        f2 = _make_fact("2", "event B", created_at=t0 + timedelta(seconds=100))
        f3 = _make_fact("3", "event C", created_at=t0 + timedelta(seconds=3600))
        idx = InvertedIndex([f1, f2, f3])
        result = spreading_activation([(f1, 1.0)], idx, topk=5, temporal_window_sec=300)
        ids = {f.id for f, _ in result}
        assert "2" in ids
        assert "3" not in ids

    def test_temporal_window_disabled_at_zero(self) -> None:
        t0 = datetime.now(UTC)
        f1 = _make_fact("1", "event A", tags=[], created_at=t0)
        f2 = _make_fact("2", "event B", tags=[], created_at=t0 + timedelta(seconds=10))
        idx = InvertedIndex([f1, f2])
        result = spreading_activation([(f1, 1.0)], idx, topk=5, temporal_window_sec=0)
        ids = {f.id for f, _ in result}
        assert "2" not in ids

    def test_temporal_score_below_tag_score(self) -> None:
        t0 = datetime.now(UTC)
        f_seed = _make_fact("0", "seed", tags=["x"], created_at=t0)
        f_tag = _make_fact("1", "tag-match", tags=["x"], created_at=t0 + timedelta(hours=1))
        f_time = _make_fact("2", "time-match", tags=[], created_at=t0 + timedelta(seconds=60))
        idx = InvertedIndex([f_seed, f_tag, f_time])
        result = spreading_activation([(f_seed, 1.0)], idx, topk=5, temporal_window_sec=300)
        score_map = {f.id: s for f, s in result}
        # Tag-hop score = 1.0 * 0.6 = 0.6; temporal = 1.0 * 0.6 * 0.4 = 0.24
        assert score_map.get("1", 0.0) > score_map.get("2", 0.0)


class TestTopkAndBudget:
    def test_topk_cap(self) -> None:
        f_seed = _make_fact("0", "seed", tags=["a"])
        neighbours = [_make_fact(str(i), f"n{i}", tags=["a"]) for i in range(1, 20)]
        idx = InvertedIndex([f_seed, *neighbours])
        result = spreading_activation([(f_seed, 5.0)], idx, topk=3, temporal_window_sec=0)
        assert len(result) == 3

    def test_seed_comes_first_when_topk_tight(self) -> None:
        f_seed = _make_fact("0", "seed", tags=["a"])
        neighbours = [_make_fact(str(i), f"n{i}", tags=["a"]) for i in range(1, 10)]
        idx = InvertedIndex([f_seed, *neighbours])
        result = spreading_activation([(f_seed, 5.0)], idx, topk=2, temporal_window_sec=0)
        assert result[0][0].id == "0"

    def test_activation_budget_limits_extras(self) -> None:
        f_seed = _make_fact("0", "seed", tags=["a"])
        neighbours = [_make_fact(str(i), f"n{i}", tags=["a"]) for i in range(1, 20)]
        idx = InvertedIndex([f_seed, *neighbours])
        result = spreading_activation(
            [(f_seed, 5.0)], idx, topk=100, temporal_window_sec=0, activation_budget=3
        )
        # seed + at most 3 activated
        assert len(result) <= 4

    def test_output_sorted_by_score_descending(self) -> None:
        f1 = _make_fact("1", "high seed", tags=["t"])
        f2 = _make_fact("2", "low seed", tags=["t"])
        f3 = _make_fact("3", "activated", tags=["t"])
        idx = InvertedIndex([f1, f2, f3])
        result = spreading_activation([(f1, 1.0), (f2, 0.5)], idx, topk=5, temporal_window_sec=0)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)


class TestNoSideEffects:
    def test_does_not_mutate_seeds_list(self) -> None:
        f1 = _make_fact("1", "x", tags=["t"])
        f2 = _make_fact("2", "y", tags=["t"])
        idx = InvertedIndex([f1, f2])
        seeds = [(f1, 1.0)]
        spreading_activation(seeds, idx, topk=5)
        assert seeds == [(f1, 1.0)]

    def test_does_not_mutate_index(self) -> None:
        f1 = _make_fact("1", "x", tags=["t"])
        idx = InvertedIndex([f1])
        before = dict(idx.tags_postings.get("t", {}))
        spreading_activation([(f1, 1.0)], idx, topk=5)
        assert idx.tags_postings.get("t", {}) == before
