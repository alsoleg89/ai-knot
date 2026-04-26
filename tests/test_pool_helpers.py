"""Unit tests for ai_knot._pool_helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_knot._pool_helpers import (
    _extract_claim_key,
    _pool_rerank,
    _resolve_claim_conflicts,
)
from ai_knot.types import Fact, MESIState

# ---- _extract_claim_key -----------------------------------------------------


class TestExtractClaimKey:
    def test_returns_empty_for_short_input(self) -> None:
        assert _extract_claim_key("two words") == ""

    def test_returns_empty_when_no_attribute_keyword(self) -> None:
        assert _extract_claim_key("one two three four") == ""

    def test_extracts_entity_and_attribute(self) -> None:
        # "alpha" + "beta" precede "sla" (attribute stem) → key = "alpha_beta::sla"
        assert _extract_claim_key("alpha beta sla rule") == "alpha_beta::sla"

    def test_caps_entities_at_two_tokens(self) -> None:
        # Only the first two non-attr tokens are kept as entity tokens;
        # "gamma" is dropped because the entity-token cap is 2.
        key = _extract_claim_key("alpha beta gamma rate per second")
        assert key.endswith("::rate")
        # Exactly two entity tokens joined with underscore.
        assert key.count("_") == 1

    def test_attribute_stem_match(self) -> None:
        key = _extract_claim_key("service availability uptime measure")
        assert key.endswith("::uptim")

    def test_returns_empty_when_attribute_is_first_token(self) -> None:
        # No entity token collected before the attribute.
        assert _extract_claim_key("price something else") == ""


# ---- _pool_rerank -----------------------------------------------------------


def _fact(
    fact_id: str,
    *,
    created_at: datetime,
    mesi: MESIState = MESIState.EXCLUSIVE,
    slot_key: str = "",
) -> Fact:
    return Fact(
        content=f"content for {fact_id}",
        id=fact_id,
        created_at=created_at,
        mesi_state=mesi,
        slot_key=slot_key,
    )


class TestPoolRerank:
    def test_passthrough_for_short_lists(self) -> None:
        f = _fact("a", created_at=datetime(2024, 1, 1, tzinfo=UTC))
        assert _pool_rerank([(f, 1.0)]) == [(f, 1.0)]
        assert _pool_rerank([]) == []

    def test_recency_boost_orders_newer_higher(self) -> None:
        old = _fact("old", created_at=datetime(2024, 1, 1, tzinfo=UTC))
        new = _fact("new", created_at=datetime(2024, 6, 1, tzinfo=UTC))
        ranked = _pool_rerank([(old, 1.0), (new, 1.0)], recency_weight=0.10)
        scores = {f.id: s for f, s in ranked}
        assert scores["new"] > scores["old"]

    def test_freshness_boost_for_modified_or_shared(self) -> None:
        cold = _fact("cold", created_at=datetime(2024, 1, 1, tzinfo=UTC))
        modified = _fact(
            "mod", created_at=datetime(2024, 1, 1, tzinfo=UTC), mesi=MESIState.MODIFIED
        )
        shared = _fact("shr", created_at=datetime(2024, 1, 1, tzinfo=UTC), mesi=MESIState.SHARED)
        ranked = dict(
            (f.id, s)
            for f, s in _pool_rerank(
                [(cold, 1.0), (modified, 1.0), (shared, 1.0)],
                freshness_weight=0.20,
                recency_weight=0.0,
            )
        )
        assert ranked["mod"] > ranked["cold"]
        assert ranked["shr"] > ranked["cold"]

    def test_slot_winner_boost_only_for_modified_with_slot_key(self) -> None:
        plain = _fact("p", created_at=datetime(2024, 1, 1, tzinfo=UTC))
        slot_winner = _fact(
            "sw",
            created_at=datetime(2024, 1, 1, tzinfo=UTC),
            mesi=MESIState.MODIFIED,
            slot_key="user::pref",
        )
        ranked = dict(
            (f.id, s)
            for f, s in _pool_rerank(
                [(plain, 1.0), (slot_winner, 1.0)],
                slot_winner_weight=0.50,
                recency_weight=0.0,
                freshness_weight=0.0,
            )
        )
        assert ranked["sw"] > ranked["p"] + 0.4


# ---- _resolve_claim_conflicts -----------------------------------------------


def _claim_fact(
    fact_id: str,
    *,
    claim_key: str = "",
    slot_key: str = "",
    origin: str = "",
    created: datetime | None = None,
) -> Fact:
    return Fact(
        content=f"f-{fact_id}",
        id=fact_id,
        claim_key=claim_key,
        slot_key=slot_key,
        origin_agent_id=origin,
        created_at=created or datetime(2024, 1, 1, tzinfo=UTC),
    )


class TestResolveClaimConflicts:
    def test_empty_input(self) -> None:
        assert _resolve_claim_conflicts([], lambda _: 1.0) == []

    def test_unclustered_facts_pass_through(self) -> None:
        f = _claim_fact("a")
        result = _resolve_claim_conflicts([(f, 0.5)], lambda _: 1.0)
        assert result == [(f, 0.5)]

    def test_single_member_cluster_kept(self) -> None:
        f = _claim_fact("a", claim_key="svc::sla")
        result = _resolve_claim_conflicts([(f, 0.7)], lambda _: 1.0)
        assert result == [(f, 0.7)]

    def test_slotted_member_wins_over_unslotted(self) -> None:
        slotted = _claim_fact("s", claim_key="svc::sla", slot_key="svc")
        unslotted = _claim_fact("u", claim_key="svc::sla", created=datetime(2025, 6, 1, tzinfo=UTC))
        result = _resolve_claim_conflicts(
            [(unslotted, 0.9), (slotted, 0.4)],
            lambda _: 1.0,
        )
        assert len(result) == 1
        assert result[0][0].id == "s"

    def test_unslotted_winner_uses_trust_times_recency(self) -> None:
        old_high_trust = _claim_fact(
            "old", claim_key="x::y", origin="alice", created=datetime(2024, 1, 1, tzinfo=UTC)
        )
        new_low_trust = _claim_fact(
            "new", claim_key="x::y", origin="bob", created=datetime(2025, 1, 1, tzinfo=UTC)
        )

        # alice has 100x more trust than bob → wins despite being older.
        result = _resolve_claim_conflicts(
            [(old_high_trust, 0.5), (new_low_trust, 0.5)],
            lambda agent: 100.0 if agent == "alice" else 1.0,
        )
        assert len(result) == 1
        assert result[0][0].id == "old"

    @pytest.mark.parametrize("claim_key", ["", "  "])
    def test_empty_claim_key_pass_through(self, claim_key: str) -> None:
        f = _claim_fact("a", claim_key=claim_key)
        result = _resolve_claim_conflicts([(f, 0.3)], lambda _: 1.0)
        # Non-truthy claim_key is treated as unclustered.
        assert result == [(f, 0.3)]
