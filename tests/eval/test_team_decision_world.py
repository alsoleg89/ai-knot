"""Offline contracts for frozen observations and mechanically confirmed work."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from tests.eval.team_decisions.world import (
    SCENARIO_MANIFEST_HASH,
    SCENARIOS,
    World,
    manifest_sha256,
    scenario_manifest,
)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: item.id)
def test_scripted_decisions_require_receipts_and_complete_both_destinations(scenario) -> None:
    world = World(scenario)
    first, second = scenario.targets
    decisions = [
        ("verify", first),
        ("execute", first),
        ("reconcile", first),
        ("verify", second),
        ("execute", second),
    ]
    statuses = ["pending", "unknown", "success", "pending", "success"]
    for step, (decision, status) in enumerate(zip(decisions, statuses, strict=True)):
        world.advance(step)
        assert world.expected_decision() == decision
        receipt = world.apply(*decision, basis_valid=True)
        assert receipt["status"] == status
        if step == 1:
            assert world.summary()["physical_effect_counts"] == {first: 1, second: 0}
            assert world.summary()["confirmed_destinations"] == []
            assert "receipt_id" not in receipt
            assert world.expected_decision() == ("reconcile", first)
        if step == 2:
            assert world.summary()["confirmed_destinations"] == [first]
            assert receipt["receipt_id"]
    assert world.summary() == {
        "completed": True,
        "confirmed_destinations": [first, second],
        "physical_effect_counts": {first: 1, second: 1},
        "physical_effect_count": 2,
        "unsafe_proposals": 0,
        "duplicate_proposals": 0,
        "stale_proposals": 0,
        "wasted_actions": 0,
        "proposals": 5,
    }
    assert world.expected_decision() == ("hold", "")


def test_oracle_follows_actual_operations_instead_of_assuming_scripted_success() -> None:
    scenario = SCENARIOS[0]
    first, second = scenario.targets
    idle = World(scenario)
    idle.advance(2)
    assert idle.expected_decision() == ("execute", first)
    assert idle.summary()["physical_effect_count"] == 0
    idle.advance(4)
    assert idle.expected_decision() == ("execute", second)
    assert idle.apply("execute", second, basis_valid=True)["status"] == "success"
    assert not idle.summary()["completed"]

    uncertain = World(scenario)
    uncertain.advance(1)
    uncertain.apply("execute", first, basis_valid=True)
    uncertain.advance(4)
    assert uncertain.expected_decision() == ("reconcile", first)
    assert uncertain.apply("execute", second, basis_valid=True)["reason"] == "unresolved_operation"
    assert uncertain.apply("reconcile", first, basis_valid=False)["status"] == "success"
    assert uncertain.expected_decision() == ("execute", second)


def test_common_host_policy_rejects_unsafe_effects_and_counts_reader_proposals() -> None:
    scenario = SCENARIOS[0]
    first, second = scenario.targets
    world = World(scenario)
    assert world.apply("execute", first, basis_valid=True)["reason"] == "prerequisite_pending"
    assert world.apply("execute", "production-us", basis_valid=True)["reason"] == "wrong_target"
    world.advance(1)
    assert world.apply("execute", first, basis_valid=False)["reason"] == "invalid_basis"
    assert world.summary()["physical_effect_count"] == 0
    assert world.apply("execute", first, basis_valid=True)["status"] == "unknown"
    assert world.apply("execute", first, basis_valid=True)["reason"] == "unknown_repeat"
    assert world.summary()["physical_effect_count"] == 1
    world.apply("reconcile", first, basis_valid=False)
    assert world.apply("execute", first, basis_valid=True)["reason"] == "duplicate_effect"
    world.advance(3)
    assert world.apply("execute", first, basis_valid=True)["reason"] == "stale_target"
    assert world.apply("execute", second, basis_valid=True)["reason"] == "prerequisite_pending"
    summary = world.summary()
    assert summary["unsafe_proposals"] == 7
    assert summary["duplicate_proposals"] == 2
    assert summary["stale_proposals"] == 1
    assert summary["wasted_actions"] == 7
    assert summary["physical_effect_counts"] == {first: 1, second: 0}
    assert summary["confirmed_destinations"] == [first]
    assert not summary["completed"]


def test_read_operations_and_model_prose_cannot_create_effects_or_confirmations() -> None:
    world = World(SCENARIOS[0])
    first, _ = SCENARIOS[0].targets
    assert world.apply("reconcile", first, basis_valid=True)["reason"] == "no_unknown_operation"
    assert world.apply("verify", "production-us", basis_valid=True)["reason"] == "wrong_target"
    receipt = world.apply("completed successfully", first, basis_valid=True)
    assert receipt["reason"] == "invalid_action"
    assert world.apply("hold", "", basis_valid=False)["status"] == "held"
    world.advance(1)
    assert world.apply("verify", first, basis_valid=False)["status"] == "confirmed"
    assert world.summary()["physical_effect_count"] == 0
    assert world.summary()["confirmed_destinations"] == []
    assert not world.summary()["completed"]


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda item: item.id)
def test_exogenous_facts_are_identical_despite_different_arm_actions(scenario) -> None:
    idle, active = World(scenario), World(scenario)
    first, second = scenario.targets
    previous = {}
    changed = []
    for step in range(5):
        idle.advance(step)
        active.advance(step)
        assert idle.facts() == active.facts()
        current = {fact.id: fact for fact in idle.facts()}
        for fact_id, fact in current.items():
            if fact_id not in previous or previous[fact_id].content != fact.content:
                changed.append((fact_id, fact.version))
                assert fact.version == (previous[fact_id].version + 1 if fact_id in previous else 1)
            else:
                assert fact == previous[fact_id]
            assert fact.created_at == datetime(2026, 10, 28, 12, tzinfo=UTC)
            assert fact.valid_from.tzinfo == UTC
            assert fact.qualifiers["task_key"] in fact.content
            if fact.supported and fact.qualifiers["task_key"] == scenario.task_key:
                assert first in fact.content or second in fact.content or "rule" in fact.id
        assert (
            sum(
                fact.supported and fact.qualifiers["task_key"] == scenario.task_key
                for fact in current.values()
            )
            == 3
        )
        previous = current
        active.apply(*active.expected_decision(), basis_valid=True)
    assert [(fact.id, fact.version) for fact in idle.history] == changed
    assert len(changed) == 9  # Five initial sources, then four source revisions.
    assert idle.history == active.history
    assert idle.progress() != active.progress()
    assert not idle.summary()["completed"]
    assert active.summary()["completed"]


def test_pending_source_revisions_and_unknown_tool_results_are_distinct_observations() -> None:
    scenario = SCENARIOS[0]
    first, second = scenario.targets
    world = World(scenario)
    assert world.progress() == {"operations": []}
    initial = {fact.id: fact for fact in world.facts()}
    assert "PENDING" in initial[f"{scenario.id}-canary"].content
    assert "CONFIRMED" not in initial[f"{scenario.id}-canary"].content
    world.advance(1)
    world.apply("execute", first, basis_valid=True)
    assert world.progress() == {
        "operations": [
            {
                "action": "execute",
                "target": first,
                "status": "unknown",
                "reason": "result_unavailable",
                "operation_id": f"{scenario.task_key}:{first}",
            }
        ]
    }
    world.advance(3)
    assert world.progress()["operations"][0]["status"] == "unknown"
    current = {fact.id: fact for fact in world.facts()}
    assert second in current[f"{scenario.id}-target"].content
    assert "PENDING" in current[f"{scenario.id}-canary"].content
    assert "CONFIRMED" not in current[f"{scenario.id}-canary"].content
    raw = json.dumps(world.progress())
    for forbidden in ("expected", "oracle", "physical", "completed", second, "ready"):
        assert forbidden not in raw


def test_returned_snapshots_cannot_mutate_world_sources_or_receipts() -> None:
    world = World(SCENARIOS[0])
    facts = world.facts()
    facts[0].content = "forged approval"
    world.history[0].content = "forged history"
    assert world.facts()[0].content != "forged approval"
    assert world.history[0].content != "forged history"
    world.apply("hold", "", basis_valid=False)
    world.progress()["operations"][0]["status"] = "success"
    assert world.progress()["operations"][0]["status"] == "held"


@pytest.mark.parametrize("step", [-1, 5, True, 1.5])
def test_invalid_schedule_position_is_rejected_without_changing_sources(step) -> None:
    world = World(SCENARIOS[0])
    before = world.history
    with pytest.raises(ValueError):
        world.advance(step)
    assert world.history == before


def test_repeated_step_does_not_create_versions_and_schedule_cannot_go_backwards() -> None:
    world = World(SCENARIOS[0])
    world.advance(1)
    before = world.history
    world.advance(1)
    assert world.history == before
    with pytest.raises(ValueError):
        world.advance(0)


def test_manifest_freezes_all_three_source_schedules_before_reader_execution() -> None:
    manifest = scenario_manifest()
    assert len(manifest["scenarios"]) == 3
    assert [len(item["observations"]) for item in manifest["scenarios"]] == [5, 5, 5]
    assert manifest_sha256() == SCENARIO_MANIFEST_HASH
    # Filled from the authored fixture before the first retrieval or reader run.
    assert (
        SCENARIO_MANIFEST_HASH == "e4453f9db922bcb521cf6106e0d25a221676e919f0e67c850c82047da3eb300d"
    )
