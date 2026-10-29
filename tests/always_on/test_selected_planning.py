"""Partial semantic context needs a guard over the entire admissible task set."""

import asyncio
import importlib

import pytest
from tests.always_on.test_planning_turn import available, invoke, proposal

from ai_knot import KnowledgeBase
from ai_knot.always_on import AlwaysOnMemory, SemanticBudget, SemanticContext
from ai_knot.types import Fact


@pytest.fixture
def recipe():
    return importlib.import_module("examples.always_on_team")


def growing(recipe, storage, clock, path):
    kb, memory, source, intent, view = available(recipe, storage, clock, path)
    facts = [source, *[Fact(f"Inventory note {i}", id=f"noise-{i}") for i in range(8)]]
    kb.replace_facts(facts)
    return kb, memory, facts, intent, view


def vectors(texts):
    return [
        [1.0, 0.0] if text == "backup receipt" or "recovery copy" in text else [0.1, 1.0]
        for text in texts
    ]


@pytest.mark.anyio
async def test_selected_plan_retains_guard_and_suppresses_reopened_paid_acquisition(
    recipe, sql_storage, clock, tmp_path
):
    _, memory, _, intent, view = growing(recipe, sql_storage, clock, tmp_path)
    calls = []

    async def embed(texts):
        calls.append("embed")
        return vectors(texts)

    async def decide(payload):
        calls.append("plan")
        assert len(payload["facts"]) <= 2
        assert "_ai_knot_planning" not in payload["progress"]
        return proposal()

    selector = SemanticContext(memory, embed=embed)
    result = await invoke(
        recipe,
        memory,
        view,
        decide,
        max_model_calls=1,
        max_facts=2,
        selector=selector,
        semantic_budget=SemanticBudget(max_provider_calls=1),
    )
    assert (
        result.status == "planned"
        and result.model_calls_reserved == result.acquisition_calls_reserved == 1
    )
    assert result.context_guard == memory.task_guard(task_key="restore")
    saved = memory.get_intent(intent.id).progress["_ai_knot_planning"]
    assert saved["context_guard"]["digest"] == result.context_guard.digest
    reopened = AlwaysOnMemory(
        KnowledgeBase("owner", storage=sql_storage, embed_url=""), clock=clock
    )
    duplicate = await invoke(
        recipe,
        reopened,
        view,
        decide,
        max_model_calls=1,
        max_facts=2,
        selector=SemanticContext(reopened, embed=embed),
        semantic_budget=SemanticBudget(max_provider_calls=1),
    )
    assert duplicate.status == "unchanged" and duplicate.acquisition_calls_reserved == 0
    assert calls == ["embed", "plan"]


@pytest.mark.anyio
@pytest.mark.parametrize("when", ["embedding", "planning", "checkpoint"])
async def test_new_unselected_condition_rejects_selected_plan(
    recipe, sql_storage, clock, tmp_path, monkeypatch, when
):
    kb, memory, facts, intent, view = growing(recipe, sql_storage, clock, tmp_path)

    def add():
        kb.replace_facts([*facts, Fact("A separate condition now blocks work", id="late")])

    calls = []

    async def embed(texts):
        calls.append("embed")
        if when == "embedding":
            add()
        return vectors(texts)

    async def decide(payload):
        calls.append("plan")
        if when == "planning":
            add()
        return proposal()

    if when == "checkpoint":
        original = memory.checkpoint

        def late(*args, **kwargs):
            if kwargs.get("context_guard") is not None:
                add()
            return original(*args, **kwargs)

        monkeypatch.setattr(memory, "checkpoint", late)
    result = await invoke(
        recipe,
        memory,
        view,
        decide,
        max_model_calls=1,
        max_facts=2,
        selector=SemanticContext(memory, embed=embed),
        semantic_budget=SemanticBudget(max_provider_calls=1),
    )
    assert result.status == "context_changed" and result.proposal is None
    assert result.acquisition_calls_reserved == 1
    assert result.model_calls_reserved == (0 if when == "embedding" else 1)
    assert memory.get_intent(intent.id).next_step == "" and memory.get_intent(intent.id).basis == ()


@pytest.mark.anyio
async def test_zero_acquisition_budget_can_be_increased_without_paid_failure_suppression(
    recipe, sql_storage, clock, tmp_path
):
    _, memory, _, _, view = growing(recipe, sql_storage, clock, tmp_path)
    calls = []

    async def embed(texts):
        calls.append("embed")
        return vectors(texts)

    async def decide(payload):
        calls.append("plan")
        return proposal()

    selector = SemanticContext(memory, embed=embed)
    cold = await invoke(
        recipe, memory, view, decide, max_model_calls=1, max_facts=2, selector=selector
    )
    assert cold.status == "acquisition_budget_exhausted" and cold.acquisition_calls_reserved == 0
    ready = await invoke(
        recipe,
        memory,
        view,
        decide,
        max_model_calls=1,
        max_facts=2,
        selector=selector,
        semantic_budget=SemanticBudget(max_provider_calls=1),
    )
    assert ready.status == "planned" and calls == ["embed", "plan"]


@pytest.mark.anyio
@pytest.mark.parametrize("cancel", [False, True])
async def test_failed_or_cancelled_acquisition_keeps_reservation_and_deduplicates(
    recipe, sql_storage, clock, tmp_path, cancel
):
    _, memory, _, _, view = growing(recipe, sql_storage, clock, tmp_path)
    calls = []

    async def embed(texts):
        calls.append("embed")
        if cancel:
            raise asyncio.CancelledError()
        raise OSError("private provider detail")

    async def forbidden(payload):
        raise AssertionError("No plan after failed acquisition")

    selector = SemanticContext(memory, embed=embed)
    kwargs = {
        "max_model_calls": 1,
        "max_facts": 2,
        "selector": selector,
        "semantic_budget": SemanticBudget(max_provider_calls=1),
    }
    if cancel:
        with pytest.raises(asyncio.CancelledError):
            await invoke(recipe, memory, view, forbidden, **kwargs)
    else:
        result = await invoke(recipe, memory, view, forbidden, **kwargs)
        assert result.status == "acquisition_unavailable" and result.acquisition_calls_reserved == 1
    assert (await invoke(recipe, memory, view, forbidden, **kwargs)).status == "unchanged"
    assert calls == ["embed"]


@pytest.mark.anyio
async def test_supplied_selector_stays_quiet_without_event_or_planner_budget(
    recipe, sql_storage, clock, tmp_path, monkeypatch
):
    _, memory, _, _, view = growing(recipe, sql_storage, clock, tmp_path)

    async def forbidden(_):
        raise AssertionError("No provider call authorized")

    def no_scan(**kwargs):
        raise AssertionError("No acquisition scan authorized")

    monkeypatch.setattr(memory, "task_guard", no_scan)
    args = {
        "selector": SemanticContext(memory, embed=forbidden),
        "semantic_budget": SemanticBudget(max_provider_calls=1),
    }
    idle = await invoke(
        recipe, memory, view, forbidden, trigger_key=None, max_model_calls=1, **args
    )
    no_budget = await invoke(recipe, memory, view, forbidden, max_model_calls=0, **args)
    assert idle.status == "idle" and no_budget.status == "budget_exhausted"
    assert idle.acquisition_calls_reserved == no_budget.acquisition_calls_reserved == 0


@pytest.mark.anyio
async def test_fitting_snapshot_never_pays_supplied_selector(recipe, sql_storage, clock, tmp_path):
    _, memory, _, _, view = available(recipe, sql_storage, clock, tmp_path)

    async def forbidden(_):
        raise AssertionError("Whole task snapshot needs no embedding")

    async def decide(payload):
        return proposal()

    result = await invoke(
        recipe,
        memory,
        view,
        decide,
        max_model_calls=1,
        selector=SemanticContext(memory, embed=forbidden),
        semantic_budget=SemanticBudget(max_provider_calls=1),
    )
    assert result.status == "planned" and result.acquisition_calls_reserved == 0


@pytest.mark.anyio
async def test_cancel_while_waiting_for_warm_selector_without_budget_does_not_spend_event(
    recipe, sql_storage, clock, tmp_path
):
    _, memory, _, intent, view = growing(recipe, sql_storage, clock, tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    calls = []

    async def embed(texts):
        calls.append("embed")
        started.set()
        await release.wait()
        return vectors(texts)

    async def decide(payload):
        calls.append("plan")
        return proposal()

    selector = SemanticContext(memory, embed=embed)
    warming = asyncio.create_task(
        selector.acquire(
            ("backup receipt",),
            task_key="restore",
            max_facts=2,
            budget=SemanticBudget(max_provider_calls=1),
        )
    )
    await started.wait()
    pending = asyncio.create_task(
        invoke(recipe, memory, view, decide, max_model_calls=1, max_facts=2, selector=selector)
    )
    await asyncio.sleep(0)
    pending.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await pending
    finally:
        release.set()
        await warming
    marker = memory.get_intent(intent.id).progress["_ai_knot_planning"]
    assert marker["acquisition_calls_reserved"] == marker["model_calls_reserved"] == 0
    retry = await invoke(
        recipe, memory, view, decide, max_model_calls=1, max_facts=2, selector=selector
    )
    assert retry.status == "planned" and retry.acquisition_calls_reserved == 0
    assert calls == ["embed", "plan"]
