"""Frozen semantic source characterizes the task-admission extraction, not quality."""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import types
from dataclasses import asdict, replace
from datetime import timedelta
from pathlib import Path

import pytest
from tests.eval.semantic_context_cases import CASES
from tests.eval.semantic_context_runner import _seed_memory

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import AlwaysOnMemory, SemanticBudget, SemanticContext
from ai_knot.types import Fact, MemoryType


@pytest.fixture(scope="module")
def reference_semantic():
    path = Path(__file__).resolve().parents[2] / "docs/performance/semantic-context-runtime.json.gz"
    archive = json.loads(gzip.decompress(path.read_bytes()))
    source_path = "src/ai_knot/always_on/semantic.py"
    source = archive["files"][source_path]
    expected = "bac58b11b6153bbda26b5a130b92dbdd05e8c7a3306bbdcf51545ff1b47faf06"
    assert hashlib.sha256(source.encode()).hexdigest() == expected
    assert archive["source_sha256"][source_path] == expected
    # Only the archived selector is historical; shared admission/assembly dependencies
    # remain current so the comparison isolates this extraction.
    name = "ai_knot.always_on._semantic_before_task_snapshot"
    module = types.ModuleType(name)
    module.__package__ = "ai_knot.always_on"
    sys.modules[name] = module
    try:
        exec(compile(source, f"{path.name}:{source_path}", "exec"), module.__dict__)
        yield module
    finally:
        sys.modules.pop(name, None)


def selectors(memory, reference):
    calls = [[], []]

    def callback(number):
        async def embed(texts):
            calls[number].append(list(texts))
            return [
                [1.0, *(value / 255 for value in hashlib.sha256(text.encode()).digest()[:7])]
                for text in texts
            ]

        return embed

    return (
        reference.SemanticContext(memory, embed=callback(0)),
        SemanticContext(memory, embed=callback(1)),
        calls,
    )


async def same_acquisition(old, current, calls, reference, requirements, *, limits=None, **kwargs):
    limits = limits or {}
    before = [len(items) for items in calls]
    previous = await old.acquire(requirements, budget=reference.SemanticBudget(**limits), **kwargs)
    result = await current.acquire(requirements, budget=SemanticBudget(**limits), **kwargs)
    assert asdict(previous) == asdict(result)
    assert calls[0] == calls[1]
    assert [len(items) - start for items, start in zip(calls, before, strict=True)] == [
        result.provider_calls_reserved,
        result.provider_calls_reserved,
    ]
    return result


@pytest.mark.anyio
@pytest.mark.parametrize("reverse", [False, True])
async def test_equal_timestamp_nonmonotonic_ids_keep_embedding_input_order(
    sql_storage, clock, reference_semantic, reverse
):
    facts = [
        Fact("Recovery copy available", id="z-first", qualifiers={"task_key": "restore"}),
        Fact("Receipt is retained", id="a-second", qualifiers={"task_key": "restore"}),
        Fact("Route is current", id="m-third", qualifiers={"task_key": "restore"}),
        *[
            Fact(f"Inventory {i}", id=f"noise-{i:03d}", qualifiers={"task_key": "other"})
            for i in range(150)
        ],
    ]
    sql_storage.save("owner", list(reversed(facts)) if reverse else facts)
    memory = AlwaysOnMemory(KnowledgeBase("owner", storage=sql_storage, embed_url=""), clock=clock)
    old, current, calls = selectors(memory, reference_semantic)
    await same_acquisition(
        old,
        current,
        calls,
        reference_semantic,
        ("restore readiness",),
        limits={"max_provider_calls": 1},
        task_key="restore",
        max_facts=1,
    )
    assert len(calls[0]) == len(calls[1]) == 1


@pytest.mark.anyio
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
async def test_frozen_cases_keep_admission_inputs_cache_and_limits(
    sql_storage,
    reference_semantic,
    case,
):
    memory = _seed_memory(sql_storage, case)
    old, current, calls = selectors(memory, reference_semantic)
    kwargs = {"task_key": case.task_key, "max_facts": 3, "max_chars": 600}
    cold = await same_acquisition(
        old,
        current,
        calls,
        reference_semantic,
        (case.query,),
        limits={"max_provider_calls": 1},
        **kwargs,
    )
    assert cold.status in ("snapshot", "ok")
    warm = await same_acquisition(old, current, calls, reference_semantic, (case.query,), **kwargs)
    assert warm.provider_calls_reserved == 0
    assert warm.review == cold.review
    # New requirements cannot spend on a cold query unless explicitly authorized.
    changed = (case.query + " Check only the retained sources.",)
    unspent = await same_acquisition(old, current, calls, reference_semantic, changed, **kwargs)
    assert unspent.status == ("snapshot" if cold.status == "snapshot" else "budget_exhausted")
    limited = await same_acquisition(
        old,
        current,
        calls,
        reference_semantic,
        changed,
        limits={"max_input_bytes": 1, "max_provider_calls": 1},
        **kwargs,
    )
    assert limited.status == ("snapshot" if cold.status == "snapshot" else "input_limit")
    await same_acquisition(
        old,
        current,
        calls,
        reference_semantic,
        (case.query,),
        limits={"max_candidates": 1, "max_provider_calls": 1},
        **kwargs,
    )


@pytest.mark.anyio
@pytest.mark.parametrize("task_key", ["", "repair"])
async def test_generic_namespace_collisions_and_source_withdrawal_keep_cached_semantics(
    sql_storage,
    clock,
    reference_semantic,
    task_key,
):
    def fact(content, fact_id, **kwargs):
        return Fact(content, id=fact_id, created_at=clock(), valid_from=clock(), **kwargs)

    private = fact("Private generic receipt", "same")
    task = fact("Repair isolation rule", "task", qualifiers={"task_key": "repair"})
    kb = KnowledgeBase("owner", storage=sql_storage, embed_url="")
    kb.replace_facts(
        [
            private,
            task,
            fact("Other task private content", "wrong", qualifiers={"task_key": "different"}),
            fact("Unsupported private content", "unsupported", supported=False),
            replace(
                fact("Future private content", "future"), valid_from=clock() + timedelta(days=1)
            ),
            fact("Raw episode content", "episode", type=MemoryType.EPISODIC),
        ]
    )
    shared = fact("Shared generic receipt", "same", visibility_scope="ops", origin_agent_id="peer")
    sql_storage.save(
        "__shared__",
        [
            shared,
            fact(
                "Denied shared content", "denied", visibility_scope="closed", origin_agent_id="peer"
            ),
        ],
    )
    pool = SharedMemoryPool(sql_storage, persist_stats=True, clock=clock, embed_url="")
    pool.grant_read("owner", "ops")
    memory = AlwaysOnMemory(kb, pool=pool, clock=clock)
    memory.derive_fact(
        "Summary retaining the shared root",
        task_key=task_key,
        basis=memory.capture_evidence((("__shared__", "same"),)),
    )
    old, current, calls = selectors(memory, reference_semantic)
    kwargs = {"task_key": task_key, "max_facts": 1, "max_chars": 600}
    await same_acquisition(
        old,
        current,
        calls,
        reference_semantic,
        ("receipt readiness",),
        limits={"max_provider_calls": 1},
        **kwargs,
    )
    assert "Private generic receipt" in calls[0][0]
    assert "Shared generic receipt" in calls[0][0]
    assert not any("content" in text for text in calls[0][0])
    await same_acquisition(
        old, current, calls, reference_semantic, ("receipt readiness",), **kwargs
    )

    kb.replace_facts(
        [
            replace(item, content="Private generic receipt corrected")
            if item.id == "same"
            else item
            for item in kb.list_facts()
        ]
    )
    await same_acquisition(
        old,
        current,
        calls,
        reference_semantic,
        ("receipt readiness",),
        limits={"max_provider_calls": 1},
        **kwargs,
    )
    assert calls[0][-1] == ["Private generic receipt corrected"]
    pool.revoke_read("owner", "ops")
    withdrawn = await same_acquisition(
        old, current, calls, reference_semantic, ("receipt readiness",), **kwargs
    )
    assert all(item.ref.namespace == "owner" for item in withdrawn.review.accepted)
    assert all("shared" not in item.content.lower() for item in withdrawn.review.accepted)
