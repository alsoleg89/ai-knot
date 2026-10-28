import asyncio
import json
import os
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from ai_knot.always_on import EvidenceRef
from tests.eval.task_context.runner import isolated_backend
from tests.eval.team_repos import runner


@pytest.fixture(params=("sqlite", "postgres"))
def repo_storage(request, tmp_path):
    if request.param == "postgres" and not os.environ.get("AI_KNOT_TEST_PG_DSN"):
        if os.environ.get("AI_KNOT_REQUIRE_POSTGRES") == "1":
            pytest.fail("PostgreSQL DSN required")
        pytest.skip("PostgreSQL DSN not configured")
    with isolated_backend(request.param, tmp_path) as storage:
        yield storage


def fixture():
    doc = SimpleNamespace(path="docs/rule.md", text="Return the value seven.", task_key="sample")
    stage = SimpleNamespace(
        id="first", brief="Implement value().", documents=(doc,), test_source=""
    )
    return SimpleNamespace(
        id="repository-sample",
        task_key="sample",
        goal="Implement the documented API",
        requirements=("value",),
        initial_code="def value(): raise NotImplementedError",
        stages=(stage,),
    )


def test_stage_uses_public_sources_and_persists_its_own_candidate(repo_storage):
    case = fixture()
    storage = repo_storage
    docs = runner.projected_documents(case, 0)
    runner.seed_stage(storage, case, 0, None)
    memory = runner.memory_for(storage, 0)
    intent = memory.create_intent(client_key="repo", goal=case.goal)
    lease = memory.claim(intent.id, worker_id="seed")
    memory.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=memory.get_intent(intent.id).revision,
        state="ready",
        progress={"code": case.initial_code, "history": runner.entries(docs)},
    )

    class Client:
        calls = []
        spent_tokens = 0

        async def complete(self, messages, **kwargs):
            self.calls.append({"kind": "complete", "charged_tokens": 10})
            self.spent_tokens += 10
            assert "reference_solution" not in json.dumps(messages)
            return json.dumps({"action": "patch", "code": "def value(): return 7", "basis": [1]})

    class Sandbox:
        def run(self, code, tests):
            assert code == "def value(): return 7"
            return {
                "status": "pass",
                "tests_run": 1,
                "failures": 0,
                "errors": 0,
                "failed_test_names": [],
                "exit_code": 0,
            }

    row = asyncio.run(
        runner.run_stage(storage, case, 0, "direct_sources", intent.id, Client(), Sandbox())
    )
    assert row["status"] == "complete" and row["tests_passed"]
    saved = memory.get_intent(intent.id)
    assert saved.progress["code"] == "def value(): return 7" and saved.state == "completed"
    assert row["attempts"][0]["basis_valid"] and row["spent_tokens"] == 10
    assert row["attempts"][0]["candidate_code"] == saved.progress["code"]


def test_source_revision_and_stable_handoff_are_independent_of_id_order(repo_storage):
    case = fixture()
    revised = SimpleNamespace(
        id="second",
        brief="Use the corrected rule",
        test_source="",
        documents=(SimpleNamespace(path="docs/rule.md", text="Return eight.", task_key="sample"),),
    )
    case.stages = (*case.stages, revised)
    first = runner.projected_documents(case, 0)
    second = runner.projected_documents(case, 1)
    assert first[0].id == second[0].id and first[0].version < second[0].version
    storage = repo_storage
    ref = EvidenceRef("__shared__", first[0].id, runner.evidence_revision(first[0]))
    note = {"summary": "Earlier rule", "basis": [asdict(ref)]}
    runner.seed_stage(storage, case, 0, note)
    memory = runner.memory_for(storage, 0)
    old = memory.task_snapshot(task_key=case.task_key)
    assert len(old) == 2
    runner.seed_stage(storage, case, 1, None)
    current = runner.memory_for(storage, 1).task_snapshot(task_key=case.task_key)
    assert len(current) == 1 and not memory.review_context(tuple(f.ref for f in old)).complete


def test_history_cap_counts_rendered_observation_metadata(repo_storage):
    case = fixture()
    storage = repo_storage
    runner.seed_stage(storage, case, 0, None)
    memory = runner.memory_for(storage, 0)
    history = runner.entries(runner.projected_documents(case, 0))
    history[0]["observed_at"] = "x" * runner.MAX_CONTEXT_CHARS
    with pytest.raises(ValueError, match="context exceeds"):
        asyncio.run(runner.context_for(storage, memory, case, 0, "raw_team_history", None, history))


def test_conflicting_history_citations_consume_an_attempt_without_aborting(repo_storage):
    case = fixture()
    revised = SimpleNamespace(
        id="second",
        brief="Use the current rule",
        test_source="",
        documents=(SimpleNamespace(path="docs/rule.md", text="Return eight.", task_key="sample"),),
    )
    case.stages = (*case.stages, revised)
    old = runner.entries(runner.projected_documents(case, 0))[0]
    runner.seed_stage(repo_storage, case, 1, None)
    current = runner.entries(runner.projected_documents(case, 1))[0]
    memory = runner.memory_for(repo_storage, 1)
    intent = memory.create_intent(client_key="rejection", goal=case.goal)
    lease = memory.claim(intent.id, worker_id="seed")
    memory.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=memory.get_intent(intent.id).revision,
        state="ready",
        progress={"code": case.initial_code, "history": [old, current]},
    )

    class Client:
        calls = []
        spent_tokens = 0

        async def complete(self, messages, **kwargs):
            self.calls.append({"kind": "complete", "charged_tokens": 10})
            self.spent_tokens += 10
            return json.dumps(
                {
                    "action": "patch",
                    "code": "def value(): return 8",
                    "basis": [1, 2] if len(self.calls) == 1 else [2],
                }
            )

    class Sandbox:
        calls = 0

        def run(self, code, tests):
            self.calls += 1
            return {
                "status": "pass",
                "tests_run": 1,
                "failures": 0,
                "errors": 0,
                "failed_test_names": [],
                "exit_code": 0,
            }

    sandbox = Sandbox()
    row = asyncio.run(
        runner.run_stage(repo_storage, case, 1, "raw_team_history", intent.id, Client(), sandbox)
    )
    assert row["status"] == "complete" and row["tests_passed"]
    assert row["spent_tokens"] == 20 and len(row["attempts"]) == 2
    assert row["attempts"][0]["status"] == "invalid_proposal"
    assert row["attempts"][0]["error_type"] == "StaleEvidence"
    assert not row["attempts"][0]["applied"] and sandbox.calls == 1


def test_aborted_stage_does_not_report_unchecked_sources_as_mutated(repo_storage):
    case = fixture()
    runner.seed_stage(repo_storage, case, 0, None)
    memory = runner.memory_for(repo_storage, 0)
    intent = memory.create_intent(client_key="abort", goal=case.goal)
    lease = memory.claim(intent.id, worker_id="seed")
    memory.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=memory.get_intent(intent.id).revision,
        state="ready",
        progress={"code": case.initial_code, "history": []},
    )

    class Client:
        calls = [{"kind": "complete", "charged_tokens": 10}]
        spent_tokens = 10

        async def complete(self, *args, **kwargs):
            raise RuntimeError("Provider failure")

    row = asyncio.run(
        runner.run_stage(repo_storage, case, 0, "direct_sources", intent.id, Client(), None)
    )
    assert row["status"] == "failed" and row["source_unchanged"] is None
    assert row["spent_tokens"] == 10 and len(row["calls"]) == 1
