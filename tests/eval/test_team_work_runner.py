import asyncio
import json

from tests.eval.team_repos.cases import Case, Document, Stage
from tests.eval.team_work.state import memory_for, publish_stage, source_facts
from tests.eval.test_team_work_state import backend as backend


def sample():
    return Case(
        "coding",
        "task",
        "Coding",
        "Implement value",
        ("value",),
        "def value(): return 0",
        tuple(
            Stage(
                str(i),
                "Implement the current rule",
                (Document("rule.md", f"Return {i + 1} from value", "task"),),
                "PRIVATE_TEST",
                "PRIVATE_REFERENCE",
            )
            for i in range(3)
        ),
    )


def initial(storage, case):
    publish_stage(storage, case, 0)
    memory = memory_for(storage, 0)
    intent = memory.create_intent(client_key="code", goal=case.goal)
    lease = memory.claim(intent.id, worker_id="setup")
    memory.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=memory.get_intent(intent.id).revision,
        state="ready",
        progress={"code": case.initial_code, "last_receipt": None, "history": []},
    )
    return intent.id


class Client:
    def __init__(self, outputs, effect=None):
        self.outputs = iter(outputs)
        self.messages = []
        self.effect = effect

    async def complete(self, messages, **kwargs):
        self.messages.append(messages)
        if self.effect:
            self.effect()
        return next(self.outputs)


class Sandbox:
    def __init__(self, passed=True):
        self.codes = []
        self.passed = passed

    def run(self, code, tests):
        assert tests == "PRIVATE_TEST"
        self.codes.append(code)
        return {
            "status": "pass" if self.passed else "fail",
            "tests_run": 5,
            "failures": 0 if self.passed else 1,
            "errors": 0,
            "failed_test_names": [],
        }


def output(code="def value(): return 1"):
    return json.dumps({"action": "patch", "code": code, "basis": [1]})


def test_invalid_proposals_leave_old_code_and_stale_basis_intact(backend):
    from tests.eval.team_work.runner import run_stage

    case = sample()
    intent_id = initial(backend, case)
    memory = memory_for(backend, 0)
    basis = memory.capture_evidence((("__shared__", source_facts(case, 0)[0].id),))
    lease = memory.claim(intent_id, worker_id="old")
    memory.checkpoint(
        intent_id,
        lease_token=lease.token,
        expected_revision=memory.get_intent(intent_id).revision,
        state="ready",
        progress={"code": "OWN_FAILED_CODE", "last_receipt": None, "history": []},
        basis=basis,
    )
    publish_stage(backend, case, 1)
    sandbox = Sandbox()
    result = asyncio.run(
        run_stage(
            backend,
            case,
            1,
            "direct_sources",
            intent_id,
            None,
            Client(["bad JSON", "bad JSON"]),
            sandbox,
        )
    )
    after = memory.get_intent(intent_id)
    assert not result["tests_passed"] and len(result["attempts"]) == 2
    assert after.progress["code"] == "OWN_FAILED_CODE" and after.basis == basis
    assert after.state.value == "blocked" and after.lease is None and not sandbox.codes


def test_new_proposal_replaces_basis_only_after_validation_and_executes_guarded(backend):
    from tests.eval.team_work.runner import run_stage

    case = sample()
    intent_id = initial(backend, case)
    sandbox = Sandbox()
    client = Client([output()])
    result = asyncio.run(
        run_stage(backend, case, 0, "direct_sources", intent_id, None, client, sandbox)
    )
    after = memory_for(backend, 0).get_intent(intent_id)
    assert result["tests_passed"] and sandbox.codes == ["def value(): return 1"]
    assert (
        after.progress["code"] == sandbox.codes[0]
        and after.progress["last_receipt"]["status"] == "pass"
    )
    assert after.state.value == "waiting" and len(after.basis) == 1
    assert "PRIVATE_TEST" not in json.dumps(client.messages)
    assert "PRIVATE_REFERENCE" not in json.dumps(client.messages)


def test_truncation_is_rejected_even_if_the_partial_json_happens_to_parse(backend):
    from tests.eval.team_work.runner import run_stage

    case = sample()
    intent_id = initial(backend, case)
    client = Client([output(), output()])
    client.completion_truncated = True
    sandbox = Sandbox()
    result = asyncio.run(
        run_stage(backend, case, 0, "direct_sources", intent_id, None, client, sandbox)
    )
    assert len(result["attempts"]) == 2 and all(a["truncated"] for a in result["attempts"])
    assert not sandbox.codes and not result["tests_passed"]
    assert memory_for(backend, 0).get_intent(intent_id).progress["code"] == case.initial_code


def test_embedding_budget_refusal_keeps_its_type_for_arm_isolation(backend):
    from dataclasses import replace

    import httpx

    from tests.eval.team_decisions.openai import MeteredOpenAI
    from tests.eval.team_work.provider import RecordingClient
    from tests.eval.team_work.runner import run_stage

    case = sample()
    intent_id = initial(backend, case)
    root = source_facts(case, 0)[0]
    backend.atomic_update(
        "__shared__",
        lambda rows: (
            rows
            + [replace(root, id="large", content="Large source " + "x" * 2100, slot_key="large")]
        ),
    )
    sandbox = Sandbox()

    def forbidden(request):
        raise AssertionError("Budget refusal reached HTTP")

    async def run():
        client = RecordingClient(
            MeteredOpenAI("offline", max_calls=0, transport=httpx.MockTransport(forbidden)),
            on_update=lambda snapshot: None,
        )
        async with client:
            result = await run_stage(backend, case, 0, "semantic", intent_id, None, client, sandbox)
        assert result["error_type"] == "BudgetExceeded"
        assert client.snapshot()["requests"][0]["error_type"] == "BudgetExceeded"
        assert client.spent_tokens == client.actual_provider_calls == 0
        assert not sandbox.codes
        assert memory_for(backend, 0).get_intent(intent_id).progress["code"] == case.initial_code

    asyncio.run(run())


def test_history_survives_invalid_proposals_without_replacing_saved_basis(backend):
    from tests.eval.team_work.runner import run_stage

    case = sample()
    intent_id = initial(backend, case)
    first = asyncio.run(
        run_stage(
            backend, case, 0, "raw_team_history", intent_id, None, Client(["bad", "bad"]), Sandbox()
        )
    )
    saved = memory_for(backend, 0).get_intent(intent_id)
    assert saved.progress["history"] == [] and saved.progress["code"] == case.initial_code
    publish_stage(backend, case, 1)
    client = Client(
        [json.dumps({"action": "patch", "code": "def value(): return 2", "basis": [2]})]
    )
    second = asyncio.run(
        run_stage(
            backend,
            case,
            1,
            "raw_team_history",
            intent_id,
            None,
            client,
            Sandbox(),
            observed_history=first["observed_history"],
        )
    )
    assert second["tests_passed"]
    payload = json.loads(client.messages[0][-1]["content"])
    assert [r["version"] for r in payload["entries"]] == [1, 2]
    assert payload["current_code"] == case.initial_code


def test_late_unselected_source_prevents_candidate_application(backend):
    from ai_knot.types import Fact
    from tests.eval.team_work.runner import run_stage

    case = sample()
    intent_id = initial(backend, case)

    def late():
        stamp = source_facts(case, 0)[0].created_at
        backend.atomic_update(
            "__shared__",
            lambda old: (
                old + [Fact("New task condition", id="late", created_at=stamp, valid_from=stamp)]
            ),
        )

    sandbox = Sandbox()
    result = asyncio.run(
        run_stage(
            backend, case, 0, "direct_sources", intent_id, None, Client([output()], late), sandbox
        )
    )
    assert result["status"] == "context_changed" and not sandbox.codes
    assert memory_for(backend, 0).get_intent(intent_id).progress["code"] == case.initial_code


def test_history_render_preserves_temporal_order_and_same_set_equivalence():
    from tests.eval.team_work.runner import render_entries

    old = {
        "index": 9,
        "content": "Old",
        "version": 1,
        "observed_at": "2026-01-01T00:00:00+00:00",
        "ref": {"namespace": "__shared__", "fact_id": "same", "revision": "z"},
    }
    new = {
        "index": 3,
        "content": "New",
        "version": 2,
        "observed_at": "2026-01-02T00:00:00+00:00",
        "ref": {"namespace": "__shared__", "fact_id": "same", "revision": "a"},
    }
    assert [r["content"] for r in render_entries([new, old])] == ["Old", "New"]
    assert render_entries([new, old]) == render_entries([old, new])


def test_stale_history_citation_gets_second_proposal_without_clearing_basis(backend):
    from tests.eval.team_repos.runner import entries
    from tests.eval.team_work.runner import run_stage

    case = sample()
    intent_id = initial(backend, case)
    memory = memory_for(backend, 0)
    old = source_facts(case, 0)[0]
    refs = memory.capture_evidence((("__shared__", old.id),))
    lease = memory.claim(intent_id, worker_id="saved")
    memory.checkpoint(
        intent_id,
        lease_token=lease.token,
        expected_revision=memory.get_intent(intent_id).revision,
        state="ready",
        basis=refs,
        progress={"code": "def value(): return 1", "last_receipt": None, "history": entries([old])},
    )
    publish_stage(backend, case, 1)
    good = json.dumps({"action": "patch", "code": "def value(): return 2", "basis": [2]})
    result = asyncio.run(
        run_stage(
            backend,
            case,
            1,
            "raw_team_history",
            intent_id,
            None,
            Client([output(), good]),
            Sandbox(),
        )
    )
    assert [r["status"] for r in result["attempts"]] == ["invalid_proposal", "tested"]
    assert result["tests_passed"]


def test_handoff_never_binds_unseen_new_root_revision(backend, monkeypatch):
    from dataclasses import replace

    from ai_knot.always_on.lineage import evidence_roots
    from tests.eval.team_work.runner import publish_handoff

    case = sample()
    publish_stage(backend, case, 0)
    memory = memory_for(backend, 0)
    old = source_facts(case, 0)[0]
    refs = memory.capture_evidence((("__shared__", old.id),))
    original = backend.load
    changed = False

    def intervening_load(namespace):
        nonlocal changed
        rows = original(namespace)
        if namespace == "__shared__" and not changed:
            changed = True
            backend.atomic_update(
                "__shared__",
                lambda facts: [
                    replace(f, content="Changed root", version=f.version + 1)
                    if f.id == old.id
                    else f
                    for f in facts
                ],
            )
        return rows

    monkeypatch.setattr(backend, "load", intervening_load)
    publish_handoff(
        backend,
        case,
        0,
        {
            "summary": "Old observation",
            "basis": [
                {"namespace": r.namespace, "fact_id": r.fact_id, "revision": r.revision}
                for r in refs
            ],
        },
    )
    note = next(f for f in original("__shared__") if f.origin_agent_id == "repo-researcher")
    assert evidence_roots("__shared__", note) == refs
