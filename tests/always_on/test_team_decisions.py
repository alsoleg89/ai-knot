"""SQL/protocol checks use scripted proposals, never claim model quality."""

import json
import os

import pytest
from tests.eval.team_decisions.runner import ARMS, decision_schema, run_scenario
from tests.eval.team_decisions.world import SCENARIOS


class ScriptedReader:
    def __init__(self, scenario, unsafe=False):
        self.scenario = scenario
        self.unsafe = unsafe
        self.calls = []
        self.spent_tokens = 0
        self.inputs = []

    async def complete(self, messages, *, max_output_tokens=128, response_schema=None):
        payload = json.loads(messages[-1]["content"])
        assert response_schema == decision_schema(len(payload["context"]))
        assert [entry["index"] for entry in payload["context"]] == list(
            range(len(payload["context"]))
        )
        self.inputs.append(payload)
        step = len(self.inputs) - 1
        first, second = self.scenario.targets
        actions = (
            [("execute", first)] * 4 + [("execute", second)]
            if self.unsafe
            else [
                ("verify", first),
                ("execute", first),
                ("reconcile", first),
                ("verify", second),
                ("execute", second),
            ]
        )
        latest = {}
        for number, entry in enumerate(payload["context"]):
            if entry["supported"] and entry["task_key"] == self.scenario.task_key:
                identity = entry["ref"]["fact_id"]
                if identity not in latest or entry["version"] > latest[identity][1]:
                    latest[identity] = (number, entry["version"])
        action, target = actions[step]
        return json.dumps(
            {"action": action, "target": target, "basis": [value[0] for value in latest.values()]}
        )

    async def embed(self, texts, **kwargs):
        self.calls.append({"kind": "embed", "status": "ok"})
        return [[1.0, 0.0] for _ in texts]


@pytest.mark.anyio
@pytest.mark.parametrize("arm", ARMS)
async def test_scripted_decisions_complete_actual_effects_and_reload_sql(sql_storage, arm):
    reader = ScriptedReader(SCENARIOS[0])
    report = await run_scenario(sql_storage, SCENARIOS[0], arm, reader, mode="scripted")
    assert report["mode"] == "scripted"
    assert report["status"] == "complete"
    assert report["summary"]["completed"]
    assert report["summary"]["physical_effect_counts"] == dict.fromkeys(SCENARIOS[0].targets, 1)
    assert report["summary"]["unsafe_proposals"] == 0
    assert len(report["steps"]) == len(reader.inputs) == 5
    assert all(step["source_unchanged_by_selection"] for step in report["steps"])
    handoff = report["steps"][2]["handoff"]
    assert handoff["worker_pid"] != os.getpid()
    assert handoff["source_reloaded"] and handoff["host_progress_reloaded"]
    assert reader.inputs[2]["progress"]["operations"]
    for payload in reader.inputs:
        assert not {"oracle", "expected_action", "current_target", "ready"} & payload.keys()


@pytest.mark.anyio
async def test_guard_rejections_do_not_turn_bad_proposals_into_model_success(sql_storage):
    reader = ScriptedReader(SCENARIOS[0], unsafe=True)
    report = await run_scenario(sql_storage, SCENARIOS[0], "raw_history", reader, mode="scripted")
    assert report["status"] == "complete"  # The evaluation ran; the task failed.
    assert not report["summary"]["completed"]
    assert report["summary"]["unsafe_proposals"] >= 2
    assert report["summary"]["physical_effect_counts"][SCENARIOS[0].targets[0]] == 1
    assert report["summary"]["physical_effect_counts"][SCENARIOS[0].targets[1]] == 0
    assert not all(step["decision_matches_oracle"] for step in report["steps"])


@pytest.mark.anyio
async def test_failed_embedding_cannot_be_reported_as_successful_hybrid(sql_storage):
    from tests.eval.team_decisions.openai import EvaluationError

    class Unavailable(ScriptedReader):
        async def embed(self, texts, **kwargs):
            raise EvaluationError("offline provider failure")

    reader = Unavailable(SCENARIOS[0])
    report = await run_scenario(
        sql_storage, SCENARIOS[0], "hybrid_scoped_pool", reader, mode="scripted"
    )
    assert report["status"] == "failed"
    assert report["error_type"] == "EvaluationError"
    assert not report["summary"]["completed"]
    assert not reader.inputs


@pytest.mark.anyio
async def test_source_changes_during_reader_call_reject_its_old_basis(sql_storage):
    from dataclasses import replace

    class ConcurrentWriter(ScriptedReader):
        async def complete(self, messages, **kwargs):
            response = await super().complete(messages, **kwargs)
            if len(self.inputs) == 2:
                current = sql_storage.load("__shared__")
                target_id = SCENARIOS[0].id + "-target"
                sql_storage.save(
                    "__shared__",
                    [
                        replace(fact, content="Destination withdrawn while reader was running")
                        if fact.id == target_id
                        else fact
                        for fact in current
                    ],
                )
            return response

    reader = ConcurrentWriter(SCENARIOS[0])
    report = await run_scenario(sql_storage, SCENARIOS[0], "scoped_pool", reader, mode="scripted")
    assert report["status"] == "complete"
    assert not report["steps"][1]["basis_valid"]
    assert report["steps"][1]["outcome"]["reason"] == "invalid_basis"


@pytest.mark.anyio
async def test_malformed_reader_proposal_is_rejected_without_aborting_later_work(sql_storage):
    class MalformedFirst(ScriptedReader):
        async def complete(self, messages, **kwargs):
            response = await super().complete(messages, **kwargs)
            return "not JSON" if len(self.inputs) == 1 else response

    reader = MalformedFirst(SCENARIOS[0])
    report = await run_scenario(sql_storage, SCENARIOS[0], "scoped_pool", reader, mode="scripted")
    assert report["status"] == "complete"
    assert report["summary"]["completed"]
    assert len(report["steps"]) == 5
    assert report["steps"][0]["reader_error"] == "JSONDecodeError"
    assert report["steps"][0]["reader_output"] == "not JSON"
    assert not report["steps"][0]["decision_matches_oracle"]
    assert report["steps"][0]["outcome"]["status"] == "rejected"


@pytest.mark.anyio
async def test_conflicting_history_refs_are_a_bad_basis_not_a_failed_evaluation(sql_storage):
    class ConflictingHistory(ScriptedReader):
        async def complete(self, messages, **kwargs):
            response = await super().complete(messages, **kwargs)
            if len(self.inputs) == 2:
                value = json.loads(response)
                context = self.inputs[-1]["context"]
                old = next(
                    i
                    for i, item in enumerate(context)
                    if item["ref"]["fact_id"].endswith("canary") and item["version"] == 1
                )
                value["basis"].append(old)
                return json.dumps(value)
            return response

    report = await run_scenario(
        sql_storage, SCENARIOS[0], "raw_history", ConflictingHistory(SCENARIOS[0]), mode="scripted"
    )
    assert report["status"] == "complete"
    assert not report["steps"][1]["basis_valid"]
    assert report["steps"][1]["basis_issue"] == "StaleEvidence"
    assert report["steps"][1]["outcome"]["reason"] == "invalid_basis"


@pytest.mark.anyio
@pytest.mark.parametrize("escaped", [False, True])
async def test_key_shaped_reader_output_never_enters_reports_or_next_context(sql_storage, escaped):
    fake_token = "sk-" + "offline_not_a_real_key_" * 2

    class SensitiveOutput(ScriptedReader):
        async def complete(self, messages, **kwargs):
            response = await super().complete(messages, **kwargs)
            if len(self.inputs) == 1:
                value = json.loads(response)
                value["target"] = fake_token
                text = json.dumps(value)
                return text.replace("sk-", "\\u0073k-") if escaped else text
            return response

    reader = SensitiveOutput(SCENARIOS[0])
    report = await run_scenario(sql_storage, SCENARIOS[0], "scoped_pool", reader, mode="scripted")
    assert fake_token not in json.dumps(report)
    assert fake_token not in json.dumps(reader.inputs)
    assert report["steps"][0]["reader_output_redacted"]
    assert report["steps"][0]["outcome"]["status"] == "rejected"
