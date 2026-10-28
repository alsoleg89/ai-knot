"""Offline reader transport checks exercise real selectors and request accounting."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace

import httpx
import pytest

from ai_knot.storage.sqlite_storage import SQLiteStorage

from . import semantic_context_runner as runner
from .semantic_context_cases import CASES
from .team_decisions.openai import MeteredOpenAI


@pytest.fixture
def evaluation():
    return runner


def transport(raw='{"choice":"ABSTAIN","basis":[]}'):
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if request.url.path.endswith("/embeddings"):
            return httpx.Response(
                200,
                json={
                    "model": "text-embedding-3-small",
                    "data": [
                        {"index": i, "embedding": [1.0] + [0.0] * 1535}
                        for i in range(len(body["input"]))
                    ],
                    "usage": {"prompt_tokens": 20, "total_tokens": 20},
                },
            )
        return httpx.Response(
            200,
            json={
                "model": "gpt-4.1-mini-2025-04-14",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": raw(requests) if callable(raw) else raw,
                        },
                    }
                ],
                "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
            },
        )

    return httpx.MockTransport(respond), requests


def entries():
    return [
        {
            "index": 1,
            "ref": {
                "namespace": "__shared__",
                "fact_id": "source-one",
                "revision": "first-revision",
            },
            "content": "First condition holds.",
        },
        {
            "index": 2,
            "ref": {
                "namespace": "__shared__",
                "fact_id": "source-two",
                "revision": "second-revision",
            },
            "content": "Second condition holds.",
        },
    ]


def test_reader_input_cannot_observe_gold_or_source_ids(evaluation):
    case = CASES[0]
    altered = replace(case, expected_choice="SECRET_GOLD", required_groups=(("SECRET_GROUP",),))
    original = evaluation.reader_messages(case, entries())
    assert evaluation.reader_messages(altered, entries()) == original
    payload = json.loads(original[-1]["content"])
    assert payload["query"] == case.query
    assert payload["context"] == [
        {"index": 1, "content": "First condition holds."},
        {"index": 2, "content": "Second condition holds."},
    ]
    assert not {"expected_choice", "required_groups", "gold"} & payload.keys()
    schema = evaluation.decision_schema(2)
    assert schema["properties"]["choice"]["enum"] == ["A", "B", "ABSTAIN"]
    assert schema["properties"]["basis"]["items"]["enum"] == [1, 2]


@pytest.mark.parametrize(
    ("raw", "correct", "grounded"),
    [
        ('{"choice":"A","basis":[1,2]}', True, True),
        ('{"choice":"A","basis":[1]}', True, False),
        ('{"choice":"B","basis":[1,2]}', False, False),
    ],
)
def test_grounded_scoring_requires_every_independent_group(evaluation, raw, correct, grounded):
    case = replace(
        CASES[0],
        expected_choice="A",
        required_groups=(("source-one", "alternative"), ("source-two",)),
    )
    scored = evaluation.score_response(case, entries(), raw)
    assert scored["correct_choice"] is correct
    assert scored["all_required_groups_visible"]
    assert scored["grounded_correct"] is grounded


@pytest.mark.parametrize(
    "raw",
    [
        '{"choice":"A","basis":[0]}',
        '{"choice":"A","basis":[3]}',
        '{"choice":"A","basis":[true]}',
        '{"choice":"A","basis":[1,1]}',
        '{"choice":"ABSTAIN","basis":[1]}',
        '{"choice":"C","basis":[]}',
        '{"choice":"A","choice":"B","basis":[]}',
    ],
)
def test_invalid_or_ambiguous_visible_citations_fail_closed(evaluation, raw):
    with pytest.raises(ValueError):
        evaluation.score_response(CASES[0], entries(), raw)


def test_supported_abstention_requires_no_citations(evaluation):
    case = replace(CASES[0], expected_choice="ABSTAIN", required_groups=())
    scored = evaluation.score_response(case, entries(), '{"choice":"ABSTAIN","basis":[]}')
    assert scored["correct_choice"] and scored["grounded_correct"]


@pytest.mark.anyio
@pytest.mark.parametrize("arm", ["lexical_task_context", "scoped_bundle", "semantic_context"])
async def test_equal_context_and_reader_caps_exclude_denied_sources(evaluation, arm, tmp_path):
    case = next(case for case in CASES if case.id == "queue-capacity")
    wire, requests = transport()
    async with MeteredOpenAI(
        "offline-key", max_calls=2, max_tokens=12000, transport=wire
    ) as client:
        row = await evaluation.run_case(
            SQLiteStorage(str(tmp_path / "eval.db")), case, arm, client, mode="offline_transport"
        )
    assert row["status"] == "complete"
    assert len(row["context"]) <= 3
    assert sum(len(entry["content"]) for entry in row["context"]) <= 600
    visible = {entry["ref"]["fact_id"] for entry in row["context"]}
    assert "b2000007" not in visible  # Same task, denied queue-lab scope.
    assert "b2000002" not in visible  # Readable, different task.
    denied = next(f.content for f in case.facts if f.id == "b2000007")
    wrong_task = next(f.content for f in case.facts if f.id == "b2000002")
    assert denied not in json.dumps(requests, ensure_ascii=False)
    assert wrong_task not in json.dumps(requests, ensure_ascii=False)
    complete = next(body for body in requests if "messages" in body)
    payload = json.loads(complete["messages"][-1]["content"])
    assert payload["query"] == case.query
    assert complete["max_completion_tokens"] == 128
    assert complete["response_format"]["json_schema"]["strict"] is True
    assert row["spent_tokens"] == (60 if arm == "semantic_context" else 40)
    assert len(row["calls"]) == (2 if arm == "semantic_context" else 1)
    assert row["source_unchanged"]


@pytest.mark.anyio
async def test_shared_request_meter_prevents_reader_after_embedding_spends_last_call(
    evaluation,
    tmp_path,
):
    wire, requests = transport()
    async with MeteredOpenAI(
        "offline-key", max_calls=1, max_tokens=12000, transport=wire
    ) as client:
        row = await evaluation.run_case(
            SQLiteStorage(str(tmp_path / "eval.db")),
            CASES[0],
            "semantic_context",
            client,
            mode="offline_transport",
        )
    assert row["status"] == "failed" and row["error_type"] == "BudgetExceeded"
    assert not row["correct_choice"] and not row["grounded_correct"]
    assert len(requests) == 1 and "input" in requests[0]
    assert len(row["calls"]) == 1 and row["spent_tokens"] == 20


@pytest.mark.anyio
@pytest.mark.parametrize("escaped", [False, True])
async def test_credential_shaped_reader_output_is_redacted_and_not_scored(
    evaluation,
    tmp_path,
    escaped,
):
    fake_token = "sk-" + "synthetic_offline_token_" * 2
    raw = json.dumps({"choice": fake_token, "basis": []})
    if escaped:
        raw = raw.replace("sk-", "\\u0073k-")
    wire, _ = transport(raw)
    async with MeteredOpenAI(
        "offline-key", max_calls=2, max_tokens=12000, transport=wire
    ) as client:
        row = await evaluation.run_case(
            SQLiteStorage(str(tmp_path / "eval.db")),
            CASES[0],
            "scoped_bundle",
            client,
            mode="offline_transport",
        )
    assert row["status"] == "failed" and row["reader_output_redacted"]
    assert fake_token not in json.dumps(row)
    assert "[REDACTED]" in row["reader_output"]
    assert not row["correct_choice"] and not row["grounded_correct"]


@pytest.mark.anyio
async def test_suite_uses_identical_fresh_ceilings_and_retains_partial_report(
    evaluation,
    tmp_path,
    monkeypatch,
):
    ceilings = []
    wire, _ = transport()

    def client(key, **kwargs):
        ceilings.append(kwargs)
        return MeteredOpenAI(key, transport=wire, **kwargs)

    monkeypatch.setattr(evaluation, "MeteredOpenAI", client)
    output = tmp_path / "report.json"
    report = await evaluation.run_suite(
        "sqlite", "offline-key", output=output, mode="offline_transport"
    )
    assert len(report["records"]) == len(CASES) * 3
    assert ceilings == [{"max_calls": 2, "max_tokens": 12000}] * 48
    assert report["status"] == "complete"
    assert report["source_unchanged_during_run"]
    assert report["source_sha256"]["tests/eval/semantic_context_cases.py"]
    saved = json.loads(output.read_text())
    assert saved["status"] == "complete" and len(saved["records"]) == 48
    assert all(row["calls"] and row["spent_tokens"] <= 12000 for row in saved["records"])


@pytest.mark.anyio
async def test_reader_failure_is_saved_before_the_next_arm_starts(
    evaluation,
    tmp_path,
    monkeypatch,
):
    output = tmp_path / "partial.json"
    observed = []

    def response(requests):
        if len(requests) == 1:
            return "not JSON"
        observed.append(json.loads(output.read_text()))
        return '{"choice":"ABSTAIN","basis":[]}'

    wire, _ = transport(response)
    monkeypatch.setattr(
        evaluation,
        "MeteredOpenAI",
        lambda key, **kwargs: MeteredOpenAI(key, transport=wire, **kwargs),
    )
    monkeypatch.setattr(evaluation, "CASES", (CASES[0],))
    monkeypatch.setattr(evaluation, "ARMS", ("scoped_bundle", "lexical_task_context"))
    report = await evaluation.run_suite(
        "sqlite", "offline-key", output=output, mode="offline_transport"
    )
    assert len(observed) == 1 and len(observed[0]["records"]) == 1
    first = observed[0]["records"][0]
    assert first["status"] == "failed" and first["spent_tokens"] == 40
    assert first["reader_output"] == "not JSON" and first["calls"]
    assert report["status"] == "failed" and report["records"][1]["status"] == "complete"


@pytest.mark.anyio
@pytest.mark.parametrize("failure", ["storage_exit", "client_exit"])
async def test_cleanup_failure_preserves_already_charged_requests(
    evaluation,
    tmp_path,
    monkeypatch,
    failure,
):
    wire, requests = transport()

    class Client(MeteredOpenAI):
        async def __aexit__(self, *args):
            await super().__aexit__(*args)
            if failure == "client_exit":
                raise OSError("Sensitive cleanup details must not enter the report")

    def client(key, **kwargs):
        return Client(key, transport=wire, **kwargs)

    @contextmanager
    def backend(kind, directory):
        yield SQLiteStorage(str(directory / "cleanup.db"))
        if failure == "storage_exit":
            raise OSError("Sensitive cleanup details must not enter the report")

    monkeypatch.setattr(evaluation, "MeteredOpenAI", client)
    monkeypatch.setattr(evaluation, "isolated_backend", backend)
    monkeypatch.setattr(evaluation, "CASES", (CASES[0],))
    monkeypatch.setattr(evaluation, "ARMS", ("scoped_bundle",))
    output = tmp_path / "cleanup.json"
    report = await evaluation.run_suite(
        "sqlite", "offline-key", output=output, mode="offline_transport"
    )
    row = report["records"][0]
    assert row["status"] == "failed" and row["error_type"] == "OSError"
    assert len(requests) == len(row["calls"]) == 1
    assert row["spent_tokens"] == 40 and row["calls"][0]["usage"]["total_tokens"] == 40
    assert not row["correct_choice"] and not row["grounded_correct"]
    assert "Sensitive cleanup details" not in output.read_text()
    assert json.loads(output.read_text())["records"][0]["spent_tokens"] == 40
