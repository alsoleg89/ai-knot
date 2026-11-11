"""Offline contracts for the decision benchmark's OpenAI budget boundary."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import traceback

import httpx
import pytest

from tests.eval.team_decisions import openai

KEY = "test-key-must-never-appear-in-public-data"
MESSAGES = [{"role": "user", "content": "Receipt confirmed"}]
VECTOR = [1.0] + [0.0] * 1535


def test_longer_coder_timeout_is_explicit_and_bounded():
    def handle(request):
        assert request.extensions["timeout"]["read"] == 90
        return httpx.Response(200, json=completion())

    async def run():
        async with openai.MeteredOpenAI(
            KEY, completion_timeout=90, transport=httpx.MockTransport(handle)
        ) as client:
            await client.complete(MESSAGES)
        for invalid in (0, True, float("nan"), 121):
            with pytest.raises(openai.EvaluationError):
                openai.MeteredOpenAI(KEY, completion_timeout=invalid)

    asyncio.run(run())


def test_truncated_completion_is_retained_and_larger_explicit_authority_is_bounded():
    from tests.eval.team_work.provider import RecordingClient

    def handle(request):
        assert json.loads(request.content)["max_completion_tokens"] == 8192
        return httpx.Response(
            200,
            json=completion(
                choices=[{"message": {"content": "partial source"}, "finish_reason": "length"}]
            ),
        )

    async def run():
        provider = openai.MeteredOpenAI(
            KEY, max_output_tokens_per_call=8192, transport=httpx.MockTransport(handle)
        )
        client = RecordingClient(provider, on_update=lambda snapshot: None)
        async with client:
            with pytest.raises(openai.EvaluationError):
                await client.complete(MESSAGES, max_output_tokens=8192)
        snapshot = client.snapshot()
        assert snapshot["spent_tokens"] == 31
        assert snapshot["requests"][0]["status"] == "failed"
        assert snapshot["raw_completions"] == [
            dict(meter_call_index=0, finish_reason="length", content="partial source")
        ]
        assert KEY not in json.dumps(snapshot)
        snapshot["raw_completions"].clear()
        assert len(client.snapshot()["raw_completions"]) == 1
        with pytest.raises(openai.EvaluationError):
            openai.MeteredOpenAI(KEY, max_output_tokens_per_call=8193)

    asyncio.run(run())


def test_raw_completion_keeps_its_call_index_when_responses_arrive_out_of_order():
    async def run():
        second = asyncio.Event()

        async def handle(request):
            label = json.loads(request.content)["messages"][0]["content"]
            if label == "first":
                await second.wait()
            else:
                second.set()
            return httpx.Response(
                200,
                json=completion(choices=[{"message": {"content": label}, "finish_reason": "stop"}]),
            )

        async with openai.MeteredOpenAI(KEY, transport=httpx.MockTransport(handle)) as client:
            await asyncio.gather(
                *(
                    client.complete([{"role": "user", "content": label}])
                    for label in ("first", "second")
                )
            )
        assert {r["content"]: r["meter_call_index"] for r in client.raw_completions} == {
            "first": 0,
            "second": 1,
        }

    asyncio.run(run())


def completion(**changes):
    body = {
        "id": "chatcmpl-offline",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4.1-mini-2025-04-14",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": '{"action":"verify"}'},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 24, "completion_tokens": 7, "total_tokens": 31},
    }
    body.update(changes)
    return body


def embedding(**changes):
    body = {
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [
            {"object": "embedding", "index": 1, "embedding": [0.0, 1.0] + [0.0] * 1534},
            {"object": "embedding", "index": 0, "embedding": VECTOR},
        ],
        "usage": {"prompt_tokens": 9, "total_tokens": 9},
    }
    body.update(changes)
    return body


def test_completion_and_embedding_charge_actual_usage_and_record_safe_metadata():
    requests = []

    def handle(request):
        requests.append(request)
        body = completion() if request.url.path.endswith("completions") else embedding()
        return httpx.Response(200, json=body)

    async def run():
        async with openai.MeteredOpenAI(KEY, transport=httpx.MockTransport(handle)) as client:
            assert await client.complete(MESSAGES) == '{"action":"verify"}'
            vectors = await client.embed(["first", "second"], api_key=KEY)
            assert vectors[0] == VECTOR
            assert vectors[1] == [0.0, 1.0] + [0.0] * 1534
            assert client.spent_tokens == 40
            assert [call["charged_tokens"] for call in client.calls] == [31, 9]
            assert [call["returned_model"] for call in client.calls] == [
                "gpt-4.1-mini-2025-04-14",
                "text-embedding-3-small",
            ]
            assert [call["usage"]["total_tokens"] for call in client.calls] == [31, 9]
            assert all(call["status"] == "ok" for call in client.calls)
            assert all(call["http_status"] == 200 for call in client.calls)
            assert all(call["latency_seconds"] >= 0 for call in client.calls)
            encoded = json.dumps(MESSAGES, sort_keys=True, separators=(",", ":")).encode()
            assert client.calls[0]["input_sha256"] == hashlib.sha256(encoded).hexdigest()
            public = json.dumps(client.calls)
            assert KEY not in public
            assert "Receipt confirmed" not in public
            assert "Authorization" not in public
            # Editing a report snapshot must not change budget accounting.
            client.calls.clear()
            assert len(client.calls) == 2

    asyncio.run(run())
    assert [str(request.url) for request in requests] == [
        "https://api.openai.com/v1/chat/completions",
        "https://api.openai.com/v1/embeddings",
    ]
    reader = json.loads(requests[0].content)
    assert reader["model"] == "gpt-4.1-mini"
    assert reader["messages"] == MESSAGES
    assert reader["max_completion_tokens"] == 128
    assert reader["temperature"] == 0
    assert reader["response_format"] == {"type": "json_object"}
    assert json.loads(requests[1].content) == {
        "model": "text-embedding-3-small",
        "input": ["first", "second"],
    }
    assert requests[0].headers["Authorization"] == f"Bearer {KEY}"


def test_call_and_token_ceilings_reject_before_transport():
    reached = []

    def handle(request):
        reached.append(request)
        return httpx.Response(200, json=completion())

    async def run():
        async with openai.MeteredOpenAI(
            KEY, max_calls=1, transport=httpx.MockTransport(handle)
        ) as client:
            await client.complete(MESSAGES)
            with pytest.raises(openai.BudgetExceeded):
                await client.embed(["first"])
            assert len(client.calls) == 1
        async with openai.MeteredOpenAI(
            KEY, max_tokens=128, transport=httpx.MockTransport(handle)
        ) as client:
            with pytest.raises(openai.BudgetExceeded):
                await client.complete(MESSAGES)
            assert client.calls == []
            assert client.spent_tokens == 0

    asyncio.run(run())
    assert len(reached) == 1


def test_concurrent_calls_reserve_tokens_before_either_can_finish():
    async def run():
        entered = asyncio.Event()
        release = asyncio.Event()
        reached = []

        async def handle(request):
            reached.append(request)
            entered.set()
            await release.wait()
            return httpx.Response(200, json=completion())

        async with openai.MeteredOpenAI(
            KEY, max_tokens=600, transport=httpx.MockTransport(handle)
        ) as client:
            first = asyncio.create_task(client.complete(MESSAGES))
            await asyncio.wait_for(entered.wait(), 1)
            try:
                with pytest.raises(openai.BudgetExceeded):
                    await asyncio.wait_for(client.complete(MESSAGES), 1)
                assert len(reached) == 1
            finally:
                release.set()
                await first
            assert client.spent_tokens == 31
            # A budget failure ends spending, even if pending calls later settle.
            with pytest.raises(openai.BudgetExceeded):
                await client.complete(MESSAGES)
            assert len(reached) == 1

    asyncio.run(run())


def test_settled_usage_releases_unused_reservation_when_budget_has_not_failed():
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=completion()))

    async def run():
        async with openai.MeteredOpenAI(KEY, max_tokens=600, transport=transport) as client:
            await client.complete(MESSAGES)
            await client.complete(MESSAGES)
            assert client.spent_tokens == 62

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["http", "network", "json"])
def test_unknown_failed_billing_stays_reserved_and_errors_never_leak(failure):
    def handle(request):
        if failure == "network":
            raise httpx.ConnectError(f"secret {KEY}", request=request)
        if failure == "http":
            return httpx.Response(429, json={"error": {"message": f"secret {KEY}"}})
        return httpx.Response(200, content=f"not json {KEY}")

    async def run():
        async with openai.MeteredOpenAI(
            KEY, max_tokens=600, transport=httpx.MockTransport(handle)
        ) as client:
            with pytest.raises(openai.EvaluationError) as caught:
                await client.complete(MESSAGES)
            assert not isinstance(caught.value, httpx.HTTPError)
            rendered = "".join(traceback.format_exception(caught.value))
            assert KEY not in rendered
            assert KEY not in json.dumps(client.calls)
            assert client.spent_tokens == client.calls[0]["reserved_tokens"]
            assert client.spent_tokens > 128
            assert client.calls[0]["status"] != "ok"
            with pytest.raises(openai.BudgetExceeded):
                await client.complete(MESSAGES)
            assert len(client.calls) == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "change",
    [
        {"model": "gpt-4.1"},
        {"model": f"substituted-{KEY}"},
        {"choices": []},
        {"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]},
        {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
    ],
)
def test_invalid_reader_results_fail_but_actual_usage_is_charged(change):
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=completion(**change)))

    async def run():
        async with openai.MeteredOpenAI(KEY, transport=transport) as client:
            with pytest.raises(openai.EvaluationError) as caught:
                await client.complete(MESSAGES)
            assert KEY not in str(caught.value)
            assert KEY not in json.dumps(client.calls)
            assert client.spent_tokens == 31
            assert client.calls[0]["usage"]["total_tokens"] == 31
            assert client.calls[0]["status"] != "ok"

    asyncio.run(run())


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"prompt_tokens": 24, "completion_tokens": -1, "total_tokens": 23},
        {"prompt_tokens": True, "completion_tokens": 7, "total_tokens": 8},
        {"prompt_tokens": 24, "completion_tokens": 7, "total_tokens": 0},
        {"prompt_tokens": 24.0, "completion_tokens": 7, "total_tokens": 31},
    ],
)
def test_unusable_usage_keeps_conservative_reservation(usage):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=completion(usage=usage))
    )

    async def run():
        async with openai.MeteredOpenAI(KEY, transport=transport) as client:
            with pytest.raises(openai.EvaluationError):
                await client.complete(MESSAGES)
            assert client.spent_tokens >= client.calls[0]["reserved_tokens"]
            assert client.spent_tokens > 31
            assert client.calls[0]["returned_model"] == "gpt-4.1-mini-2025-04-14"

    asyncio.run(run())


@pytest.mark.parametrize("problem", ["count", "index", "dimension", "nan", "number", "model"])
def test_invalid_embeddings_raise_instead_of_enabling_lexical_fallback(problem):
    body = copy.deepcopy(embedding())
    if problem == "count":
        body["data"].pop()
    elif problem == "index":
        body["data"][0]["index"] = 0
    elif problem == "dimension":
        body["data"][0]["embedding"].pop()
    elif problem == "nan":
        body["data"][0]["embedding"][0] = float("nan")
    elif problem == "number":
        body["data"][0]["embedding"][0] = True
    else:
        body["model"] = "text-embedding-3-large"
    # httpx intentionally disallows emitting NaN through its JSON encoder.
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, content=json.dumps(body).encode())
    )

    async def run():
        async with openai.MeteredOpenAI(KEY, transport=transport) as client:
            with pytest.raises(openai.EvaluationError):
                await client.embed(["first", "second"])
            assert client.spent_tokens == 9
            assert client.calls[0]["status"] != "ok"

    asyncio.run(run())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_url": "https://api.openai.com.evil.invalid"},
        {"base_url": "http://api.openai.com"},
        {"base_url": "https://api.openai.com/v1"},
        {"base_url": ""},
        {"model": "nomic-embed-text"},
    ],
)
def test_embedding_configuration_cannot_send_credentials_to_another_endpoint(kwargs):
    def handle(request):
        pytest.fail("Invalid embedding configuration reached the transport")

    async def run():
        async with openai.MeteredOpenAI(KEY, transport=httpx.MockTransport(handle)) as client:
            with pytest.raises(openai.EvaluationError):
                await client.embed(["first"], **kwargs)
            assert client.calls == []
            assert client.spent_tokens == 0

    asyncio.run(run())


def test_utf8_reservations_include_framing_and_output_before_sending():
    messages = [{"role": "user", "content": "Я" * 400}]

    def handle(request):
        pytest.fail("The UTF-8 input alone exceeds this token budget")

    async def run():
        async with openai.MeteredOpenAI(
            KEY, max_tokens=800, transport=httpx.MockTransport(handle)
        ) as client:
            with pytest.raises(openai.BudgetExceeded):
                await client.complete(messages)
            assert client.calls == []

    asyncio.run(run())


def test_actual_provider_overrun_is_fully_charged_and_stops_further_spending():
    body = completion(usage={"prompt_tokens": 1000, "completion_tokens": 7, "total_tokens": 1007})
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))

    async def run():
        async with openai.MeteredOpenAI(KEY, max_tokens=600, transport=transport) as client:
            with pytest.raises(openai.BudgetExceeded):
                await client.complete(MESSAGES)
            assert client.spent_tokens == 1007
            assert client.calls[0]["charged_tokens"] == 1007
            assert client.calls[0]["status"] == "budget_exceeded"
            with pytest.raises(openai.BudgetExceeded):
                await client.embed(["first"])
            assert len(client.calls) == 1

    asyncio.run(run())


def test_inconsistent_usage_cannot_erase_a_larger_reported_token_charge():
    body = completion(usage={"prompt_tokens": 24, "completion_tokens": 7, "total_tokens": 1007})
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))

    async def run():
        async with openai.MeteredOpenAI(KEY, max_tokens=600, transport=transport) as client:
            with pytest.raises(openai.EvaluationError):
                await client.complete(MESSAGES)
            assert client.spent_tokens == 1007
            with pytest.raises(openai.BudgetExceeded):
                await client.complete(MESSAGES)
            assert len(client.calls) == 1

    asyncio.run(run())


def test_output_overrun_is_rejected_after_its_actual_tokens_are_charged():
    body = completion(usage={"prompt_tokens": 24, "completion_tokens": 129, "total_tokens": 153})
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=body))

    async def run():
        async with openai.MeteredOpenAI(KEY, transport=transport) as client:
            with pytest.raises(openai.EvaluationError):
                await client.complete(MESSAGES)
            assert client.spent_tokens == 153

    asyncio.run(run())


def test_http_embedding_failure_raises_and_does_not_follow_redirects():
    reached = []

    def handle(request):
        reached.append(str(request.url))
        return httpx.Response(307, headers={"Location": f"https://evil.invalid/{KEY}"})

    async def run():
        async with openai.MeteredOpenAI(KEY, transport=httpx.MockTransport(handle)) as client:
            with pytest.raises(openai.EvaluationError) as caught:
                await client.embed(["first"])
            assert caught.value.__context__ is None
            assert KEY not in str(caught.value)
            assert KEY not in json.dumps(client.calls)
            assert client.spent_tokens == client.calls[0]["reserved_tokens"]

    asyncio.run(run())
    assert reached == ["https://api.openai.com/v1/embeddings"]


def test_cancelled_request_retains_its_reservation_and_has_no_secret_error_context():
    async def run():
        entered = asyncio.Event()

        async def handle(request):
            entered.set()
            await asyncio.Event().wait()

        async with openai.MeteredOpenAI(KEY, transport=httpx.MockTransport(handle)) as client:
            request = asyncio.create_task(client.complete(MESSAGES))
            await asyncio.wait_for(entered.wait(), 1)
            request.cancel(KEY)
            with pytest.raises(asyncio.CancelledError) as caught:
                await request
            assert KEY not in str(caught.value)
            assert caught.value.__context__ is None
            assert client.calls[0]["status"] == "cancelled"
            assert client.spent_tokens == client.calls[0]["reserved_tokens"]

    asyncio.run(run())


def test_empty_embeddings_and_closed_client_do_not_consume_budget():
    def handle(request):
        pytest.fail("No request should be made")

    async def run():
        client = openai.MeteredOpenAI(KEY, transport=httpx.MockTransport(handle))
        with pytest.raises(openai.EvaluationError):
            await client.complete(MESSAGES)
        async with client:
            assert await client.embed([]) == []
        with pytest.raises(openai.EvaluationError):
            await client.embed(["first"])
        assert client.calls == []
        assert client.spent_tokens == 0

    asyncio.run(run())


@pytest.mark.parametrize("ceiling", [0, -1, True, 129])
def test_reader_output_cannot_exceed_the_frozen_128_token_ceiling(ceiling):
    def handle(request):
        pytest.fail("Invalid output ceiling reached the transport")

    async def run():
        async with openai.MeteredOpenAI(KEY, transport=httpx.MockTransport(handle)) as client:
            with pytest.raises(openai.EvaluationError):
                await client.complete(MESSAGES, max_output_tokens=ceiling)
            assert client.spent_tokens == 0
            assert client.calls == []

    asyncio.run(run())


def test_structured_output_schema_is_forwarded_without_changing_observation_hash():
    schema = {
        "type": "object",
        "properties": {"basis": {"type": "array", "items": {"type": "integer", "enum": [0, 1]}}},
        "required": ["basis"],
        "additionalProperties": False,
    }
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=completion())

    async def run():
        async with openai.MeteredOpenAI(KEY, transport=httpx.MockTransport(handle)) as client:
            await client.complete(MESSAGES)
            assert await client.complete(MESSAGES, response_schema=schema) == '{"action":"verify"}'
            plain, structured = client.calls
            assert structured["input_sha256"] == plain["input_sha256"]
            assert structured["request_sha256"] != plain["request_sha256"]
            assert structured["request_sha256"] == hashlib.sha256(requests[1].content).hexdigest()
            assert client.spent_tokens == 62
            assert structured["charged_tokens"] == 31
            assert structured["reserved_tokens"] > plain["reserved_tokens"]
            assert "additionalProperties" not in json.dumps(client.calls)

    asyncio.run(run())
    plain, structured = [json.loads(request.content) for request in requests]
    assert plain["response_format"] == {"type": "json_object"}
    assert structured["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "memory_decision", "strict": True, "schema": schema},
    }
    assert structured["messages"] == MESSAGES
    assert structured["model"] == "gpt-4.1-mini"
    assert structured["max_completion_tokens"] == 128
    assert structured["temperature"] == 0


def test_schema_utf8_bytes_are_reserved_before_http_and_retained_on_unknown_billing():
    schema = {"type": "object", "description": "Я" * 100}
    schema_size = len(
        json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )
    reached = []

    def handle(request):
        reached.append(request)
        return httpx.Response(500, json={"error": {"message": KEY}})

    async def run():
        transport = httpx.MockTransport(handle)
        async with openai.MeteredOpenAI(KEY, transport=transport) as plain:
            with pytest.raises(openai.EvaluationError):
                await plain.complete(MESSAGES)
            plain_reservation = plain.spent_tokens
        exact_budget = plain_reservation + schema_size
        async with openai.MeteredOpenAI(
            KEY, max_tokens=exact_budget - 1, transport=httpx.MockTransport(handle)
        ) as too_small:
            with pytest.raises(openai.BudgetExceeded):
                await too_small.complete(MESSAGES, response_schema=schema)
            assert too_small.calls == []
            assert too_small.spent_tokens == 0
        async with openai.MeteredOpenAI(
            KEY, max_tokens=exact_budget, transport=httpx.MockTransport(handle)
        ) as exact:
            with pytest.raises(openai.EvaluationError):
                await exact.complete(MESSAGES, response_schema=schema)
            assert exact.spent_tokens == exact_budget
            assert exact.calls[0]["reserved_tokens"] == exact_budget
            assert "Я" not in json.dumps(exact.calls, ensure_ascii=False)
            assert KEY not in json.dumps(exact.calls)

    asyncio.run(run())
    assert len(reached) == 2


@pytest.mark.parametrize("schema", [{"secret": {KEY}}, {"secret": float("nan")}, [KEY]])
def test_unserializable_or_nonobject_schema_is_rejected_without_spending_or_leaking(schema):
    def handle(request):
        pytest.fail("Invalid schema reached HTTP transport")

    async def run():
        async with openai.MeteredOpenAI(KEY, transport=httpx.MockTransport(handle)) as client:
            with pytest.raises(openai.EvaluationError) as caught:
                await client.complete(MESSAGES, response_schema=schema)
            assert KEY not in "".join(traceback.format_exception(caught.value))
            assert client.calls == []
            assert client.spent_tokens == 0

    asyncio.run(run())
