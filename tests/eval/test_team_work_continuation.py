import asyncio

import httpx
import pytest

from tests.eval.team_decisions.openai import BudgetExceeded, EvaluationError, MeteredOpenAI
from tests.eval.team_work.experiment import healthy
from tests.eval.team_work.provider import PrefixClient, RecordingClient, ReplayClient
from tests.eval.test_team_decision_openai import KEY, MESSAGES, completion


@pytest.mark.parametrize("partial", [False, True])
def test_one_transport_retry_is_durable_charged_and_exactly_replayable(partial):
    async def run():
        reached = []

        def handle(request):
            reached.append(request.content)
            if len(reached) == 1:
                raise httpx.ReadTimeout("private transport diagnostics")
            body = completion()
            if partial:
                body["choices"][0]["finish_reason"] = "length"
            return httpx.Response(200, json=body)

        provider = MeteredOpenAI(KEY, transport=httpx.MockTransport(handle))
        updates = []
        client = RecordingClient(
            provider, on_update=updates.append, retry_transport=True, allow_truncated=partial
        )
        async with client:
            raw = await client.complete(MESSAGES, max_output_tokens=128)
        source = client.snapshot()
        assert len(reached) == 2 and reached[0] == reached[1]
        assert source["spent_tokens"] == provider.calls[0]["reserved_tokens"] + 31
        assert len(updates) == 4 and healthy(source)
        with pytest.raises(ValueError):
            ReplayClient(source)
        replay = ReplayClient(source, allow_transport_failure=True, allow_truncated=partial)
        reader = RecordingClient(
            replay, on_update=lambda s: None, retry_transport=True, allow_truncated=partial
        )
        async with reader:
            assert await reader.complete(MESSAGES, max_output_tokens=128) == raw
        replay.assert_consumed()
        assert reader.spent_tokens == client.spent_tokens
        assert reader.actual_provider_calls == 0 and healthy(reader.snapshot())

    asyncio.run(run())


@pytest.mark.parametrize("remaining_tokens", [20000, 1])
def test_failed_prefix_is_reused_before_new_io_and_reduces_live_authority(remaining_tokens):
    async def run():
        failed = RecordingClient(
            MeteredOpenAI(
                KEY,
                transport=httpx.MockTransport(
                    lambda request: (_ for _ in ()).throw(httpx.ReadTimeout("private"))
                ),
            ),
            on_update=lambda s: None,
        )
        async with failed:
            with pytest.raises(EvaluationError):
                await failed.complete(MESSAGES, max_output_tokens=128)
        original = failed.snapshot()
        authority = []
        reached = []

        def factory(**limits):
            authority.append(limits)

            def handle(request):
                reached.append(request)
                return httpx.Response(200, json=completion())

            return MeteredOpenAI(KEY, transport=httpx.MockTransport(handle), **limits)

        charged = original["spent_tokens"]
        prefix = PrefixClient(
            original,
            factory,
            max_calls=3,
            max_tokens=charged + remaining_tokens,
            max_output_tokens_per_call=128,
        )
        client = RecordingClient(prefix, on_update=lambda s: None, retry_transport=True)
        async with client:
            if remaining_tokens == 1:
                with pytest.raises(BudgetExceeded):
                    await client.complete(MESSAGES, max_output_tokens=128)
            else:
                assert await client.complete(MESSAGES, max_output_tokens=128)
        assert authority == [
            dict(max_calls=2, max_tokens=remaining_tokens, max_output_tokens_per_call=128)
        ]
        assert client.spent_tokens == charged + (31 if remaining_tokens > 1 else 0)
        assert client.actual_provider_calls == (1 if remaining_tokens > 1 else 0)
        assert len(reached) == client.actual_provider_calls

    asyncio.run(run())


def test_prefix_drift_never_falls_back_to_paid_transport():
    async def run():
        original = RecordingClient(
            MeteredOpenAI(
                KEY,
                transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, json=completion())
                ),
            ),
            on_update=lambda s: None,
        )
        async with original:
            await original.complete(MESSAGES, max_output_tokens=128)
        reached = []

        def forbidden(**limits):
            reached.append(limits)
            raise AssertionError("Paid fallback on drift")

        prefix = PrefixClient(
            original.snapshot(),
            forbidden,
            max_calls=3,
            max_tokens=20000,
            max_output_tokens_per_call=128,
        )
        with pytest.raises(ValueError):
            async with prefix:
                await prefix.complete(
                    [{"role": "user", "content": "changed"}], max_output_tokens=128
                )
        assert reached == []

    asyncio.run(run())


def test_truncation_is_a_recorded_invalid_answer_not_an_unbounded_retry():
    async def run():
        def handle(request):
            return httpx.Response(
                200,
                json=completion(
                    choices=[{"message": {"content": "partial source"}, "finish_reason": "length"}]
                ),
            )

        client = RecordingClient(
            MeteredOpenAI(KEY, transport=httpx.MockTransport(handle)),
            on_update=lambda s: None,
            retry_transport=True,
            allow_truncated=True,
        )
        async with client:
            assert await client.complete(MESSAGES, max_output_tokens=128) == "partial source"
        assert client.completion_truncated and healthy(client.snapshot())
        assert len(client.calls) == 1
        with pytest.raises(ValueError):
            ReplayClient(client.snapshot())
        replay = ReplayClient(client.snapshot(), allow_truncated=True)
        recorded = RecordingClient(replay, on_update=lambda s: None, allow_truncated=True)
        async with recorded:
            assert await recorded.complete(MESSAGES, max_output_tokens=128) == "partial source"
        replay.assert_consumed()
        assert recorded.completion_truncated and recorded.actual_provider_calls == 0

    asyncio.run(run())


def test_zero_charge_budget_refusal_can_be_replayed_without_provider_io():
    async def run():
        provider = MeteredOpenAI(
            KEY, max_calls=0, transport=httpx.MockTransport(lambda request: pytest.fail("I/O"))
        )
        client = RecordingClient(provider, on_update=lambda s: None)
        async with client:
            with pytest.raises(BudgetExceeded):
                await client.complete(MESSAGES, max_output_tokens=128)
        source = client.snapshot()
        with pytest.raises(ValueError):
            ReplayClient(source)
        replay = ReplayClient(source, allow_budget_refusal=True)
        with pytest.raises(BudgetExceeded):
            await replay.complete(MESSAGES, max_output_tokens=128)
        replay.assert_consumed()
        assert replay.spent_tokens == replay.actual_provider_calls == 0

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["length", "http", "twice"])
def test_retry_never_hides_truncation_http_errors_or_repeats_indefinitely(failure):
    async def run():
        reached = []

        def handle(request):
            reached.append(request)
            if failure == "twice":
                raise httpx.ReadTimeout("private")
            if failure == "http":
                return httpx.Response(503)
            return httpx.Response(
                200,
                json=completion(
                    choices=[{"message": {"content": "partial"}, "finish_reason": "length"}]
                ),
            )

        client = RecordingClient(
            MeteredOpenAI(KEY, transport=httpx.MockTransport(handle)),
            on_update=lambda s: None,
            retry_transport=True,
        )
        async with client:
            with pytest.raises(EvaluationError):
                await client.complete(MESSAGES, max_output_tokens=128)
        assert len(reached) == (2 if failure == "twice" else 1)
        assert not healthy(client.snapshot())

    asyncio.run(run())
