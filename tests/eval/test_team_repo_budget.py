import asyncio
import json

import httpx
import pytest

from tests.eval.team_decisions.openai import BudgetExceeded, EvaluationError, MeteredOpenAI


def test_larger_outputs_are_opt_in_and_fully_reserved():
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "gpt-4.1-mini-2025-04-14",
                "choices": [{"finish_reason": "stop", "message": {"content": "code"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            },
        )

    async def run():
        messages = [{"role": "user", "content": "Implement the documented function"}]
        wire = httpx.MockTransport(respond)
        async with MeteredOpenAI("offline", transport=wire) as old:
            with pytest.raises(EvaluationError):
                await old.complete(messages, max_output_tokens=129)
        async with MeteredOpenAI(
            "offline", max_output_tokens_per_call=2048, transport=wire
        ) as coding:
            assert await coding.complete(messages, max_output_tokens=2048) == "code"
            assert coding.calls[0]["reserved_tokens"] >= 2048
            assert coding.spent_tokens == 30
        async with MeteredOpenAI(
            "offline", max_tokens=2047, max_output_tokens_per_call=2048, transport=wire
        ) as small:
            with pytest.raises(BudgetExceeded):
                await small.complete(messages, max_output_tokens=2048)
            assert small.calls == []

    asyncio.run(run())
    assert len(requests) == 1 and requests[0]["max_completion_tokens"] == 2048


@pytest.mark.parametrize("limit", [0, 4097, True, 1.5])
def test_output_authority_is_bounded(limit):
    with pytest.raises(EvaluationError):
        MeteredOpenAI("offline", max_output_tokens_per_call=limit)
