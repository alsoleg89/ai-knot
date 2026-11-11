"""Local fake-provider checks; never opens a provider transport."""

import asyncio
import copy
import json

import pytest

from tests.eval.team_work.provider import RecordingClient, ReplayClient


class FakeProvider:
    def __init__(self, *, failure=None, known_charge=False, budget_refusal=False, pause=None):
        self.calls = []
        self.spent_tokens = 0
        self.failure = failure
        self.known_charge = known_charge
        self.budget_refusal = budget_refusal
        self.pause = pause
        self.opened = False
        self.inside = 0
        self.max_inside = 0

    async def __aenter__(self):
        self.opened = True
        return self

    async def __aexit__(self, *args):
        self.opened = False

    async def complete(self, messages, *, max_output_tokens, response_schema=None):
        return await self._call("complete", messages, response_schema, max_output_tokens)

    async def embed(self, texts):
        if not texts:
            return []
        return await self._call("embed", texts, None, 0)

    async def _call(self, kind, inputs, schema, limit):
        assert self.opened
        if self.budget_refusal:
            raise RuntimeError("secret refusal message")
        reservation = (
            len(
                json.dumps(
                    inputs, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            )
            + 256
            + limit
        )
        if schema is not None:
            reservation += len(
                json.dumps(
                    schema, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode()
            )
        call = {
            "kind": kind,
            "status": "pending",
            "reserved_tokens": reservation,
            "charged_tokens": reservation,
            "usage": None,
            "latency_seconds": 0.25,
        }
        self.calls.append(call)
        self.spent_tokens += reservation
        self.inside += 1
        self.max_inside = max(self.max_inside, self.inside)
        try:
            if self.pause:
                await self.pause()
            else:
                await asyncio.sleep(0)
            if self.failure:
                if self.known_charge:
                    self.spent_tokens += 7 - reservation
                    call.update(charged_tokens=7, usage={"total_tokens": 7})
                raise self.failure
            self.spent_tokens += 7 - reservation
            call.update(status="ok", charged_tokens=7, usage={"total_tokens": 7})
            if kind == "complete":
                return ' \n{"текст":"é"}\r\n '
            return [[-0.0, 0.123456789] for _ in inputs]
        except asyncio.CancelledError:
            call["status"] = "cancelled"
            raise
        except Exception:
            call["status"] = "error"
            raise
        finally:
            self.inside -= 1


MESSAGES = [{"role": "user", "content": "Я"}]
SCHEMA = {"type": "object"}


def test_persists_exact_input_before_io_and_raw_output_afterward():
    async def check():
        provider = FakeProvider()
        updates = []

        def persist(snapshot):
            updates.append(copy.deepcopy(snapshot))
            snapshot["requests"].clear()  # Callbacks cannot mutate the recorder.

        client = RecordingClient(provider, on_update=persist)
        async with client:
            raw = await client.complete(MESSAGES, max_output_tokens=8, response_schema=SCHEMA)
            vectors = await client.embed(["Я", "Я"])
            assert provider.opened
        assert not provider.opened
        pending = updates[0]
        assert pending["requests"][0]["inputs"] == MESSAGES
        assert pending["requests"][0]["response_schema"] == SCHEMA
        assert pending["requests"][0]["max_output_tokens"] == 8
        # UTF-8 canonical messages: 32 bytes; schema: 17; frame: 256; output: 8.
        assert pending["requests"][0]["reserved_tokens"] == 313
        assert pending["accounted_tokens"] == 313
        assert pending["spent_tokens"] == 0
        assert pending["meter_calls"] == []
        final = client.snapshot()
        assert final["requests"][0]["raw_output"].encode() == raw.encode()
        assert final["requests"][1]["vectors"] == vectors
        assert final["spent_tokens"] == final["accounted_tokens"] == 14
        assert final["actual_provider_calls"] == 2
        assert client.calls == provider.calls
        final["requests"].clear()
        assert len(client.snapshot()["requests"]) == 2
        json.dumps(client.snapshot(), allow_nan=False)

    asyncio.run(check())


def test_serializes_duplicates_and_preserves_order_and_original_usage_in_replay():
    async def check():
        provider = FakeProvider()
        client = RecordingClient(provider, on_update=lambda value: None)
        async with client:
            outputs = await asyncio.gather(
                client.complete(MESSAGES, max_output_tokens=8, response_schema=SCHEMA),
                client.complete(MESSAGES, max_output_tokens=8, response_schema=SCHEMA),
                client.embed(["duplicate", "duplicate"]),
            )
        assert provider.max_inside == 1
        source = client.snapshot()
        assert [r["meter_call_indexes"] for r in source["requests"]] == [[0], [1], [2]]
        replay = ReplayClient(source)
        source["requests"].clear()  # A caller cannot mutate replay's source.
        updates = []
        outer = RecordingClient(replay, on_update=updates.append)
        async with outer:
            assert (
                await outer.complete(MESSAGES, max_output_tokens=8, response_schema=SCHEMA)
                == outputs[0]
            )
            assert outer.spent_tokens == 7
            assert len(outer.calls) == 1
            assert (
                await outer.complete(MESSAGES, max_output_tokens=8, response_schema=SCHEMA)
                == outputs[1]
            )
            vectors = await outer.embed(["duplicate", "duplicate"])
            assert vectors == outputs[2]
            vectors[0][0] = 9
        replay.assert_consumed()
        assert replay.spent_tokens == outer.spent_tokens == 21
        assert outer.snapshot()["actual_provider_calls"] == 0
        assert replay.snapshot()["actual_provider_calls"] == 0
        assert replay.calls == client.calls
        assert replay.snapshot()["requests"][2]["vectors"][0][0] == -0.0

    asyncio.run(check())


@pytest.mark.parametrize("mutation", ["input", "schema", "limit", "kind", "extra", "missing"])
def test_replay_fails_closed_for_any_request_drift(mutation):
    async def check():
        client = RecordingClient(FakeProvider(), on_update=lambda value: None)
        async with client:
            await client.complete(MESSAGES, max_output_tokens=8, response_schema=SCHEMA)
        replay = ReplayClient(client.snapshot())
        async with replay:
            if mutation == "missing":
                with pytest.raises(ValueError, match="missing"):
                    replay.assert_consumed()
                return
            if mutation == "extra":
                await replay.complete(MESSAGES, max_output_tokens=8, response_schema=SCHEMA)
            with pytest.raises(ValueError):
                if mutation == "kind":
                    await replay.embed(["Я"])
                else:
                    await replay.complete(
                        [{"role": "user", "content": "Я "}] if mutation == "input" else MESSAGES,
                        max_output_tokens=9 if mutation == "limit" else 8,
                        response_schema={"type": "string"} if mutation == "schema" else SCHEMA,
                    )
            assert replay.spent_tokens == (7 if mutation == "extra" else 0)
            assert len(replay.calls) == (1 if mutation == "extra" else 0)

    asyncio.run(check())


def test_boundary_and_persistence_failures_never_enter_provider():
    async def check():
        provider = FakeProvider()
        client = RecordingClient(provider, on_update=lambda value: None, boundary=lambda: False)
        async with client:
            with pytest.raises(RuntimeError, match="SQL"):
                await client.embed(["text"])
        assert client.snapshot()["requests"] == []
        assert client.calls == []

        closed = True
        updates = []

        def persist(snapshot):
            nonlocal closed
            updates.append(snapshot)
            closed = False

        client = RecordingClient(provider, on_update=persist, boundary=lambda: closed)
        async with client:
            with pytest.raises(RuntimeError, match="SQL"):
                await client.embed(["text"])
        assert updates[0]["accounted_tokens"] > 0
        assert client.snapshot()["accounted_tokens"] == 0
        assert client.calls == []

        def failed_persist(snapshot):
            raise OSError("write failed")

        client = RecordingClient(provider, on_update=failed_persist)
        async with client:
            with pytest.raises(OSError):
                await client.embed(["text"])
        assert client.calls == []

    asyncio.run(check())


@pytest.mark.parametrize("mode", ["http", "known_http", "cancel", "budget"])
def test_failed_outcomes_keep_authoritative_unknown_charges_and_no_exception_messages(mode):
    async def check():
        started = asyncio.Event()

        async def pause():
            started.set()
            await asyncio.Event().wait()

        provider = FakeProvider(
            failure=RuntimeError("credential SECRET http details") if "http" in mode else None,
            known_charge=mode == "known_http",
            budget_refusal=mode == "budget",
            pause=pause if mode == "cancel" else None,
        )
        updates = []
        client = RecordingClient(provider, on_update=updates.append)
        async with client:
            if mode == "cancel":
                task = asyncio.create_task(client.embed(["text"]))
                await started.wait()
                # Live snapshot uses the meter reservation once entered, never twice.
                assert client.snapshot()["accounted_tokens"] == provider.spent_tokens
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                with pytest.raises(RuntimeError):
                    await client.embed(["text"])
        snapshot = client.snapshot()
        assert snapshot["spent_tokens"] == snapshot["accounted_tokens"] == provider.spent_tokens
        assert snapshot["actual_provider_calls"] == (0 if mode == "budget" else 1)
        assert snapshot["spent_tokens"] == {"budget": 0, "known_http": 7}.get(mode, 264)
        assert updates[0]["accounted_tokens"] == 264
        assert "SECRET" not in json.dumps(snapshot)
        assert "secret refusal" not in json.dumps(snapshot)
        assert updates[-1] == snapshot
        with pytest.raises(ValueError):
            ReplayClient(snapshot)
        with pytest.raises(ValueError):
            ReplayClient(updates[0])

    asyncio.run(check())


def test_replay_rejects_corrupt_meter_coverage_and_accounting():
    async def check():
        client = RecordingClient(FakeProvider(), on_update=lambda value: None)
        async with client:
            await client.embed(["text"])
        source = client.snapshot()
        for key, value in [
            ("status", "pending"),
            ("meter_call_indexes", []),
            ("reserved_tokens", 0),
            ("meter_start", 1),
        ]:
            corrupt = copy.deepcopy(source)
            corrupt["requests"][0][key] = value
            with pytest.raises(ValueError):
                ReplayClient(corrupt)
        for key, value in [("status", "error"), ("charged_tokens", -1), ("kind", "complete")]:
            corrupt = copy.deepcopy(source)
            corrupt["meter_calls"][0][key] = value
            with pytest.raises(ValueError):
                ReplayClient(corrupt)
        source["spent_tokens"] += 1
        with pytest.raises(ValueError):
            ReplayClient(source)

    asyncio.run(check())


def test_empty_embedding_and_later_local_parse_failure_are_replayable():
    class MalformedProvider(FakeProvider):
        async def complete(self, messages, *, max_output_tokens, response_schema=None):
            await super().complete(
                messages, max_output_tokens=max_output_tokens, response_schema=response_schema
            )
            return " \r\n {malformed é Я"

    async def check():
        client = RecordingClient(MalformedProvider(), on_update=lambda value: None)
        async with client:
            assert await client.embed([]) == []
            raw = await client.complete(MESSAGES, max_output_tokens=8)
            with pytest.raises(json.JSONDecodeError):
                json.loads(raw)
        replay = ReplayClient(client.snapshot())
        async with replay:
            assert await replay.embed([]) == []
            assert replay.spent_tokens == 0
            replayed = await replay.complete(MESSAGES, max_output_tokens=8)
            assert replayed.encode() == raw.encode()
            with pytest.raises(json.JSONDecodeError):
                json.loads(replayed)
        replay.assert_consumed()
        assert replay.spent_tokens == 7

    asyncio.run(check())
