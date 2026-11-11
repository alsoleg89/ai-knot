import asyncio
import contextlib
import json
import os
import queue
import subprocess
import sys
import threading

from ai_knot.mcp_server import _make_server
from ai_knot.server.always_on import ROUTES


def test_mcp_strict_owner_and_capabilities(ao_factory, kb):
    app = _make_server(kb, always_on=ao_factory())

    async def calls():
        result = await app._tool_manager.call_tool(
            "intent_create", {"client_key": "k", "goal": "g", "owner_agent_id": "foreign"}
        )
        assert json.loads(result)["error"]["code"] == "InvalidInput"
        entries = json.loads(await app._tool_manager.call_tool("capabilities", {}))
        assert {"intent_create", "memory_maintenance", "memory_context_guard", "recall"} <= {
            entry["name"] for entry in entries
        }
        intent = json.loads(
            await app._tool_manager.call_tool("intent_create", {"client_key": "k", "goal": "g"})
        )
        assert (
            json.loads(await app._tool_manager.call_tool("intent_get", {"intent_id": intent["id"]}))
            == intent
        )

    asyncio.run(calls())
    assert len(ROUTES) == 26
    assert ("POST", "/evidence/guard", "task_guard", None) in ROUTES


def test_mcp_experience_records_exact_outcome_and_rejects_spoofing(ao_factory, kb):
    from dataclasses import asdict

    from tests.always_on.test_team_experience import completed

    memory = ao_factory()
    basis = memory.capture_evidence((("owner", kb.add("Receipt").id),))
    action, _ = completed(memory, "failed")
    app = _make_server(kb, always_on=memory)

    async def workflow():
        body = {
            "action_id": action.id,
            "expected_action_revision": action.revision,
            "experience_key": "one",
            "content": "Restore warning",
            "basis": [asdict(ref) for ref in basis],
        }
        result = json.loads(await app._tool_manager.call_tool("action_experience_record", body))
        assert result["outcome"] == "failed" and result["lesson"]["ref"]["namespace"] == "owner"
        for extra in ({"expected_action_revision": "3"}, {"owner_agent_id": "peer"}):
            invalid = json.loads(
                await app._tool_manager.call_tool("action_experience_record", {**body, **extra})
            )
            assert invalid["error"]["code"] == "InvalidInput"

    asyncio.run(workflow())


def test_mcp_recovery_and_context_review_use_real_typed_boundary(ao_factory, kb):
    app = _make_server(kb, always_on=ao_factory())
    fact = kb.add("Persisted team decision")

    async def call(name, args):
        return json.loads(await app._tool_manager.call_tool(name, args))

    async def workflow():
        await call(
            "memory_subscribe", {"subscription_id": "task", "selector": {"namespace": "owner"}}
        )
        snapshot = await call("memory_resync", {"subscription_id": "task"})
        assert snapshot["is_snapshot"] and snapshot["events"][0]["fact_id"] == fact.id
        ref = {
            "namespace": "owner",
            "fact_id": fact.id,
            "revision": snapshot["events"][0]["after_revision"],
        }
        review = await call("memory_context_review", {"refs": [ref], "max_chars": 100})
        assert review["complete"] and review["accepted"][0]["content"] == "Persisted team decision"
        invalid = await call("memory_context_review", {"refs": [ref], "max_chars": True})
        assert invalid["error"]["code"] == "InvalidInput"
        await call(
            "memory_ack", {"subscription_id": "task", "delivery_id": snapshot["delivery_id"]}
        )
        assert (await call("memory_poll", {"subscription_id": "task"}))["delivery_id"] is None

    asyncio.run(workflow())


def test_mcp_task_selection_keeps_requirements_separate_from_role(ao_factory, kb):
    app = _make_server(kb, always_on=ao_factory())
    kb.add("Backup restoration verified")

    async def workflow():
        view = json.loads(
            await app._tool_manager.call_tool(
                "memory_context_select",
                {
                    "requirements": ["backup restoration", "rollback checkpoint"],
                    "role": "operator",
                },
            )
        )
        assert view["covered_requirements"] == ["backup restoration"]
        assert view["missing_requirements"] == ["rollback checkpoint"]
        invalid = json.loads(
            await app._tool_manager.call_tool(
                "memory_context_select",
                {
                    "requirements": "backup restoration",
                },
            )
        )
        assert invalid["error"]["code"] == "InvalidInput"

    asyncio.run(workflow())


class Session:
    def __init__(self, path):
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "ai_knot.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={
                **os.environ,
                "AI_KNOT_STORAGE": "sqlite",
                "AI_KNOT_DB_PATH": str(path),
                "AI_KNOT_AGENT_ID": "owner",
                "AI_KNOT_ALWAYS_ON": "true",
                "AI_KNOT_EMBED_URL": "",
            },
            text=True,
        )
        self.responses = queue.Queue()
        self.identifier = 0

        def read():
            for line in self.proc.stdout:
                with contextlib.suppress(ValueError):
                    self.responses.put(json.loads(line))

        threading.Thread(target=read, daemon=True).start()
        self.request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        )
        self.proc.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        )
        self.proc.stdin.flush()

    def request(self, method, params):
        self.identifier += 1
        self.proc.stdin.write(
            json.dumps(
                {"jsonrpc": "2.0", "id": self.identifier, "method": method, "params": params}
            )
            + "\n"
        )
        self.proc.stdin.flush()
        while True:
            response = self.responses.get(timeout=10)
            if response.get("id") == self.identifier:
                assert "error" not in response, response
                return response["result"]

    def tool(self, name, args):
        result = self.request("tools/call", {"name": name, "arguments": args})
        return json.loads("".join(item.get("text", "") for item in result["content"]))

    def close(self):
        self.proc.kill()
        self.proc.wait(timeout=5)


def test_always_on_restart_replays_delivery(tmp_path):
    from ai_knot.knowledge import KnowledgeBase
    from ai_knot.storage.sqlite_storage import SQLiteStorage

    path = tmp_path / "mcp.db"
    first = Session(path)
    try:
        first.tool("memory_subscribe", {"subscription_id": "w", "selector": {"namespace": "owner"}})
        KnowledgeBase("owner", storage=SQLiteStorage(str(path)), embed_url="").add("Changed memory")
        batch = first.tool("memory_poll", {"subscription_id": "w"})
        assert batch["delivery_id"] and len(batch["events"]) == 1
    finally:
        first.close()
    second = Session(path)
    try:
        assert second.tool("memory_poll", {"subscription_id": "w"}) == batch
        assert (
            second.tool("memory_ack", {"subscription_id": "w", "delivery_id": batch["delivery_id"]})
            is None
        )
        assert second.tool("memory_poll", {"subscription_id": "w"})["events"] == []
    finally:
        second.close()
