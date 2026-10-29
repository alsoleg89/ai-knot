"""The optional guard uses the same strict configured-owner transport codec."""

import asyncio
import json
from dataclasses import asdict

from fastapi.testclient import TestClient

from ai_knot.always_on import IntentState
from ai_knot.mcp_server import _make_server
from ai_knot.server.app import create_app


def test_http_guard_capture_and_guarded_writes_use_strict_owner_contract(ao_factory, kb):
    memory = ao_factory()
    kb.add("A sourced private observation with enough unique text to test redaction")
    client = TestClient(create_app(kb, token="secret", always_on=memory))
    path = "/v1/always-on/evidence/guard"
    headers = {"Authorization": "Bearer secret"}
    assert client.post(path, json={"task_key": "rollout"}).status_code == 401
    for bad in (
        {},
        {"task_key": False},
        {"task_key": "rollout", "max_facts": True},
        {"task_key": "rollout", "max_facts": "1"},
        {"task_key": "rollout", "owner_agent_id": "other"},
    ):
        assert client.post(path, headers=headers, json=bad).status_code == 422
    result = client.post(path, headers=headers, json={"task_key": "rollout", "max_facts": 2})
    assert result.status_code == 200, result.text
    guard = result.json()
    assert set(guard) == {"task_key", "max_facts", "digest"} and len(guard["digest"]) == 64
    intent = memory.create_intent(client_key="one", goal="Plan")
    lease = memory.claim(intent.id, worker_id="host")
    body = {
        "lease_token": lease.token,
        "expected_revision": memory.get_intent(intent.id).revision,
        "state": "ready",
        "progress": {},
        "context_guard": guard,
    }
    kb.add("A separate unselected blocker")
    checkpoint = f"/v1/always-on/intents/{intent.id}/checkpoint"
    response = client.post(checkpoint, headers=headers, json=body)
    assert response.status_code == 409 and response.json()["error"]["code"] == "StaleEvidence"
    for bad in (
        {**guard, "unexpected": True},
        {**guard, "max_facts": True},
        {**guard, "digest": "INVALID"},
    ):
        assert (
            client.post(
                checkpoint, headers=headers, json={**body, "context_guard": bad}
            ).status_code
            == 422
        )
    fresh = client.post(path, headers=headers, json={"task_key": "rollout", "max_facts": 2}).json()
    assert (
        client.post(checkpoint, headers=headers, json={**body, "context_guard": fresh}).status_code
        == 200
    )
    lease = memory.claim(intent.id, worker_id="executor")
    action = memory.prepare_action(
        intent.id, lease_token=lease.token, action_key="work", request={}
    )
    kb.add("Another newly applicable condition")
    response = client.post(
        f"/v1/always-on/actions/{action.id}/begin",
        headers=headers,
        json={"lease_token": lease.token, "context_guard": fresh},
    )
    assert response.status_code == 409 and response.json()["error"]["code"] == "StaleEvidence"
    assert memory.resume(intent.id).intent.state == IntentState.BLOCKED
    assert (
        client.post(path, headers=headers, json={"task_key": "rollout", "max_facts": 2}).status_code
        == 413
    )


def test_mcp_guard_is_strict_and_preserves_nested_value(ao_factory, kb):
    memory = ao_factory()
    kb.add("A current observation")
    app = _make_server(kb, always_on=memory)

    async def call(name, arguments):
        return json.loads(await app._tool_manager.call_tool(name, arguments))

    async def workflow():
        tool = app._tool_manager.get_tool("memory_context_guard")
        assert tool is not None and tool.parameters["required"] == ["task_key"]
        guard = await call("memory_context_guard", {"task_key": "rollout"})
        assert guard == asdict(memory.task_guard(task_key="rollout"))
        assert (await call("memory_context_guard", {"task_key": "rollout", "max_facts": False}))[
            "error"
        ]["code"] == "InvalidInput"
        intent = memory.create_intent(client_key="one", goal="Plan")
        lease = memory.claim(intent.id, worker_id="host")
        payload = {
            "intent_id": intent.id,
            "lease_token": lease.token,
            "expected_revision": memory.get_intent(intent.id).revision,
            "state": "ready",
            "progress": {},
            "context_guard": guard,
        }
        assert (
            await call(
                "intent_checkpoint", {**payload, "context_guard": {**guard, "owner": "other"}}
            )
        )["error"]["code"] == "InvalidInput"
        assert (await call("intent_checkpoint", payload))["state"] == "ready"

    asyncio.run(workflow())
