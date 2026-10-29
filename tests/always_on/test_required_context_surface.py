import asyncio
import json
from dataclasses import asdict, replace

from fastapi.testclient import TestClient
from tests.always_on.test_semantic_context import setup

from ai_knot.mcp_server import _make_server
from ai_knot.server.app import create_app


def test_required_refs_cross_strict_http_and_mcp_boundary(sql_storage, clock):
    kb, memory, facts = setup(sql_storage, clock)
    refs = memory.capture_evidence((("owner", "coffee"), ("owner", "office")))
    body = {"requirements": ["backup"], "max_facts": 2, "required_refs": [asdict(r) for r in refs]}
    client = TestClient(create_app(kb, token="secret", always_on=memory))
    app = _make_server(kb, always_on=memory)

    def http(value):
        return client.post(
            "/v1/always-on/evidence/select", json=value, headers={"Authorization": "Bearer secret"}
        )

    async def mcp(value):
        return json.loads(await app._tool_manager.call_tool("memory_context_select", value))

    result = http(body)
    assert result.status_code == 200
    assert [f["ref"]["fact_id"] for f in result.json()["facts"]] == ["coffee", "office"]
    assert result.json()["missing_requirements"] == ["backup"]
    assert asyncio.run(mcp(body)) == result.json()
    for bad in (
        False,
        "coffee",
        [{**asdict(refs[0]), "revision": True}],
        [{**asdict(refs[0]), "extra": "invalid"}],
    ):
        assert http({**body, "required_refs": bad}).status_code == 422
        assert asyncio.run(mcp({**body, "required_refs": bad}))["error"]["code"] == "InvalidInput"
    assert http({**body, "max_facts": 1}).status_code == 413
    assert asyncio.run(mcp({**body, "max_facts": 1}))["error"]["code"] == "SnapshotTooLarge"
    kb.replace_facts([facts[0], replace(facts[1], content="Changed supply record"), facts[2]])
    assert http(body).status_code == 409
    assert asyncio.run(mcp(body))["error"]["code"] == "StaleEvidence"
