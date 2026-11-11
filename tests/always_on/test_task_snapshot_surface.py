"""The MCP task snapshot uses the same strict configured-owner boundary as HTTP."""

import asyncio
import json

from ai_knot.mcp_server import _make_server


def test_mcp_task_snapshot_is_queryless_strict_and_all_or_error(ao_factory, kb):
    memory = ao_factory()
    fact = kb.add("Backup receipt verified")
    app = _make_server(kb, always_on=memory)

    async def call(arguments):
        return json.loads(await app._tool_manager.call_tool("memory_context_snapshot", arguments))

    async def workflow():
        tool = app._tool_manager.get_tool("memory_context_snapshot")
        assert tool is not None and tool.parameters["required"] == ["task_key"]
        result = await call({"task_key": "restore", "max_facts": 1})
        assert len(result) == 1 and result[0]["content"] == fact.content
        assert result[0]["ref"]["namespace"] == "owner"
        assert result[0]["ref"]["fact_id"] == fact.id
        memory.derive_fact(
            "Derived handoff " * 100,
            basis=memory.capture_evidence((("owner", fact.id),)),
            task_key="restore",
        )
        source_only = await call(
            {
                "task_key": "restore",
                "max_facts": 1,
                "max_chars": len(fact.content),
                "include_derived": False,
            }
        )
        assert source_only == result
        assert (await call({"task_key": "restore", "max_facts": 1}))["error"][
            "code"
        ] == "SnapshotTooLarge"
        for invalid in (
            {},
            {"task_key": False},
            {"task_key": "restore", "max_facts": "1"},
            {"task_key": "restore", "max_chars": True},
            {"task_key": "restore", "owner_agent_id": "foreign"},
            {"task_key": "restore", "requirements": []},
            {"task_key": "restore", "include_derived": "false"},
            {"task_key": "restore", "include_derived": 0},
            {"task_key": "restore", "include_derived": None},
        ):
            assert (await call(invalid))["error"]["code"] == "InvalidInput"
        kb.add("Another complete observation")
        overflow = await call({"task_key": "restore", "max_facts": 1})
        assert overflow["error"]["code"] == "SnapshotTooLarge"
        assert "facts" not in overflow and fact.content not in json.dumps(overflow)

    asyncio.run(workflow())
