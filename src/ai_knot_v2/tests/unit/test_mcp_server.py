"""Sprint 22 — MCP server smoke tests (no live MCP transport)."""

from __future__ import annotations

import importlib


class TestMCPServerModule:
    def test_importable(self) -> None:
        mod = importlib.import_module("ai_knot_v2.api.mcp_server")
        assert hasattr(mod, "mcp")
        assert hasattr(mod, "main")

    def test_tools_registered(self) -> None:
        mod = importlib.import_module("ai_knot_v2.api.mcp_server")
        # FastMCP stores registered tools; check the functions exist
        for fn_name in ("learn", "recall", "explain", "trace", "inspect_memory", "health"):
            assert hasattr(mod, fn_name), f"missing tool function: {fn_name}"

    def test_health_returns_ok(self) -> None:
        mod = importlib.import_module("ai_knot_v2.api.mcp_server")
        result = mod.health()
        assert result["status"] == "ok"
        assert isinstance(result["total_atoms"], int)

    def test_learn_and_recall_roundtrip(self) -> None:
        """learn → recall returns at least one atom mentioning the fact."""
        mod = importlib.import_module("ai_knot_v2.api.mcp_server")
        # Use a fresh in-memory API to avoid cross-test state
        from ai_knot_v2.api.product import MemoryAPI

        fresh_api = MemoryAPI(db_path=":memory:")
        original_api = mod._api
        mod._api = fresh_api

        try:
            resp = mod.learn([{"text": "Alice is allergic to penicillin.", "speaker": "user"}])
            assert len(resp["atom_ids"]) >= 1

            recall_resp = mod.recall("Does Alice have any drug allergies?", max_atoms=50)
            texts = [
                " ".join(
                    filter(
                        None,
                        [a.get("object_value"), a.get("subject"), a.get("predicate", "")],
                    )
                ).lower()
                for a in recall_resp["atoms"]
            ]
            assert any("penicillin" in t for t in texts)
        finally:
            mod._api = original_api
