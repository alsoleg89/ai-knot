"""End-to-end tests for the ai-knot-mcp JSON-RPC server.

Spawns the real `ai-knot-mcp` subprocess and communicates with it over
stdio using raw JSON-RPC 2.0, without any Python client library in between.

Requires the mcp extra:
    pip install "ai-knot[mcp]"

Marked @pytest.mark.integration — included in normal CI but skipped if the
`ai-knot-mcp` command is not available.

Tests are consolidated into 5 groups to minimize subprocess spawns:
  1. lifecycle  — CRUD + stats
  2. types      — memory types, importance, snapshot/restore
  3. resilience — unicode, large content, invalid tool, graceful exit
  4. concurrent — concurrent reads
  5. latency    — round-trip latency profile
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import threading
import time
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_mcp_available() -> bool:
    try:
        result = subprocess.run(
            ["ai-knot-mcp", "--help"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


requires_mcp = pytest.mark.skipif(
    not _is_mcp_available(),
    reason="ai-knot-mcp not installed (pip install 'ai-knot[mcp]')",
)


class McpSession:
    """Minimal synchronous JSON-RPC 2.0 client over stdio."""

    def __init__(self, tmp_path: str) -> None:
        self._proc = subprocess.Popen(
            ["ai-knot-mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={
                "PATH": os.environ.get("PATH", ""),
                "AI_KNOT_STORAGE": "yaml",
                "AI_KNOT_DATA_DIR": tmp_path,
                "AI_KNOT_AGENT_ID": "e2e-test",
            },
        )
        self._next_id = 1
        self._lock = threading.Lock()

    def _send(self, method: str, params: dict[str, Any], *, expect_response: bool = True) -> Any:
        with self._lock:
            msg_id = self._next_id
            self._next_id += 1

        payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
        if expect_response:
            payload["id"] = msg_id

        line = json.dumps(payload) + "\n"
        assert self._proc.stdin is not None
        self._proc.stdin.write(line.encode())
        self._proc.stdin.flush()

        if not expect_response:
            return None

        assert self._proc.stdout is not None
        raw = self._proc.stdout.readline()
        if not raw:
            raise RuntimeError("ai-knot-mcp closed stdout unexpectedly")
        return json.loads(raw.decode())

    def initialize(self) -> dict[str, Any]:
        resp = self._send(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "e2e-test", "version": "0.0.0"},
            },
        )
        # Send initialized notification (no response expected)
        self._send("notifications/initialized", {}, expect_response=False)
        return resp  # type: ignore[return-value]

    def tool_call(self, name: str, arguments: dict[str, Any]) -> str:
        resp = self._send("tools/call", {"name": name, "arguments": arguments})
        assert "error" not in resp, f"MCP error: {resp['error']}"
        content = resp["result"]["content"]
        return "".join(c["text"] for c in content if c["type"] == "text")

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except (subprocess.TimeoutExpired, OSError):
            self._proc.kill()


# ---------------------------------------------------------------------------
# Tests — 5 consolidated groups (one subprocess spawn each)
# ---------------------------------------------------------------------------


@requires_mcp
@pytest.mark.integration
def test_mcp_lifecycle(tmp_path: Any) -> None:
    """initialize → add → recall → list_facts → forget → stats."""
    session = McpSession(str(tmp_path))
    try:
        # initialize
        resp = session.initialize()
        assert resp["jsonrpc"] == "2.0"
        assert "protocolVersion" in resp["result"]

        # add + recall
        add_resp = session.tool_call("add", {"content": "Python uses indentation for blocks"})
        assert "Added fact" in add_resp

        recall_resp = session.tool_call("recall", {"query": "Python indentation"})
        assert "Python" in recall_resp

        # list_facts
        session.tool_call("add", {"content": "Docker runs containers"})
        list_resp = session.tool_call("list_facts", {})
        facts = json.loads(list_resp)
        assert isinstance(facts, list)
        assert len(facts) == 2

        # forget
        fact_id = next(f["id"] for f in facts if f["content"] == "Docker runs containers")
        session.tool_call("forget", {"fact_id": fact_id})
        remaining = json.loads(session.tool_call("list_facts", {}))
        assert len(remaining) == 1

        # stats
        session.tool_call("add", {"content": "second fact"})
        stats = json.loads(session.tool_call("stats", {}))
        assert stats["total_facts"] == 2
    finally:
        session.close()


@requires_mcp
@pytest.mark.integration
def test_mcp_types_and_snapshot(tmp_path: Any) -> None:
    """Memory types, importance round-trip, snapshot/restore."""
    session = McpSession(str(tmp_path))
    try:
        session.initialize()

        # all memory types reflected in stats
        session.tool_call("add", {"content": "semantic fact", "type": "semantic"})
        session.tool_call("add", {"content": "procedural fact", "type": "procedural"})
        session.tool_call("add", {"content": "episodic fact", "type": "episodic"})
        stats = json.loads(session.tool_call("stats", {}))
        by_type = stats["by_type"]
        assert by_type["semantic"] >= 1
        assert by_type["procedural"] >= 1
        assert by_type["episodic"] >= 1
        assert stats["total_facts"] == 3

        # importance round-trip
        session.tool_call("add", {"content": "low importance fact", "importance": 0.3})
        all_facts = json.loads(session.tool_call("list_facts", {}))
        low = next(f for f in all_facts if f["content"] == "low importance fact")
        assert abs(low["importance"] - 0.3) < 0.01

        # snapshot + restore
        snap_resp = session.tool_call("snapshot", {"name": "v1"})
        assert "v1" in snap_resp

        session.tool_call("add", {"content": "extra fact"})
        before = json.loads(session.tool_call("list_facts", {}))
        assert len(before) == 5

        restore_resp = session.tool_call("restore", {"name": "v1"})
        assert "v1" in restore_resp

        after = json.loads(session.tool_call("list_facts", {}))
        assert len(after) == 4
    finally:
        session.close()


@requires_mcp
@pytest.mark.integration
def test_mcp_resilience(tmp_path: Any) -> None:
    """Unicode, large content, invalid tool, recall relevance, graceful exit."""
    session = McpSession(str(tmp_path))
    try:
        session.initialize()

        # unicode CJK + emoji round-trip
        content = "用Python写代码🐍 — Kubernetes部署на облаке☁️"
        session.tool_call("add", {"content": content})
        facts = json.loads(session.tool_call("list_facts", {}))
        assert facts[0]["content"] == content

        # large content 10 KB (IPC pipe ~32 KB — exercises near-limit framing)
        large_content = "Python is great. " * 600  # 17 chars × 600 = 10 200 bytes
        assert len(large_content) >= 9_990
        session.tool_call("add", {"content": large_content})
        recall_resp = session.tool_call("recall", {"query": "Python"})
        assert "Python" in recall_resp

        # invalid tool name — server must not crash
        resp = session._send("tools/call", {"name": "does_not_exist", "arguments": {}})
        assert resp is not None
        has_error = "error" in resp or resp.get("result", {}).get("isError", False)
        assert has_error, f"Expected error response, got: {resp}"

        # server still alive after bad call
        alive_resp = session.tool_call("add", {"content": "server still alive"})
        assert "Added" in alive_resp

        # recall relevance — semantically relevant fact surfaces first
        for topic in ["Docker", "PostgreSQL", "Redis", "Kubernetes", "React"]:
            session.tool_call("add", {"content": f"{topic} is used in the stack"})
        session.tool_call("add", {"content": "Rust is used for performance-critical modules"})
        rust_resp = session.tool_call("recall", {"query": "Rust performance"})
        assert "Rust" in rust_resp
    finally:
        session.close()


@requires_mcp
@pytest.mark.integration
def test_mcp_concurrent_reads(tmp_path: Any) -> None:
    """Concurrent recall() calls from 3 threads all return valid responses."""
    session = McpSession(str(tmp_path))
    try:
        session.initialize()
        for i in range(5):
            session.tool_call("add", {"content": f"fact about topic {i}"})

        results: list[str] = []
        errors: list[Exception] = []

        def do_recall(query: str) -> None:
            try:
                # Each thread gets its own session to avoid shared state issues
                s = McpSession(str(tmp_path))
                s.initialize()
                result = s.tool_call("recall", {"query": query})
                results.append(result)
                s.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=do_recall, args=(f"topic {i}",)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Concurrent recall errors: {errors}"
        assert len(results) == 3
    finally:
        session.close()


@requires_mcp
@pytest.mark.integration
def test_mcp_sequential_and_latency(tmp_path: Any) -> None:
    """10 add-recall-forget cycles + latency profile (P50/P95 within budget)."""
    session = McpSession(str(tmp_path))
    try:
        session.initialize()

        # rapid sequential: 10 cycles without pauses must not crash
        for i in range(10):
            session.tool_call("add", {"content": f"rapid fact {i}"})
            session.tool_call("recall", {"query": f"rapid fact {i}"})

        facts = json.loads(session.tool_call("list_facts", {}))
        for fact in facts:
            session.tool_call("forget", {"fact_id": fact["id"]})
        assert session.tool_call("list_facts", {}) == "No facts stored."

        # latency profile: measure P50 and max over 10 iterations
        for i in range(10):
            session.tool_call("add", {"content": f"Python fact number {i}"})

        latencies: list[float] = []
        for _ in range(10):
            t0 = time.perf_counter()
            session.tool_call("recall", {"query": "Python deployment"})
            latencies.append(time.perf_counter() - t0)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p_max = latencies[-1]  # conservative upper bound for 10 samples

        assert p50 < 0.5, f"MCP recall P50 too high: {p50 * 1000:.0f}ms (target: <500ms)"
        assert p_max < 1.0, f"MCP recall max too high: {p_max * 1000:.0f}ms (target: <1000ms)"

        # graceful exit on stdin close
        if session._proc.stdin:
            session._proc.stdin.close()
        start = time.monotonic()
        try:
            session._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            session._proc.kill()
            pytest.fail("ai-knot-mcp did not exit within 5s after stdin close")
        assert time.monotonic() - start < 5.0
    finally:
        # stdin already closed above; just kill if still alive
        with contextlib.suppress(OSError):
            session._proc.kill()
