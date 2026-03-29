"""End-to-end tests for the agentmemo-mcp JSON-RPC server.

Spawns the real `agentmemo-mcp` subprocess and communicates with it over
stdio using raw JSON-RPC 2.0, without any Python client library in between.

Requires the mcp extra:
    pip install "agentmemo[mcp]"

Marked @pytest.mark.integration — included in normal CI but skipped if the
`agentmemo-mcp` command is not available.
"""

from __future__ import annotations

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
            ["agentmemo-mcp", "--help"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


requires_mcp = pytest.mark.skipif(
    not _is_mcp_available(),
    reason="agentmemo-mcp not installed (pip install 'agentmemo[mcp]')",
)


class McpSession:
    """Minimal synchronous JSON-RPC 2.0 client over stdio."""

    def __init__(self, tmp_path: str) -> None:
        self._proc = subprocess.Popen(
            ["agentmemo-mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env={
                "PATH": os.environ.get("PATH", ""),
                "AGENTMEMO_STORAGE": "yaml",
                "AGENTMEMO_DATA_DIR": tmp_path,
                "AGENTMEMO_AGENT_ID": "e2e-test",
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
            raise RuntimeError("agentmemo-mcp closed stdout unexpectedly")
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
# Tests
# ---------------------------------------------------------------------------


@requires_mcp
@pytest.mark.integration
def test_mcp_initialize(tmp_path: Any) -> None:
    """Server responds to MCP initialize with protocolVersion."""
    session = McpSession(str(tmp_path))
    try:
        resp = session.initialize()
        assert resp["jsonrpc"] == "2.0"
        result = resp["result"]
        assert "protocolVersion" in result
    finally:
        session.close()


@requires_mcp
@pytest.mark.integration
def test_mcp_add_and_recall(tmp_path: Any) -> None:
    """add() then recall() returns the stored fact."""
    session = McpSession(str(tmp_path))
    try:
        session.initialize()

        add_resp = session.tool_call("add", {"content": "Python uses indentation for blocks"})
        assert "Added fact" in add_resp

        recall_resp = session.tool_call("recall", {"query": "Python indentation"})
        assert "Python" in recall_resp
    finally:
        session.close()


@requires_mcp
@pytest.mark.integration
def test_mcp_list_facts(tmp_path: Any) -> None:
    """list_facts() returns JSON array after adding a fact."""
    session = McpSession(str(tmp_path))
    try:
        session.initialize()
        session.tool_call("add", {"content": "Docker runs containers"})

        list_resp = session.tool_call("list_facts", {})
        facts = json.loads(list_resp)
        assert isinstance(facts, list)
        assert len(facts) == 1
        assert facts[0]["content"] == "Docker runs containers"
    finally:
        session.close()


@requires_mcp
@pytest.mark.integration
def test_mcp_forget(tmp_path: Any) -> None:
    """forget() removes the fact; list_facts() returns empty afterwards."""
    session = McpSession(str(tmp_path))
    try:
        session.initialize()
        session.tool_call("add", {"content": "Temporary fact"})

        facts = json.loads(session.tool_call("list_facts", {}))
        assert len(facts) == 1
        fact_id = facts[0]["id"]

        session.tool_call("forget", {"fact_id": fact_id})
        after = session.tool_call("list_facts", {})
        assert after == "No facts stored."
    finally:
        session.close()


@requires_mcp
@pytest.mark.integration
def test_mcp_stats(tmp_path: Any) -> None:
    """stats() returns a JSON object with total_facts."""
    session = McpSession(str(tmp_path))
    try:
        session.initialize()
        session.tool_call("add", {"content": "fact one"})
        session.tool_call("add", {"content": "fact two"})

        stats = json.loads(session.tool_call("stats", {}))
        assert stats["total_facts"] == 2
    finally:
        session.close()


@requires_mcp
@pytest.mark.integration
def test_mcp_graceful_exit_on_stdin_close(tmp_path: Any) -> None:
    """Server exits cleanly when stdin is closed (no zombie process)."""
    session = McpSession(str(tmp_path))
    session.initialize()

    if session._proc.stdin:
        session._proc.stdin.close()

    start = time.monotonic()
    try:
        session._proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        session._proc.kill()
        pytest.fail("agentmemo-mcp did not exit within 5s after stdin close")

    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"Process took {elapsed:.1f}s to exit"


@requires_mcp
@pytest.mark.integration
def test_mcp_concurrent_reads(tmp_path: Any) -> None:
    """Multiple concurrent recall() calls all return valid responses."""
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


# ---------------------------------------------------------------------------
# Role 5: QA Engineer — functional correctness
# ---------------------------------------------------------------------------


@requires_mcp
@pytest.mark.integration
def test_mcp_snapshot_and_restore(tmp_path: Any) -> None:
    """snapshot() + restore() round-trip: state reverts to snapshot contents."""
    session = McpSession(str(tmp_path))
    try:
        session.initialize()
        session.tool_call("add", {"content": "fact alpha"})
        session.tool_call("add", {"content": "fact beta"})
        session.tool_call("add", {"content": "fact gamma"})

        snap_resp = session.tool_call("snapshot", {"name": "v1"})
        assert "v1" in snap_resp

        session.tool_call("add", {"content": "fact delta"})
        session.tool_call("add", {"content": "fact epsilon"})

        before_restore = json.loads(session.tool_call("list_facts", {}))
        assert len(before_restore) == 5

        restore_resp = session.tool_call("restore", {"name": "v1"})
        assert "v1" in restore_resp

        after_restore = json.loads(session.tool_call("list_facts", {}))
        assert len(after_restore) == 3
        contents = {f["content"] for f in after_restore}
        assert contents == {"fact alpha", "fact beta", "fact gamma"}
    finally:
        session.close()


@requires_mcp
@pytest.mark.integration
def test_mcp_all_memory_types(tmp_path: Any) -> None:
    """stats() by_type reflects all three memory types after adding each."""
    session = McpSession(str(tmp_path))
    try:
        session.initialize()
        session.tool_call("add", {"content": "semantic fact", "type": "semantic"})
        session.tool_call("add", {"content": "procedural fact", "type": "procedural"})
        session.tool_call("add", {"content": "episodic fact", "type": "episodic"})

        stats = json.loads(session.tool_call("stats", {}))
        by_type = stats["by_type"]
        assert by_type["semantic"] >= 1
        assert by_type["procedural"] >= 1
        assert by_type["episodic"] >= 1
        assert stats["total_facts"] == 3
    finally:
        session.close()


@requires_mcp
@pytest.mark.integration
def test_mcp_recall_relevance_ordering(tmp_path: Any) -> None:
    """recall() returns the semantically relevant fact among unrelated ones."""
    session = McpSession(str(tmp_path))
    try:
        session.initialize()
        for topic in [
            "Python",
            "Docker",
            "PostgreSQL",
            "Redis",
            "Kubernetes",
            "React",
            "FastAPI",
            "Nginx",
            "Prometheus",
            "Grafana",
        ]:
            session.tool_call("add", {"content": f"{topic} is used in the stack"})
        session.tool_call("add", {"content": "Rust is used for performance-critical modules"})

        recall_resp = session.tool_call("recall", {"query": "Rust performance"})
        assert "Rust" in recall_resp
    finally:
        session.close()


@requires_mcp
@pytest.mark.integration
def test_mcp_importance_round_trip(tmp_path: Any) -> None:
    """add() with custom importance → list_facts() returns that importance value."""
    session = McpSession(str(tmp_path))
    try:
        session.initialize()
        session.tool_call("add", {"content": "low importance fact", "importance": 0.3})

        facts = json.loads(session.tool_call("list_facts", {}))
        assert len(facts) == 1
        assert abs(facts[0]["importance"] - 0.3) < 0.01
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Role 6: Chaos Engineer — boundary conditions and resilience
# ---------------------------------------------------------------------------


@requires_mcp
@pytest.mark.integration
def test_mcp_large_content_10kb(tmp_path: Any) -> None:
    """A 10 KB fact survives add() → recall() round-trip intact.

    IPC pipe buffers average ~32 KB (research: netmeister.org/blog/ipcbufs.html).
    A 10 KB payload exercises near-limit framing without deadlock risk.
    """
    large_content = "Python is great. " * 600  # 17 chars × 600 = 10 200 bytes
    assert len(large_content) >= 9_990

    session = McpSession(str(tmp_path))
    try:
        session.initialize()
        session.tool_call("add", {"content": large_content})

        recall_resp = session.tool_call("recall", {"query": "Python"})
        assert "Python" in recall_resp
    finally:
        session.close()


@requires_mcp
@pytest.mark.integration
def test_mcp_unicode_cjk_emoji(tmp_path: Any) -> None:
    """CJK characters and emoji survive add() → list_facts() round-trip unchanged."""
    content = "用Python写代码🐍 — Kubernetes部署на облаке☁️"

    session = McpSession(str(tmp_path))
    try:
        session.initialize()
        session.tool_call("add", {"content": content})

        facts = json.loads(session.tool_call("list_facts", {}))
        assert len(facts) == 1
        assert facts[0]["content"] == content
    finally:
        session.close()


@requires_mcp
@pytest.mark.integration
def test_mcp_invalid_tool_name_error(tmp_path: Any) -> None:
    """Calling a nonexistent tool returns an error without crashing the server."""
    session = McpSession(str(tmp_path))
    try:
        session.initialize()
        resp = session._send("tools/call", {"name": "does_not_exist", "arguments": {}})
        # Server must respond (not crash); response contains error or isError
        assert resp is not None
        has_error = "error" in resp or resp.get("result", {}).get("isError", False)
        assert has_error, f"Expected error response, got: {resp}"

        # Server must still be alive after the bad call
        add_resp = session.tool_call("add", {"content": "server still alive"})
        assert "Added" in add_resp
    finally:
        session.close()


@requires_mcp
@pytest.mark.integration
def test_mcp_rapid_sequential_calls(tmp_path: Any) -> None:
    """10 add-recall-forget cycles without pauses must not crash the server."""
    session = McpSession(str(tmp_path))
    try:
        session.initialize()
        for i in range(10):
            session.tool_call("add", {"content": f"rapid fact {i}"})
            session.tool_call("recall", {"query": f"rapid fact {i}"})

        # Clean up all facts
        facts = json.loads(session.tool_call("list_facts", {}))
        for fact in facts:
            session.tool_call("forget", {"fact_id": fact["id"]})

        final = session.tool_call("list_facts", {})
        assert final == "No facts stored."
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Role 7: Observability Engineer — latency profiling
# ---------------------------------------------------------------------------


@requires_mcp
@pytest.mark.integration
def test_mcp_round_trip_latency_profile(tmp_path: Any) -> None:
    """Measure real MCP tool-call latency: P50 and P95 must be within budget.

    Research context:
    - Pure MCP stdio JSON-RPC (no tool execution): P95 ~10ms (tmdevlab benchmark)
    - Anthropic agent memory budget: <100ms per tool call (platform.claude.com docs)
    - mem0 with LLM: P95 ~1.4s — our goal: <500ms (TF-IDF + YAML, no LLM)

    This test uses 10 iterations; the maximum latency is used as a conservative
    upper bound (equivalent to P100 for this sample size).
    """
    session = McpSession(str(tmp_path))
    try:
        session.initialize()
        # Pre-populate so recall() does real work
        for i in range(10):
            session.tool_call("add", {"content": f"Python fact number {i}"})

        latencies: list[float] = []
        for _ in range(10):
            t0 = time.perf_counter()
            session.tool_call("recall", {"query": "Python deployment"})
            latencies.append(time.perf_counter() - t0)

        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[-1]  # max = p100 for 10 samples

        assert p50 < 0.5, f"MCP recall P50 too high: {p50 * 1000:.0f}ms (target: <500ms)"
        assert p95 < 1.0, f"MCP recall P95 too high: {p95 * 1000:.0f}ms (target: <1000ms)"
    finally:
        session.close()
