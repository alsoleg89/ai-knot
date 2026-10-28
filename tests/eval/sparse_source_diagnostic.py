"""Offline S9/S19/S26 source traces over isolated local SQL, never a live-model score."""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib
import json
import shutil
import socket
import subprocess
import tempfile
import types
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from tests.conftest import _fake_embed_texts
from tests.eval.benchmark.backends.ai_knot_multi_agent_backend import AiKnotMultiAgentBackend
from tests.eval.benchmark.base import BenchmarkMetrics
from tests.eval.benchmark.judge import MockJudge
from tests.eval.benchmark.ma_gate import evaluate_gate, gate_passed
from tests.eval.task_context.runner import isolated_backend, storage_descriptor

BASE = "987f885eba075d22aa43bbcbf4dba32bad4cf32a"
S26_PATH = "tests/eval/benchmark/scenarios/s26_sparse_assembly.py"
SCENARIOS = ("s9_ma_pool_publish", "s19_incident_reconstruction", "s26_sparse_assembly")


class Replay:
    def __init__(self, rows):
        self.rows = iter(rows)

    async def pool_retrieve(self, owner, query, *, top_k):
        original_query, original_top_k, result = next(self.rows)
        assert (query, top_k) == (original_query, original_top_k)
        return result


def write(output, report):
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)


async def diagnostic(kind, directory, report, output):
    s26 = importlib.import_module("tests.eval.benchmark.scenarios.s26_sparse_assembly")
    legacy = types.ModuleType("legacy_s26_scoring")
    source = subprocess.check_output(["git", "show", f"{BASE}:{S26_PATH}"]).decode()
    exec(compile(source, "legacy-s26-scoring", "exec"), legacy.__dict__)
    with isolated_backend(kind, directory) as storage:
        descriptor = storage_descriptor(storage)
        backend = AiKnotMultiAgentBackend(kind, postgres_dsn=descriptor.get("dsn", ""))
        original = backend.pool_retrieve
        metrics = BenchmarkMetrics(backend.name, selected_scenarios=list(SCENARIOS))
        try:
            for name in SCENARIOS:
                module = importlib.import_module("tests.eval.benchmark.scenarios." + name)
                traces = []
                replays = []

                async def capture(owner, query, *, top_k, replays=replays, traces=traces):
                    result = await original(owner, query, top_k=top_k)
                    replays.append((query, top_k, copy.deepcopy(result)))
                    traces.append(
                        {
                            "query": query,
                            "top_k": top_k,
                            "rows": [
                                {
                                    "rank": i + 1,
                                    "id": f.id,
                                    "publisher": f.origin_agent_id,
                                    "text": f.content,
                                    "slot": f.slot_key,
                                    "entity": f.entity,
                                    "attribute": f.attribute,
                                    "version": f.version,
                                    "qualifiers": f.qualifiers,
                                    "active": f.is_active(),
                                }
                                for i, f in enumerate(result.facts)
                            ],
                            "texts": result.texts,
                        }
                    )
                    return result

                backend.pool_retrieve = capture
                result = await module.run(backend, MockJudge())
                metrics.scenario_results.append(result)
                row = {
                    "scenario": name,
                    "scores": result.judge_scores,
                    "notes": result.notes,
                    "traces": traces,
                }
                if name == "s26_sparse_assembly":
                    assert len(replays) == 30
                    row["legacy_scores_on_same_fresh_rows"] = {}
                    for point, n in enumerate((10, 100, 1000)):
                        batch = replays[point * 10 : (point + 1) * 10]
                        old = await legacy._measure_at_checkpoint(Replay(batch), n, 5)
                        # Replay has no SQL latency: never label it a timing measurement.
                        del old["p95_retrieve_ms"]
                        row["legacy_scores_on_same_fresh_rows"][str(n)] = old
                        markers = {s26._rare_marker(i): i for i in range(n)}
                        for i, (_, _, returned) in enumerate(batch):
                            trace = traces[point * 10 + i]
                            trace.update(scale=n, phase="conceptual" if i < 5 else "marker_named")
                            for entry, hits in zip(
                                trace["rows"], s26._retrieved_shards(returned, markers), strict=True
                            ):
                                entry["shard_source_indices"] = sorted(hits)
                                entry["publisher_domain"] = s26._agent_domain(entry["publisher"])
                report["scenarios"].append(row)
                write(output, report)
                print(kind, name, "complete", flush=True)
            gates = evaluate_gate(metrics)
            report["gate"] = {
                "binding_passed": sum(
                    r.passed for r in gates if r.applicable and not r.threshold.advisory
                ),
                "binding_total": sum(r.applicable and not r.threshold.advisory for r in gates),
                "passed": gate_passed(gates),
                "results": [asdict(r) for r in gates if r.applicable],
            }
        finally:
            if backend._tmp_dir:
                folder = Path(backend._tmp_dir)
                assert folder.name.startswith("aiknot_ma_")
                shutil.rmtree(folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sqlite", "postgres"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    attempts = []

    def forbidden(*a, **k):
        attempts.append("blocked")
        raise AssertionError("Network forbidden in offline source diagnostic")

    report = {
        "status": "running",
        "scope": "New instrumented structural source diagnostic, not earlier-run traces, "
        "real-model quality or capacity evidence",
        "backend": args.backend,
        "base_for_legacy_scorer": BASE,
        "scenarios": [],
        "provider_calls": 0,
        "embedding_model": "deterministic test stub; no provider",
        "network_attempts": 0,
        "network_policy": "Python create_connection/connect/connect_ex and httpx send blocked; "
        "PostgreSQL uses explicitly configured local libpq connection",
    }
    write(args.output, report)
    try:
        with (
            tempfile.TemporaryDirectory(prefix="sparse-source-diagnostic-") as temporary,
            patch("ai_knot.embedder.embed_texts", _fake_embed_texts),
            patch("socket.create_connection", forbidden),
            patch.object(socket.socket, "connect", forbidden),
            patch.object(socket.socket, "connect_ex", forbidden),
            patch("httpx.Client.send", forbidden),
            patch("httpx.AsyncClient.send", forbidden),
        ):
            asyncio.run(diagnostic(args.backend, Path(temporary), report, args.output))
        report["status"] = "complete"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__)
    report["network_attempts"] = len(attempts)
    write(args.output, report)
    return 0 if report["status"] == "complete" and report["gate"]["passed"] and not attempts else 1


if __name__ == "__main__":
    raise SystemExit(main())
