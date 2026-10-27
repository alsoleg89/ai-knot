"""The local load harness must check semantic invariants, not timing thresholds."""

import importlib.util
from pathlib import Path


def test_multi_agent_load_invariants(sql_storage):
    path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_multi_agent.py"
    spec = importlib.util.spec_from_file_location("multi_agent_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = "sqlite" if hasattr(sql_storage, "_db_path") else "postgres"
    location = str(sql_storage._db_path) if backend == "sqlite" else sql_storage._dsn
    report = module.benchmark(backend, location, agents=2, operations=2)
    invariants = report["invariants"]
    assert invariants["published"] == 4
    assert invariants["recalled"] == 4
    assert invariants["usage_counter"] == 4
    assert invariants["consumer_agents"] == 2
    assert invariants["permits"] == 1
    assert invariants["refresh_provider_calls"] == 1
    assert invariants["idle_provider_calls"] == 0
    assert report["publish"]["samples"] == 4
    assert report["recall"]["samples"] == 4


def test_load_harness_rejects_lost_usage(sql_storage, monkeypatch):
    import pytest

    path = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_multi_agent.py"
    spec = importlib.util.spec_from_file_location("multi_agent_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    backend = "sqlite" if hasattr(sql_storage, "_db_path") else "postgres"
    location = str(sql_storage._db_path) if backend == "sqlite" else sql_storage._dsn
    original = sql_storage.load_pool_stats

    def damaged_totals():
        totals = original()
        if totals:
            owner = next(iter(totals["used_count"]))
            totals["used_count"][owner] -= 1
        return totals

    monkeypatch.setattr(sql_storage, "load_pool_stats", damaged_totals)
    monkeypatch.setattr(module, "storage", lambda *_: sql_storage)
    with pytest.raises(AssertionError, match="lost/double recall usage"):
        module.benchmark(backend, location, agents=2, operations=1)
