"""Serial, model-free profile of repeated source checks in context reads."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from ai_knot import KnowledgeBase, SharedMemoryPool
from ai_knot.always_on import AlwaysOnMemory
from ai_knot.always_on.codec import canonical_json
from ai_knot.always_on.lineage import derivation_qualifiers
from ai_knot.always_on.storage import SQLSession
from ai_knot.types import Fact
from tests.eval.task_context.runner import isolated_backend

NOW = datetime(2026, 10, 28, 12, tzinfo=UTC)


def seeded(storage, *, lessons=64, roots=16, noise=0):
    kb = KnowledgeBase("reader", storage=storage, embed_url="")
    pool = SharedMemoryPool(storage, persist_stats=True, embed_url="", clock=lambda: NOW)
    memory = AlwaysOnMemory(kb, pool=pool, clock=lambda: NOW)
    originals = [
        Fact(
            f"Verified deployment source {i}",
            id=f"root-{i:03d}",
            created_at=NOW,
            valid_from=NOW,
            slot_key=f"root::{i}",
            version=1,
            visibility_scope="ops",
            qualifiers={"task_key": "sources"},
        )
        for i in range(roots)
    ]
    unrelated = [
        Fact(
            f"Retained unrelated observation {i}",
            id=f"noise-{i:05d}",
            created_at=NOW,
            valid_from=NOW,
            qualifiers={"task_key": "unrelated"},
        )
        for i in range(noise)
    ]
    storage.save("__shared__", [*originals, *unrelated])
    pool.grant_read("reader", "ops")
    lineage = derivation_qualifiers("__shared__", originals) if originals else {}
    kb.replace_facts(
        [
            Fact(
                f"Verified deployment recovery step {i}",
                id=f"lesson-{i:03d}",
                created_at=NOW,
                valid_from=NOW,
                slot_key=f"deploy::{i}",
                version=1,
                qualifiers={**lineage, "task_key": "deploy"},
            )
            for i in range(lessons)
        ]
    )
    return memory


def measured(storage, operation):
    counts = {"fact_loads": 0, "sql_reads": 0}
    loader = type(storage)._load_on_connection
    cursor = SQLSession._cursor

    def load(self, *args, **kwargs):
        counts["fact_loads"] += 1
        return loader(self, *args, **kwargs)

    def query(self, statement, params):
        counts["sql_reads"] += 1
        return cursor(self, statement, params)

    started = time.perf_counter()
    with (
        patch.object(type(storage), "_load_on_connection", load),
        patch.object(SQLSession, "_cursor", query),
    ):
        result = operation()
    return result, {
        **counts,
        "total_reads": sum(counts.values()),
        "ms": (time.perf_counter() - started) * 1000,
    }


def profile(backend, directory, *, noise=0, repetitions=5, lessons=64, roots=16):
    with isolated_backend(backend, directory) as storage:
        memory = seeded(storage, lessons=lessons, roots=roots, noise=noise)
        refs = tuple(
            f.ref for f in memory.task_snapshot(task_key="deploy", max_facts=128, max_chars=65536)
        )
        methods = {
            "snapshot": lambda: memory.task_snapshot(
                task_key="deploy", max_facts=128, max_chars=65536
            ),
            "lexical": lambda: memory.select_context(
                ("deployment recovery",), task_key="deploy", max_facts=128, max_chars=65536
            ),
            "review": lambda: memory.review_context(refs, max_chars=65536),
        }
        rows = {}
        for name, operation in methods.items():
            samples = []
            digests = set()
            for _ in range(repetitions):
                result, sample = measured(storage, operation)
                public = (
                    [asdict(f) for f in result] if isinstance(result, tuple) else asdict(result)
                )
                digests.add(hashlib.sha256(canonical_json(public).encode()).hexdigest())
                samples.append(sample)
            assert len(digests) == 1
            rows[name] = {
                "samples": samples,
                "median_ms": statistics.median(s["ms"] for s in samples),
                "public_sha256": digests.pop(),
            }
        return {
            "backend": backend,
            "lessons": lessons,
            "roots": roots,
            "unrelated_retained": noise,
            "model_calls": 0,
            "operations": rows,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("sqlite", "postgres"), default="sqlite")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roots", type=int, choices=(0, 16), default=16)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="context-read-profile-") as directory:
        records = []
        for noise in (0, 2000):
            folder = Path(directory) / str(noise)
            folder.mkdir()
            records.append(profile(args.backend, folder, noise=noise, roots=args.roots))
    report = {
        "scope": "Five sequential local samples, full retained scan; "
        "no fleet-capacity or task-quality claim",
        "records": records,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
