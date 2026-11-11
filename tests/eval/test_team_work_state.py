import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from ai_knot.always_on.changes import evidence_revision
from ai_knot.types import Fact
from tests.eval.task_context.runner import isolated_backend, storage_descriptor
from tests.eval.team_repos.cases import Case, Document, Stage
from tests.eval.team_repos.protocol import coder_messages, researcher_messages


def sample(number=0):
    scope = f"task-{number}"
    stages = []
    for i in range(3):
        docs = [Document("docs/rule.md", f"Value {min(i, 1)}", scope)]
        if i < 2:
            docs.append(Document("docs/withdrawn.md", "Old permission", scope))
        stages.append(
            Stage(
                f"stage-{i}",
                "Implement value",
                tuple(docs),
                "PRIVATE_TEST_SENTINEL",
                "PRIVATE_REFERENCE_SENTINEL",
            )
        )
    return Case(
        f"case-{number}",
        scope,
        "Sample",
        "Implement value",
        ("value",),
        "def value(): pass",
        tuple(stages),
    )


@pytest.fixture(params=("sqlite", "postgres"))
def backend(request, tmp_path):
    if request.param == "postgres" and not os.environ.get("AI_KNOT_TEST_PG_DSN"):
        if os.environ.get("AI_KNOT_REQUIRE_POSTGRES") == "1":
            pytest.fail("PostgreSQL required")
        pytest.skip("PostgreSQL DSN absent")
    with isolated_backend(request.param, tmp_path) as storage:
        yield storage


def encoded():
    return json.loads(
        json.dumps(
            {
                "format_version": 1,
                "provenance": {"synthetic": True},
                "cases": [asdict(sample(i)) for i in range(3)],
            }
        )
    )


def test_corpus_requires_exact_frozen_bytes_and_private_fields_stay_private(tmp_path):
    from tests.eval.team_work.corpus import load_corpus

    path = tmp_path / "cases.json"
    path.write_text(json.dumps(encoded()))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cases = load_corpus(path, expected_sha256=digest)
    assert len(cases) == 3 and cases[0] == sample()
    payload = json.dumps(researcher_messages(cases[0], cases[0].stages[0], []))
    payload += json.dumps(coder_messages(cases[0], cases[0].stages[0], "own code", [], None))
    assert "PRIVATE_TEST_SENTINEL" not in payload
    assert "PRIVATE_REFERENCE_SENTINEL" not in payload
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="freeze"):
        load_corpus(path, expected_sha256=digest)


@pytest.mark.parametrize("bad", ["extra", "duplicate_path", "duplicate_case", "bad_version"])
def test_corpus_rejects_ambiguous_or_untyped_input(tmp_path, bad):
    from tests.eval.team_work.corpus import load_corpus

    data = encoded()
    if bad == "extra":
        data["cases"][0]["owner_override"] = "foreign"
    elif bad == "duplicate_path":
        docs = data["cases"][0]["stages"][0]["documents"]
        docs.append(docs[0])
    elif bad == "duplicate_case":
        data["cases"][1] = data["cases"][0]
    else:
        data["format_version"] = True
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        load_corpus(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def test_source_publication_preserves_executor_state_and_exposes_stale_basis(backend):
    from tests.eval.team_work.state import memory_for, publish_stage, source_facts

    case = sample()
    peer = Fact("Peer memory", id="peer", origin_agent_id="peer", slot_key="case-0::peer")
    backend.save("__shared__", [peer])
    publish_stage(backend, case, 0)
    memory = memory_for(backend, 0)
    root = source_facts(case, 0)[0]
    basis = memory.capture_evidence((("__shared__", root.id),))
    lesson = memory.derive_fact("Keep the observation", basis=basis, task_key=case.task_key)
    lesson_refs = memory.capture_evidence(((memory._owner, lesson.id),))
    intent = memory.create_intent(client_key="one", goal=case.goal)
    lease = memory.claim(intent.id, worker_id="executor")
    before = memory.checkpoint(
        intent.id,
        lease_token=lease.token,
        expected_revision=memory.get_intent(intent.id).revision,
        state="ready",
        progress={"code": "OWN_CODE", "receipt": {"status": "fail"}},
        basis=basis,
    )
    publish_stage(backend, case, 1)
    assert memory.get_intent(intent.id) == before
    assert next(f for f in backend.load("__shared__") if f.id == "peer").content == "Peer memory"
    reopened = memory_for(backend, 1)
    packet = reopened.resume(intent.id)
    assert packet.intent.progress["code"] == "OWN_CODE"
    assert packet.intent.basis == basis
    assert [issue.reason for issue in packet.stale_basis] == ["changed"]
    after_resume = reopened.get_intent(intent.id)
    assert after_resume.state.value == "blocked"
    assert not reopened.review_context(lesson_refs).complete
    child = """
import json, os, sys
from tests.eval.task_context.runner import _reopen
from tests.eval.team_work.state import memory_for
from ai_knot.always_on.codec import encode_always_on
job = json.load(sys.stdin)
packet = memory_for(_reopen(job['storage']), 1).resume(job['intent_id'])
print(json.dumps({'pid': os.getpid(), 'progress': packet.intent.progress,
                 'basis': encode_always_on(packet.intent.basis),
                 'stale': [issue.reason for issue in packet.stale_basis]}))
"""
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-c", child],
        input=json.dumps({"storage": storage_descriptor(backend), "intent_id": intent.id}),
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
        env={
            "PATH": os.environ.get("PATH", os.defpath),
            "PYTHONPATH": str(root / "src") + os.pathsep + str(root),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    packet = json.loads(result.stdout)
    assert packet["pid"] != os.getpid()
    assert packet["progress"] == {"code": "OWN_CODE", "receipt": {"status": "fail"}}
    assert packet["basis"] == [asdict(ref) for ref in basis]
    assert packet["stale"] == ["changed"]
    assert memory.get_intent(intent.id) == after_resume


def test_source_replay_preserves_versions_times_and_derived_facts(backend):
    from tests.eval.team_work.state import memory_for, publish_stage, source_facts

    case = sample()
    publish_stage(backend, case, 0)
    memory = memory_for(backend, 0)
    root = source_facts(case, 0)[0]
    basis = memory.capture_evidence((("__shared__", root.id),))
    lesson = memory.derive_fact("Retained lesson", basis=basis, task_key=case.task_key)
    before = backend.load("__shared__")
    publish_stage(backend, case, 0)
    assert backend.load("__shared__") == before
    publish_stage(backend, case, 1)
    updated = {f.id: f for f in backend.load("__shared__")}
    assert updated[root.id].version == 2
    assert (
        next(f for f in backend.load(memory._owner) if f.id == lesson.id).content
        == "Retained lesson"
    )
    publish_stage(backend, case, 2)
    final = backend.load("__shared__")
    assert [f.id for f in final] == [root.id]
    assert evidence_revision(final[0]) == evidence_revision(updated[root.id])
    assert final[0].created_at == updated[root.id].created_at


def test_source_order_is_total_and_public_rendering_ignores_storage_iteration(backend):
    from tests.eval.team_work.state import memory_for, ordered_entries, publish_stage, source_facts

    case = sample()
    facts = source_facts(case, 0)
    assert len({f.created_at for f in facts}) == len(facts)
    from tests.eval.team_repos.runner import entries

    raw = entries(facts)
    assert ordered_entries(raw) == ordered_entries(tuple(reversed(raw)))
    publish_stage(backend, case, 0)
    view = memory_for(backend, 0).task_snapshot(task_key=case.task_key)
    assert {f.ref.fact_id for f in view} == {f.id for f in facts}


def test_ordered_entries_distinguishes_namespaces_and_reindexes_without_mutation():
    from tests.eval.team_work.state import ordered_entries

    rows = [
        {
            "index": 99,
            "content": "Private",
            "ref": {"namespace": "owner", "fact_id": "same", "revision": "r"},
        },
        {
            "index": 12,
            "content": "Shared",
            "ref": {"namespace": "__shared__", "fact_id": "same", "revision": "r"},
        },
    ]
    result = ordered_entries(rows)
    assert [r["content"] for r in result] == ["Shared", "Private"]
    assert [r["index"] for r in result] == [1, 2]
    assert [r["index"] for r in rows] == [99, 12]


def test_equal_source_sets_make_byte_identical_model_messages(backend):
    from tests.eval.team_repos.runner import entries
    from tests.eval.team_work.state import memory_for, ordered_entries, publish_stage

    case = sample()
    publish_stage(backend, case, 0)
    control = ordered_entries(entries(backend.load("__shared__")))
    by_ref = {tuple(sorted(row["ref"].items())): row for row in control}
    snapshot = memory_for(backend, 0).task_snapshot(task_key=case.task_key)
    selected = ordered_entries([by_ref[tuple(sorted(asdict(f.ref).items()))] for f in snapshot])
    assert json.dumps(control) == json.dumps(selected)
    assert coder_messages(case, case.stages[0], case.initial_code, control, None) == coder_messages(
        case, case.stages[0], case.initial_code, selected, None
    )
