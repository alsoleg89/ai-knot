"""The synthetic comparison checks its controls, not model answer accuracy."""

import subprocess

import pytest
from tests.eval.task_context.fixtures import CORPUS_SHA256, TRACES, manifest_sha256
from tests.eval.task_context.runner import (
    ARMS,
    _memory,
    _metrics,
    _seed,
    run_trace,
    storage_descriptor,
)

from ai_knot.always_on import EvidenceRef
from ai_knot.always_on.models import ContextFact


def test_trace_gold_and_manifest_are_frozen_before_retrieval():
    assert len(TRACES) == 16
    assert manifest_sha256() == CORPUS_SHA256
    assert len({trace.id for trace in TRACES}) == 16
    for trace in TRACES:
        assert len(trace.checkpoints) == 4
        assert len(trace.requirements) == 3
        for gold in trace.checkpoints:
            assert len(gold.required_groups) == 3
            assert not set().union(*map(set, gold.required_groups)) & set(gold.forbidden)
        if not trace.matches:
            assert all(g.required_groups[1:] == ((), ()) for g in trace.checkpoints)


def test_stronger_ordinary_controls_preserve_the_original_four_arms():
    assert set(ARMS) == {
        "ordinary_recall",
        "scoped_ordinary_recall",
        "task_only",
        "task_role",
        "ordinary_integrate_recall",
        "scoped_ordinary_integrate_recall",
    }


def test_trace_runs_equal_budget_arms_and_reopens_in_a_real_process(sql_storage, tmp_path):
    result = run_trace(sql_storage, TRACES[0], tmp_path, storage_descriptor(sql_storage))
    assert len(result) == len(ARMS) * 4
    assert {row["arm"] for row in result} == set(ARMS)
    assert all(row["controls"]["corpus_unchanged"] for row in result)
    assert all(row["controls"]["same_source_corpus"] for row in result)
    assert all(row["controls"]["same_grants"] for row in result)
    assert all(row["controls"]["inference_endpoints_blocked"] for row in result)
    assert all(row["metrics"]["context_chars"] <= 600 for row in result)
    assert all(row["metrics"]["selected_count"] <= 3 for row in result)
    assert all(row["selection_model_calls"] == 0 for row in result)
    diagnostics = [row for row in result if row["arm"].endswith("integrate_recall")]
    first, second, third = TRACES[0].requirements
    assert all(
        row["selection"]["query"] == f"Integrate {first}, {second}, and {third} "
        f"for task {TRACES[0].task_key} recipient role {TRACES[0].role}"
        for row in diagnostics
    )
    handoff = [row for row in result if row["checkpoint"] == "handoff"]
    assert all(row["controls"]["fresh_process"] for row in handoff)
    assert all(row["controls"]["saved_refs_reloaded"] for row in handoff)
    assert all(row["controls"]["source_data_reloaded"] for row in handoff)
    assert all(row["worker_pid"] != row["parent_pid"] for row in handoff)
    peer_rows = {row["arm"]: row for row in result if row["checkpoint"] == "peer_update"}
    assert all(
        row["reopened_saved_context"]["saved_ids"]
        == [fact["ref"]["fact_id"] for fact in peer_rows[row["arm"]]["selected"]]
        for row in handoff
    )
    final = [row for row in result if row["checkpoint"] == "correction_or_revoke"]
    assert all(
        not set(row["reopened_saved_context"]["accepted_ids"]) & {"rollout-lesson", "rollout-copy"}
        for row in final
    )


def test_handoff_failure_preserves_worker_diagnostics(sql_storage, tmp_path, monkeypatch):
    def failed_worker(*args, **kwargs):
        raise subprocess.CalledProcessError(9, args[0], stderr="worker-basis-error")

    monkeypatch.setattr("tests.eval.task_context.runner.subprocess.run", failed_worker)
    with pytest.raises(subprocess.CalledProcessError) as failure:
        run_trace(sql_storage, TRACES[0], tmp_path, storage_descriptor(sql_storage))
    assert any("worker-basis-error" in note for note in getattr(failure.value, "__notes__", []))


def test_worker_diagnostic_redacts_test_dsn_and_password_variants():
    pytest.importorskip("psycopg")
    from tests.eval.task_context import runner

    dsn = "postgresql://reader:p%40ss%2Fword@localhost/test?sslpassword=cert%2Bsecret"
    error = f"worker connection failed: {dsn}; password=p@ss/word; sslpassword=cert+secret"
    clean = runner._worker_diagnostic(error, {"backend": "postgres", "dsn": dsn})
    for secret in (dsn, "p@ss/word", "p%40ss%2Fword", "cert+secret", "cert%2Bsecret"):
        assert secret not in clean
    assert "worker connection failed" in clean


def test_role_words_do_not_create_evidence_or_lexical_coverage():
    trace = TRACES[0]
    role_only = ContextFact(
        EvidenceRef("__shared__", "rollout-role-only", "fixed"), "Operator reviewer duties", ""
    )
    metrics, _ = _metrics(
        trace,
        0,
        trace.checkpoints[0],
        (role_only,),
        {
            "reported_covered_requirements": list(trace.requirements),
            "coverage_report_available": True,
        },
    )
    assert metrics["required_groups_covered"] == 0
    assert metrics["falsely_reported_lexical_coverage"] == 3
    assert metrics["reported_without_gold_evidence"] == 3
    assert metrics["role_only_selected"] == 1


def test_other_role_is_valid_evidence_but_wrong_applicability_is_not():
    other_role = ContextFact(
        EvidenceRef("__shared__", "rollout-reviewer", "fixed"), TRACES[0].family.procedure, ""
    )
    for trace, forbidden, role_fit in ((TRACES[0], 0, 0.0), (TRACES[1], 1, None)):
        metrics, _ = _metrics(
            trace,
            0,
            trace.checkpoints[0],
            (other_role,),
            {"reported_covered_requirements": None, "coverage_report_available": False},
        )
        assert metrics["forbidden_selected"] == forbidden
        assert metrics["recipient_role_fit"] == role_fit


def test_copied_lessons_keep_one_independent_observation(sql_storage):
    trace = TRACES[0]
    _seed(sql_storage, trace, 1)
    memory = _memory(sql_storage, 1)
    refs = memory.capture_evidence(
        (("__shared__", "rollout-lesson"), ("__shared__", "rollout-copy"))
    )
    roots = memory.independent_evidence(refs)
    assert len(roots) == 1 and roots[0].fact_id == "rollout-success-source"
    _seed(sql_storage, trace, 3)
    assert not _memory(sql_storage, 3).review_context(refs).accepted
