"""Offline regressions for incomplete multi-agent acceptance evidence."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner, Result

from tests.eval.benchmark import runner
from tests.eval.benchmark.base import BenchmarkMetrics, ScenarioResult
from tests.eval.benchmark.ma_gate import evaluate_gate, gate_passed
from tests.eval.benchmark.report import _ma_scorecard, render_markdown

_BACKEND = "ai_knot_multi_agent"
_SCALE = "s26_sparse_assembly"
_SCALE_SCORES = {
    "target_shard_recall_at_10": [1.0],
    "equivalence_recall_at_1000": [1.0],
    "target_shard_recall_at_1000": [0.33],
    "distractor_rate_at_1000": [0.9],
    "p95_retrieve_ms_at_1000": [170.0],
}


def _result(sid: str, scores: dict[str, list[float]]) -> ScenarioResult:
    return ScenarioResult(sid, _BACKEND, scores, None, None)


@pytest.fixture
def offline_scenarios(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.eval.benchmark import scenarios
    from tests.eval.benchmark.backends import ai_knot_multi_agent_backend

    async def protocol(backend: object, judge: object) -> ScenarioResult:
        return _result("s10_ma_mesi_cas", {"cas_correctness": [1.0]})

    async def failed(backend: object, judge: object) -> ScenarioResult:
        raise RuntimeError("private-provider-DSN")

    monkeypatch.setattr(
        ai_knot_multi_agent_backend,
        "AiKnotMultiAgentBackend",
        lambda *args, **kwargs: SimpleNamespace(name=_BACKEND),
    )
    monkeypatch.setattr(
        scenarios,
        "_MA",
        [("s10_ma_mesi_cas", protocol), (_SCALE, failed), ("s8_ma_isolation", failed)],
    )


def _invoke(tmp_path: Path, selectors: str, *extra: str, standalone: bool = True) -> Result:
    return CliRunner().invoke(
        runner.main,
        [
            *(["--multi-agent"] if standalone else []),
            "--mock-judge",
            "--ma-gate",
            "--scenarios",
            selectors,
            "--output",
            str(tmp_path / "report.md"),
            "--raw-output",
            str(tmp_path / "raw.json"),
            "--jsonl-output",
            str(tmp_path / "live.jsonl"),
            *extra,
        ],
    )


def test_selected_exception_fails_cli_with_inspectable_evidence(
    tmp_path: Path, offline_scenarios: None
) -> None:
    result = _invoke(tmp_path, "s10,s26")
    assert result.exit_code == 1, result.output
    assert "1/3 binding thresholds FAIL" in result.output
    report = (tmp_path / "report.md").read_text()
    assert "1/3 | ❌ FAIL" in report
    assert _SCALE in report and "RuntimeError" in report
    raw = json.loads((tmp_path / "raw.json").read_text())
    assert raw["schema_version"] == 2
    metadata = raw["run_metadata"][f"{_BACKEND}:en"]
    assert metadata["selected_scenarios"] == ["s10_ma_mesi_cas", _SCALE]
    assert metadata["scenario_errors"] == {_SCALE: "RuntimeError"}
    assert "private-provider-DSN" not in report
    assert "private-provider-DSN" not in json.dumps(raw)


@pytest.mark.parametrize("metric", ["equivalence_recall_at_1000", "p95_retrieve_ms_at_1000"])
@pytest.mark.parametrize("empty", [False, True])
def test_present_scenario_missing_expected_metric_fails(metric: str, empty: bool) -> None:
    scores = dict(_SCALE_SCORES)
    if empty:
        scores[metric] = []
    else:
        del scores[metric]
    metrics = BenchmarkMetrics(_BACKEND, scenario_results=[_result(_SCALE, scores)])
    results = evaluate_gate(metrics)
    assert not gate_passed(results)
    missing = next(r for r in results if r.threshold.metric == metric)
    assert missing.applicable and not missing.passed
    assert "missing" in render_markdown([metrics]).lower()


def test_exception_without_thresholds_still_fails_cli(
    tmp_path: Path, offline_scenarios: None
) -> None:
    result = _invoke(tmp_path, "s10,s8")
    assert result.exit_code == 1, result.output
    assert "1/1 binding thresholds FAIL" in result.output
    assert "s8_ma_isolation" in (tmp_path / "report.md").read_text()


def test_all_selected_scenarios_failing_still_have_scorecard(
    tmp_path: Path, offline_scenarios: None
) -> None:
    result = _invoke(tmp_path, "s26")
    assert result.exit_code == 1, result.output
    report = (tmp_path / "report.md").read_text()
    assert "Acceptance Scorecard" in report
    assert "0/2 | ❌ FAIL" in report
    assert "RuntimeError" in report


def test_valid_partial_run_passes(tmp_path: Path, offline_scenarios: None) -> None:
    result = _invoke(tmp_path, "s10")
    assert result.exit_code == 0, result.output
    assert "1/1 binding thresholds PASS" in result.output
    report = (tmp_path / "report.md").read_text()
    assert "1/1 | ✅ PASS" in report
    assert "n/a" in report


def test_inline_ma_gate_is_enforced(
    tmp_path: Path, offline_scenarios: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "_build_scenarios", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        runner,
        "_build_backends_from_names",
        lambda *args, **kwargs: [SimpleNamespace(name="baseline")],
    )
    result = _invoke(tmp_path, "all", "--backends", "baseline", standalone=False)
    assert result.exit_code == 1, result.output
    assert "1/3 binding thresholds FAIL" in result.output
    assert "❌ FAIL" in (tmp_path / "report.md").read_text()


def test_missing_selected_result_cannot_disappear_from_gate() -> None:
    metrics = BenchmarkMetrics(
        _BACKEND,
        scenario_results=[_result("s10_ma_mesi_cas", {"cas_correctness": [1.0]})],
        selected_scenarios=["s10_ma_mesi_cas", _SCALE],
    )
    results = evaluate_gate(metrics)
    assert not gate_passed(results)
    assert "1/3 | ❌ FAIL" in render_markdown([metrics])
    assert "Missing result" in render_markdown([metrics])


def test_complete_scale_evidence_with_advisory_misses_passes() -> None:
    metrics = BenchmarkMetrics(_BACKEND, scenario_results=[_result(_SCALE, dict(_SCALE_SCORES))])
    assert gate_passed(evaluate_gate(metrics))


@pytest.mark.parametrize("selectors", ["", "unknown_scenario", "s10,unknown_scenario"])
def test_empty_or_unmatched_selector_cannot_pass(
    tmp_path: Path, offline_scenarios: None, selectors: str
) -> None:
    result = _invoke(tmp_path, selectors)
    assert result.exit_code != 0, result.output
    assert "binding thresholds PASS" not in result.output


@pytest.mark.parametrize(
    "metrics",
    [[], [BenchmarkMetrics(_BACKEND)], [BenchmarkMetrics("mem0_multi_agent")]],
)
def test_gate_requires_nonempty_ai_knot_verification(metrics: list[BenchmarkMetrics]) -> None:
    with pytest.raises(SystemExit, match="1"):
        runner._enforce_ma_gate(metrics)


@pytest.mark.parametrize(
    "scores",
    [
        {},
        {"overlap_coverage": []},
        {"overlap_coverage": [float("nan")]},
        {"overlap_coverage": [float("inf")]},
        {"overlap_coverage": [float("-inf")]},
        {"overlap_coverage": [1.0], "other": [float("nan")]},
    ],
)
def test_selected_nongated_scenario_requires_usable_measurements(
    scores: dict[str, list[float]], capsys: pytest.CaptureFixture[str]
) -> None:
    metrics = BenchmarkMetrics(
        _BACKEND,
        selected_scenarios=["s10_ma_mesi_cas", "s8_ma_isolation"],
        scenario_results=[
            _result("s10_ma_mesi_cas", {"cas_correctness": [1.0]}),
            _result("s8_ma_isolation", scores),
        ],
    )
    assert "s8_ma_isolation" in metrics.scenario_failures()
    assert "1/1 | ❌ FAIL" in render_markdown([metrics])
    with pytest.raises(SystemExit, match="1"):
        runner._enforce_ma_gate([metrics])
    assert "1/1 binding thresholds FAIL" in capsys.readouterr().out


@pytest.mark.parametrize("metric", ["equivalence_recall_at_1000", "p95_retrieve_ms_at_1000"])
@pytest.mark.parametrize(
    "samples",
    [[float("nan")], [float("inf")], [float("-inf")], [float("-inf"), 1.0, float("inf")]],
)
def test_nonfinite_binding_or_advisory_evidence_cannot_verify(
    metric: str, samples: list[float], tmp_path: Path
) -> None:
    scores = {**_SCALE_SCORES, metric: samples}
    metrics = BenchmarkMetrics(
        _BACKEND, selected_scenarios=[_SCALE], scenario_results=[_result(_SCALE, scores)]
    )
    assert not gate_passed(evaluate_gate(metrics))
    report = tmp_path / "report.md"
    raw = tmp_path / "raw.json"
    runner._write_reports([metrics], str(report), str(raw))
    assert "❌ FAIL" in report.read_text()
    data = json.loads(raw.read_text(), parse_constant=lambda invalid: pytest.fail(invalid))
    assert _SCALE in data["run_metadata"][f"{_BACKEND}:en"]["scenario_errors"]
    assert metric not in data["backends"][f"{_BACKEND}:en"][_SCALE]["judge_scores"]


@pytest.mark.parametrize("backend", ["mem0_multi_agent", "ai_knot_other_backend"])
def test_scorecard_never_substitutes_another_backend_for_ai_knot(backend: str) -> None:
    metrics = BenchmarkMetrics(
        backend,
        scenario_results=[_result("s10_ma_mesi_cas", {"cas_correctness": [1.0]})],
    )
    scorecard = "\n".join(_ma_scorecard([metrics]))
    assert "no ai-knot multi-agent verification" in scorecard.lower()
    assert "✅ PASS" not in scorecard
    assert backend not in scorecard
    assert backend in render_markdown([metrics])
