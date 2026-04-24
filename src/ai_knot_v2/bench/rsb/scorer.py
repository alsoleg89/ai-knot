"""RSB v1 — Reliability Scenario Bench scorer.

Runs RSB scenarios through MemoryAPI and scores each question.
Scoring: binary pass/fail per question + aggregate pass rate.

Sprint 15-17.
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any

from ai_knot_v2.api.product import MemoryAPI
from ai_knot_v2.api.sdk import EpisodeIn, LearnRequest, RecallRequest
from ai_knot_v2.bench.rsb.generator import RSBQuestion, RSBScenario, load_scenarios

_RSB_MAX_ATOMS = 100
_RSB_MAX_TOKENS = 8000


@dataclasses.dataclass(frozen=True, slots=True)
class QuestionResult:
    question: str
    passed: bool
    atoms_retrieved: int
    matched_objects: tuple[str, ...]
    missing_objects: tuple[str, ...]
    forbidden_found: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class ScenarioResult:
    name: str
    domain: str
    total_atoms: int
    question_results: tuple[QuestionResult, ...]

    @property
    def pass_rate(self) -> float:
        if not self.question_results:
            return 0.0
        return sum(1 for q in self.question_results if q.passed) / len(self.question_results)

    @property
    def passed(self) -> bool:
        return self.pass_rate == 1.0


def _score_question(
    q: RSBQuestion,
    recalled_atoms: list[Any],
) -> QuestionResult:
    """Check if recalled atoms satisfy the question requirements."""
    # Match against object_value + subject + predicate (predicate covers verb-like facts)
    atom_texts = [
        " ".join(
            [
                (a.object_value or "").lower(),
                (a.subject or "").lower(),
                a.predicate.replace("_", " "),
            ]
        )
        for a in recalled_atoms
    ]
    matched_objects: list[str] = []
    missing_objects: list[str] = []
    forbidden_found: list[str] = []

    # Check expected objects: at least one atom must contain each expected string
    for expected in q.expected_objects:
        found = any(expected.lower() in txt for txt in atom_texts)
        if found:
            matched_objects.append(expected)
        else:
            missing_objects.append(expected)

    # Check forbidden: no atom should contain these strings
    for forbidden in q.must_not_recall:
        if any(forbidden.lower() in txt for txt in atom_texts):
            forbidden_found.append(forbidden)

    # Pass if: all expected objects found, no forbidden objects found
    passed = not missing_objects and not forbidden_found

    return QuestionResult(
        question=q.text,
        passed=passed,
        atoms_retrieved=len(recalled_atoms),
        matched_objects=tuple(matched_objects),
        missing_objects=tuple(missing_objects),
        forbidden_found=tuple(forbidden_found),
    )


def run_scenario(scenario: RSBScenario) -> ScenarioResult:
    """Run a single RSB scenario through MemoryAPI and score it."""
    api = MemoryAPI(db_path=":memory:")

    episodes = [
        EpisodeIn(
            text=turn.text,
            speaker="user",
            user_id=scenario.user_id,
            session_id=turn.session_id,
            timestamp=turn.timestamp,
        )
        for turn in scenario.turns
    ]
    api.learn(LearnRequest(episodes=episodes))

    all_atoms = api._library.all_atoms()  # noqa: SLF001
    total_atoms = len(all_atoms)

    question_results: list[QuestionResult] = []
    for q in scenario.questions:
        recall_resp = api.recall(
            RecallRequest(
                query=q.text,
                max_atoms=_RSB_MAX_ATOMS,
                max_tokens=_RSB_MAX_TOKENS,
            )
        )
        recalled = recall_resp.atoms
        question_results.append(_score_question(q, recalled))

    return ScenarioResult(
        name=scenario.name,
        domain=scenario.domain,
        total_atoms=total_atoms,
        question_results=tuple(question_results),
    )


def run_rsb(
    domain: str | None = None,
    names: list[str] | None = None,
) -> list[ScenarioResult]:
    """Run all RSB scenarios (or filtered subset) and return results."""
    scenarios = load_scenarios(domain=domain, names=names)
    return [run_scenario(s) for s in scenarios]


def report_rsb(results: list[ScenarioResult]) -> None:
    """Print RSB scorecard to stdout."""
    print("\n=== RSB v1 — Reliability Scenario Bench ===\n")
    print(f"{'name':>10}  {'domain':>12}  {'atoms':>6}  {'pass_rate':>9}  status")
    print("-" * 55)

    total_q = 0
    total_pass = 0

    for sr in results:
        n_q = len(sr.question_results)
        n_pass = sum(1 for q in sr.question_results if q.passed)
        total_q += n_q
        total_pass += n_pass
        status = "PASS" if sr.passed else "FAIL"
        print(
            f"{sr.name:>10}  {sr.domain:>12}  {sr.total_atoms:>6}  {sr.pass_rate:>9.1%}  {status}"
        )
        for qr in sr.question_results:
            q_status = "✓" if qr.passed else "✗"
            print(f"    {q_status} {qr.question[:70]}")
            if qr.missing_objects:
                print(f"      missing: {qr.missing_objects}")
            if qr.forbidden_found:
                print(f"      forbidden found: {qr.forbidden_found}")

    print("-" * 55)
    overall_rate = total_pass / total_q if total_q else 0.0
    print(f"\nOverall pass rate: {overall_rate:.1%} ({total_pass}/{total_q})")
    gate = "PASS" if overall_rate >= 0.80 else "FAIL"
    print(f"RSB gate (≥ 80%): {gate}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="RSB v1 reliability scenario bench")
    parser.add_argument("--domain", type=str, default=None, help="Filter by domain")
    parser.add_argument("--names", type=str, default=None, help="Comma-separated scenario names")
    args = parser.parse_args()

    names = args.names.split(",") if args.names else None
    results = run_rsb(domain=args.domain, names=names)
    report_rsb(results)

    overall_rate = sum(1 for r in results for q in r.question_results if q.passed) / max(
        1, sum(len(r.question_results) for r in results)
    )
    if overall_rate < 0.80:
        sys.exit(1)


if __name__ == "__main__":
    main()
