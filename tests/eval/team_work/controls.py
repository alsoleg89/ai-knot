"""Explicit offline Docker controls; no provider or credential port."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from tests.eval.team_repos.sandbox import DEFAULT_IMAGE, DockerSandbox, cleanup_run

from .corpus import load_corpus


def receipt_matches(receipt: dict, *, expected_pass: bool, expected_tests: int) -> bool:
    if (
        any(
            type(receipt.get(k)) is not int or receipt[k] < 0
            for k in ("tests_run", "failures", "errors")
        )
        or expected_tests < 1
        or receipt["tests_run"] != expected_tests
    ):
        return False
    if expected_pass:
        return receipt.get("status") == "pass" and receipt["failures"] == receipt["errors"] == 0
    return receipt.get("status") == "fail" and receipt["failures"] + receipt["errors"] > 0


def method_count(source: str) -> int:
    # Authored contracts use direct unittest.TestCase subclasses, not dynamic tests.
    return sum(
        isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef))
        and method.name.startswith("test_")
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef)
        and any(
            (isinstance(base, ast.Attribute) and base.attr == "TestCase")
            or (isinstance(base, ast.Name) and base.id == "TestCase")
            for base in node.bases
        )
        for method in node.body
    )


def run_controls(cases_path: Path, expected_sha256: str, output: Path) -> dict:
    if output.exists():
        raise ValueError("Refusing to overwrite control results")
    cases = load_corpus(cases_path, expected_sha256=expected_sha256)
    run_id = uuid4().hex
    report = {
        "cases_sha256": expected_sha256,
        "image": DEFAULT_IMAGE,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "run_id": run_id,
        "complete": False,
        "quality_interpretation_allowed": False,
        "provider_calls": 0,
        "rows": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temp = output.with_suffix(output.suffix + ".pending")
        temp.write_text(json.dumps(report, indent=2) + "\n")
        temp.replace(output)

    save()
    sandbox = DockerSandbox(run_id=run_id)
    try:
        for case in cases:
            for index, stage in enumerate(case.stages):
                expected_tests = method_count(stage.test_source)
                candidates = [
                    ("current_reference", stage.reference_solution, True),
                    ("initial_stub", case.initial_code, False),
                ]
                if index:
                    candidates.append(
                        ("previous_reference", case.stages[index - 1].reference_solution, False)
                    )
                for label, code, expected_pass in candidates:
                    row = {
                        "case_id": case.id,
                        "stage_id": stage.id,
                        "control": label,
                        "expected_pass": expected_pass,
                        "expected_tests": expected_tests,
                        "matched": False,
                        "receipt": None,
                    }
                    report["rows"].append(row)
                    save()
                    row["receipt"] = sandbox.run(code, stage.test_source)
                    row["matched"] = receipt_matches(
                        row["receipt"], expected_pass=expected_pass, expected_tests=expected_tests
                    )
                    save()
        report["complete"] = True
        report["all_matched"] = len(report["rows"]) == 24 and all(
            r["matched"] for r in report["rows"]
        )
        save()
        return report
    finally:
        cleanup_run(run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_controls(args.cases, args.sha256, args.output)
    print(
        json.dumps(
            {
                "complete": result["complete"],
                "all_matched": result["all_matched"],
                "matched": sum(row["matched"] for row in result["rows"]),
                "controls": len(result["rows"]),
                "provider_calls": 0,
            }
        )
    )
    raise SystemExit(0 if result["all_matched"] else 1)
