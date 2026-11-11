"""Offline integrity and executable-contract checks for the sealed repository suite."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import types
import unittest
from dataclasses import fields
from pathlib import PurePosixPath
from unittest.mock import patch

from .team_repos.cases import CASES, Case, Document, Stage, manifest_json, manifest_sha256

# Frozen before any provider, reader, retriever or evaluation run.
EXPECTED_MANIFEST_SHA256 = "148d2368d1c841fd5005879118b041d6d4ce7b9cfe574c8038a4c65af1ffc656"


def _run(source: str, test_source: str) -> unittest.TestResult:
    """Run only the trusted, locally authored fixture code, without external services."""
    solution = types.ModuleType("solution")
    exec(compile(source, "solution.py", "exec"), solution.__dict__)
    checks = types.ModuleType("authored_checks")
    with patch.dict(sys.modules, {"solution": solution}):
        exec(compile(test_source, "test_solution.py", "exec"), checks.__dict__)
        suite = unittest.defaultTestLoader.loadTestsFromModule(checks)
        result = unittest.TestResult()
        suite.run(result)
    return result


def _assertion_count(source: str) -> int:
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr.startswith("assert")
        for node in ast.walk(ast.parse(source))
    )


def _public_signature(source: str) -> tuple[str, str]:
    functions = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)]
    assert len(functions) == 1
    return functions[0].name, ast.dump(functions[0].args)


def test_manifest_is_canonical_and_frozen() -> None:
    assert manifest_sha256() == EXPECTED_MANIFEST_SHA256
    payload = manifest_json()
    assert payload == json.dumps(
        json.loads(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == EXPECTED_MANIFEST_SHA256
    assert json.loads(payload)["version"] == 1


def test_schema_and_suite_dimensions() -> None:
    assert tuple(field.name for field in fields(Document)) == ("path", "text", "task_key")
    assert tuple(field.name for field in fields(Stage)) == (
        "id",
        "brief",
        "documents",
        "test_source",
        "reference_solution",
    )
    assert tuple(field.name for field in fields(Case)) == (
        "id",
        "task_key",
        "title",
        "goal",
        "requirements",
        "initial_code",
        "stages",
    )
    assert len(CASES) == len({case.id for case in CASES}) == 3
    assert all(len(case.stages) == 3 for case in CASES)
    assert sum(len(stage.documents) for case in CASES for stage in case.stages) == 60
    assert (
        sum(_assertion_count(stage.test_source) for case in CASES for stage in case.stages) == 110
    )


def test_documents_are_compact_task_labelled_and_stably_addressed() -> None:
    for case in CASES:
        assert case.title.strip() and case.goal.strip() and case.requirements
        assert isinstance(case.task_key, str) and case.task_key

        assert len({stage.id for stage in case.stages}) == len(case.stages)
        previous_paths: set[str] | None = None
        previous_text: dict[str, str] = {}
        for stage in case.stages:
            assert stage.brief.strip()
            assert 6 <= len(stage.documents) <= 12
            assert sum(len(document.text) for document in stage.documents) <= 4000
            paths = {document.path for document in stage.documents}
            assert len(paths) == len(stage.documents)
            assert any(not document.task_key for document in stage.documents)
            labels = {document.task_key for document in stage.documents if document.task_key}
            assert len(labels) >= 2
            assert case.task_key in labels
            assert sum(document.task_key == case.task_key for document in stage.documents) >= 3
            for document in stage.documents:
                assert document.path.isascii() and document.text.strip()
                assert not PurePosixPath(document.path).is_absolute()
                assert ".." not in PurePosixPath(document.path).parts
                if document.task_key:
                    assert document.task_key in document.text
            if previous_paths is not None:
                assert paths.intersection(previous_paths)
                # Replacements and fresh peer updates have stable paths and changed text.
                assert (
                    sum(
                        previous_text.get(document.path) != document.text
                        for document in stage.documents
                    )
                    >= 4
                )
            previous_paths = paths
            previous_text = {document.path: document.text for document in stage.documents}
            assert 8 <= _assertion_count(stage.test_source) <= 20
            assert "unittest.main" not in stage.test_source
            assert _public_signature(stage.reference_solution) == _public_signature(
                case.initial_code
            )
            assert len(stage.reference_solution) <= 4000


def test_all_nine_references_satisfy_their_authored_checks() -> None:
    for case in CASES:
        for stage in case.stages:
            result = _run(stage.reference_solution, stage.test_source)
            assert result.testsRun == 2
            assert result.wasSuccessful(), (case.id, stage.id, result.errors, result.failures)


def test_initial_implementation_fails_each_stage_by_assertion() -> None:
    for case in CASES:
        for stage in case.stages:
            result = _run(case.initial_code, stage.test_source)
            assert result.testsRun == 2
            assert result.failures and not result.errors, (case.id, stage.id)


def test_each_revision_rejects_the_previous_reference_by_assertion() -> None:
    for case in CASES:
        for previous, current in zip(case.stages, case.stages[1:], strict=False):
            result = _run(previous.reference_solution, current.test_source)
            assert result.testsRun == 2
            assert result.failures and not result.errors, (case.id, current.id)
