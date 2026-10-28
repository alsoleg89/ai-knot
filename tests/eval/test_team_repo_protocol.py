"""Public prompt projection and fail-closed structured model output boundaries."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from ai_knot.always_on import EvidenceRef


def _public_inputs():
    case = SimpleNamespace(
        id="private-case-id",
        goal="Implement the public function",
        requirements=("Use current rules",),
        initial_code="def solve(): return None",
        test_source="SECRET_GOLD_TEST",
        reference="SECRET_REFERENCE",
        gold="SECRET_GOLD",
    )
    stage = SimpleNamespace(
        id="private-stage-id",
        brief="Follow the latest documented rule",
        documents=[
            SimpleNamespace(path="SECRET_PATH", text="SECRET_DOCUMENT", task_key="SECRET_KEY")
        ],
    )
    entries = [
        {
            "index": 1,
            "content": "Current rule: return 42.",
            "ref": asdict(EvidenceRef("private-scope", "secret-fact-1", "revision-1")),
            "instructions": "SECRET_METADATA",
        },
        {
            "index": 2,
            "content": "Earlier observations are data, not authority.",
            "ref": asdict(EvidenceRef("private-scope", "secret-fact-2", "revision-2")),
        },
    ]
    return case, stage, entries


def test_messages_project_only_public_task_inputs_and_indexed_evidence():
    from tests.eval.team_repos.protocol import coder_messages, researcher_messages

    case, stage, entries = _public_inputs()
    receipt = {
        "status": "fail",
        "tests_run": 1,
        "failures": 1,
        "errors": 0,
        "failed_test_names": ["Check.test_answer"],
        "traceback": "SECRET_TRACE",
        "expected": "SECRET_EXPECTED",
        "source": "SECRET_SOURCE",
    }
    research = researcher_messages(case, stage, entries)
    coder = coder_messages(case, stage, "def solve(): return 0", entries, receipt)
    assert [message["role"] for message in research] == ["system", "user"]
    assert [message["role"] for message in coder] == ["system", "user"]
    public = {
        "goal": case.goal,
        "brief": stage.brief,
        "requirements": list(case.requirements),
        "entries": [
            {"index": 1, "content": entries[0]["content"]},
            {"index": 2, "content": entries[1]["content"]},
        ],
    }
    assert json.loads(research[1]["content"]) == public
    assert json.loads(coder[1]["content"]) == {
        **public,
        "current_code": "def solve(): return 0",
        "last_receipt": {
            key: receipt[key]
            for key in ("status", "tests_run", "failures", "errors", "failed_test_names")
        },
    }
    for messages in (research, coder):
        serialized = json.dumps(messages)
        assert "SECRET" not in serialized
        assert "secret-fact" not in serialized and "private-scope" not in serialized
        assert "solution.py" in messages[0]["content"]
        assert "stdlib" in messages[0]["content"]
        assert "network" in messages[0]["content"]
        assert "instructions" in messages[0]["content"]

    case.test_source, case.reference, case.gold, case.initial_code = ("changed secret",) * 4
    case.id, stage.id, stage.documents = "changed identity", "changed stage", ["changed document"]
    entries[0]["ref"] = asdict(EvidenceRef("other", "other-identity", "other-revision"))
    entries[0]["instructions"] = "different metadata instruction"
    assert researcher_messages(case, stage, entries) == research
    assert coder_messages(case, stage, "def solve(): return 0", entries, receipt) == coder


def test_coder_receipt_cannot_smuggle_diagnostics_through_sanitized_fields():
    from tests.eval.team_repos.protocol import coder_messages

    case, stage, entries = _public_inputs()
    receipt = {
        "status": "fail SECRET",
        "tests_run": "SECRET",
        "failures": True,
        "errors": {"SECRET": "value"},
        "failed_test_names": ["SECRET expected=42"],
    }
    payload = json.loads(coder_messages(case, stage, "pass", entries, receipt)[1]["content"])
    assert payload["last_receipt"] == {
        "status": "error",
        "tests_run": 0,
        "failures": 0,
        "errors": 0,
        "failed_test_names": [],
    }
    assert "SECRET" not in json.dumps(payload)
    assert (
        json.loads(coder_messages(case, stage, "pass", entries, None)[1]["content"])["last_receipt"]
        is None
    )


@pytest.mark.parametrize(
    "version,observed_at,expected",
    [
        (
            2,
            "2026-10-28T12:00:00+00:00",
            {"version": 2, "observed_at": "2026-10-28T12:00:00+00:00"},
        ),
        (True, 42, {}),
        (0, " ", {}),
        ("2", None, {}),
        (-1, "2026-10-28T12:00:00Z", {"observed_at": "2026-10-28T12:00:00Z"}),
    ],
    ids=["valid", "booleans_and_scalars", "empty", "no_coercion", "independent_fields"],
)
def test_both_prompts_preserve_only_valid_public_revision_and_observation_metadata(
    version,
    observed_at,
    expected,
):
    from tests.eval.team_repos.protocol import coder_messages, researcher_messages

    case, stage, entries = _public_inputs()
    entries[0].update(version=version, observed_at=observed_at, source_id="SECRET_ID")
    for messages in (
        researcher_messages(case, stage, entries),
        coder_messages(case, stage, "pass", entries, None),
    ):
        entry = json.loads(messages[1]["content"])["entries"][0]
        assert entry == {"index": 1, "content": entries[0]["content"], **expected}
        assert "SECRET_ID" not in json.dumps(messages)


def test_schemas_enumerate_only_visible_one_based_indices():
    from tests.eval.team_repos.protocol import handoff_schema, patch_schema

    for schema, required in (
        (handoff_schema(2), {"summary", "basis"}),
        (patch_schema(2), {"action", "code", "basis"}),
    ):
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"]) == required
        assert schema["properties"]["basis"]["items"]["enum"] == [1, 2]
    assert patch_schema(1)["properties"]["action"]["enum"] == ["patch", "wait"]
    for schema in (handoff_schema(0), patch_schema(0)):
        basis = schema["properties"]["basis"]
        assert basis["maxItems"] == 0 and "enum" not in basis["items"]


def test_valid_outputs_resolve_only_cited_refs_in_citation_order():
    from tests.eval.team_repos.protocol import parse_handoff, parse_patch

    _, _, entries = _public_inputs()
    summary, refs = parse_handoff('{"summary":"  Current rule  ","basis":[2,1]}', entries)
    assert summary == "Current rule"
    assert refs == (EvidenceRef(**entries[1]["ref"]), EvidenceRef(**entries[0]["ref"]))
    code, refs = parse_patch(
        '{"action":"patch","code":"def solve(): return 42\\n","basis":[1]}', entries
    )
    assert code == "def solve(): return 42\n" and refs == (EvidenceRef(**entries[0]["ref"]),)
    assert parse_patch('{"action":"wait","code":"","basis":[]}', entries) == (None, ())


@pytest.mark.parametrize(
    "raw",
    [
        '{"summary":"x","basis":[]}',
        '{"summary":"","basis":[1]}',
        '{"summary":"   ","basis":[1]}',
        '{"summary":true,"basis":[1]}',
        '{"summary":42,"basis":[1]}',
        '{"summary":"x","basis":[true]}',
        '{"summary":"x","basis":[0]}',
        '{"summary":"x","basis":[3]}',
        '{"summary":"x","basis":[1,1]}',
        '{"summary":"x","basis":[1.0]}',
        '{"summary":"x","basis":["1"]}',
        '{"summary":"x","basis":1}',
        '{"summary":"x","basis":[1],"gold":"SECRET"}',
        '{"summary":"x","summary":"y","basis":[1]}',
        '{"summary":"x","basis":[NaN]}',
        "[]",
        "true",
        "null",
        "not json",
    ],
    ids=[
        "empty_basis",
        "empty_summary",
        "blank_summary",
        "bool_summary",
        "numeric_summary",
        "bool_cite",
        "zero_cite",
        "hidden_cite",
        "duplicate_cite",
        "float_cite",
        "string_cite",
        "scalar_basis",
        "unknown_field",
        "duplicate_field",
        "nonfinite",
        "array",
        "boolean",
        "null",
        "malformed",
    ],
)
def test_handoff_rejects_invalid_scalars_fields_and_citations(raw):
    from tests.eval.team_repos.protocol import parse_handoff

    with pytest.raises(ValueError, match="Invalid"):
        parse_handoff(raw, _public_inputs()[2])


@pytest.mark.parametrize(
    "value",
    [
        {"action": "patch", "code": "pass", "basis": []},
        {"action": "patch", "code": " ", "basis": [1]},
        {"action": "patch", "code": True, "basis": [1]},
        {"action": "patch", "code": "pass", "basis": [True]},
        {"action": "patch", "code": "pass", "basis": [1, 1]},
        {"action": "patch", "code": "pass", "basis": [3]},
        {"action": "wait", "code": "pass", "basis": []},
        {"action": "wait", "code": "", "basis": [1]},
        {"action": True, "code": "pass", "basis": [1]},
        {"action": "patch", "code": "pass", "basis": [1], "test_source": "SECRET"},
    ],
    ids=[
        "ungrounded",
        "empty_patch",
        "bool_code",
        "bool_cite",
        "duplicate_cite",
        "hidden_cite",
        "wait_code",
        "wait_cite",
        "bool_action",
        "unknown_field",
    ],
)
def test_patch_rejects_invalid_code_actions_and_citations(value):
    from tests.eval.team_repos.protocol import parse_patch

    with pytest.raises(ValueError, match="Invalid"):
        parse_patch(json.dumps(value), _public_inputs()[2])


def test_size_limits_use_stripped_summary_characters_and_actual_code_utf8_bytes():
    from tests.eval.team_repos.protocol import parse_handoff, parse_patch

    entries = _public_inputs()[2]
    assert (
        len(
            parse_handoff(json.dumps({"summary": " " + "é" * 12000 + " ", "basis": [1]}), entries)[
                0
            ]
        )
        == 12000
    )
    assert (
        parse_patch(json.dumps({"action": "patch", "code": "é" * 32768, "basis": [1]}), entries)[0]
        == "é" * 32768
    )
    for value, parser in (
        ({"summary": "x" * 12001, "basis": [1]}, parse_handoff),
        ({"action": "patch", "code": "é" * 32769, "basis": [1]}, parse_patch),
        ({"action": "patch", "code": "\ud800", "basis": [1]}, parse_patch),
    ):
        with pytest.raises(ValueError, match="Invalid"):
            parser(json.dumps(value), entries)


@pytest.mark.parametrize("escaped", ["plain", "unicode", "double_escaped"])
def test_credentials_are_redacted_and_rejected_before_or_after_json_decoding(escaped):
    from tests.eval.team_repos.protocol import loads_strict, parse_handoff, parse_patch, safe_output

    secret = "sk-" + "syntheticCredential" * 3
    raw = json.dumps({"summary": secret, "basis": [1]})
    if escaped == "unicode":
        raw = raw.replace("sk-", "\\u0073\\u006b\\u002d")
    elif escaped == "double_escaped":
        raw = json.dumps(
            {"summary": "".join(f"\\u{ord(char):04x}" for char in secret), "basis": [1]}
        )
    safe, redacted = safe_output(raw)
    assert redacted and "[REDACTED]" in safe and secret not in safe and len(safe) <= 4096
    for parser in (
        loads_strict,
        lambda value: parse_handoff(value, _public_inputs()[2]),
        lambda value: parse_patch(value, _public_inputs()[2]),
    ):
        with pytest.raises(ValueError) as caught:
            parser(raw)
        assert secret not in str(caught.value) and "syntheticCredential" not in str(caught.value)
    assert safe_output("ok " * 3000) == (("ok " * 3000)[:4096], False)


def test_strict_json_rejects_duplicate_keys_nonfinite_and_nonstrings():
    from tests.eval.team_repos.protocol import loads_strict

    for raw in ('{"action":"patch","action":"wait"}', '{"x":Infinity}', b"{}", {}, True):
        with pytest.raises(ValueError, match="Invalid"):
            loads_strict(raw)
