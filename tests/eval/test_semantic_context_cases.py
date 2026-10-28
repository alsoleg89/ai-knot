"""Model-free checks of a manifest frozen before retrieval or reader execution."""

from ai_knot.always_on.changes import access_projection, visible
from ai_knot.always_on.codec import loads_strict
from ai_knot.always_on.lineage import is_derived, lineage_status
from ai_knot.types import Fact

from .semantic_context_cases import (
    ABSTAIN,
    AS_OF,
    CASES,
    READER,
    RULE,
    Case,
    manifest_json,
    manifest_sha256,
)

# Literal frozen digest, established before any retriever or model was run.
EXPECTED_MANIFEST_SHA256 = "f1cdd8136013d62f4f83e94492d6faa2ee589c942871ce4481210430331907e4"
EXPECTED_CHOICES = {
    "maintenance-isolation": "A",
    "maintenance-permit-ru": "B",
    "maintenance-prerequisites-ru": "B",
    "maintenance-unreadable": "ABSTAIN",
    "queue-capacity": "B",
    "queue-idempotence-ru": "A",
    "queue-task-boundary": "B",
    "queue-unsupported": "ABSTAIN",
    "incident-load-shedding": "B",
    "incident-manual-recovery-ru": "A",
    "incident-source-update": "B",
    "incident-unrecorded-limit": "ABSTAIN",
    "release-two-prerequisites": "A",
    "release-notification-ru": "B",
    "release-source-update": "A",
    "release-expired-approval": "ABSTAIN",
}
CROWDED = {
    "maintenance-isolation",
    "maintenance-permit-ru",
    "queue-capacity",
    "queue-task-boundary",
    "incident-load-shedding",
    "incident-manual-recovery-ru",
    "release-two-prerequisites",
    "release-notification-ru",
}


def _exclusion(case: Case, fact: Fact) -> str:
    if fact.qualifiers.get("task_key", "") not in ("", case.task_key):
        return "other-task"
    if not visible(access_projection(fact), READER, set(case.read_scopes)):
        return "scope"
    if not fact.supported:
        return "unsupported"
    if not fact.is_active(AS_OF):
        return "expired"
    by_id = {item.id: item for item in case.facts}

    def load(namespace: str, fact_id: str) -> list[Fact]:
        return [by_id[fact_id]] if namespace == "__shared__" and fact_id in by_id else []

    if not all(lineage_status("__shared__", fact, READER, set(case.read_scopes), load)):
        return "lineage"
    return ""


def test_manifest_is_frozen_and_canonical() -> None:
    assert manifest_sha256() == EXPECTED_MANIFEST_SHA256
    manifest = loads_strict(manifest_json())
    assert manifest["version"] == 1
    assert len(manifest["cases"]) == 16
    assert manifest_json() == manifest_json()


def test_choices_and_gold_are_literal_and_well_formed() -> None:
    assert {case.id: case.expected_choice for case in CASES} == EXPECTED_CHOICES
    assert len({case.id for case in CASES}) == len({case.task_key for case in CASES}) == 16
    for case in CASES:
        assert case.query.startswith(RULE)
        assert tuple(label for label, _ in case.choices) == ("A", "B", "ABSTAIN")
        assert case.choices[-1] == ABSTAIN
        assert all(text.strip() for _, text in case.choices)
        assert (case.expected_choice == "ABSTAIN") == (not case.required_groups)
        assert all(group and len(group) == len(set(group)) for group in case.required_groups)


def test_gold_is_current_supported_readable_source_evidence() -> None:
    for case in CASES:
        by_id = {fact.id: fact for fact in case.facts}
        assert len(by_id) == len(case.facts)
        for group in case.required_groups:
            for fact_id in group:
                fact = by_id[fact_id]
                assert not _exclusion(case, fact), (case.id, fact_id)
                assert not is_derived(fact)
                assert fact.content in fact.source_snippets and fact.source_spans


def test_named_boundary_controls_remain_present() -> None:
    by_case = {case.id: case for case in CASES}
    controls = (
        ("maintenance-unreadable", "a1000006", "scope"),
        ("queue-unsupported", "b2000005", "unsupported"),
        ("queue-task-boundary", "b2000004", "other-task"),
        ("incident-unrecorded-limit", "c3000007", "other-task"),
        ("incident-source-update", "c3000004", "expired"),
        ("incident-source-update", "c3000005", "lineage"),
        ("release-source-update", "d4000006", "lineage"),
        ("release-expired-approval", "d4000007", "expired"),
    )
    for case_id, fact_id, reason in controls:
        case = by_case[case_id]
        fact = next(fact for fact in case.facts if fact.id == fact_id)
        assert _exclusion(case, fact) == reason
    assert sum(bool(case.read_scopes) for case in CASES) == 4


def test_budget_control_has_eight_compact_and_eight_crowded_cases() -> None:
    assert len(CROWDED) == len(CASES) - len(CROWDED) == 8
    for case in CASES:
        eligible = [fact for fact in case.facts if not _exclusion(case, fact)]
        if case.id in CROWDED:
            assert len(eligible) >= 4
            gold = {fact_id for group in case.required_groups for fact_id in group}
            assert sum(fact.id not in gold for fact in eligible) >= 3
        else:
            assert len(eligible) <= 3
            assert sum(len(fact.content) for fact in eligible) <= 600


def test_repeated_records_have_stable_ids_and_explicit_timestamps() -> None:
    seen = {}
    for case in CASES:
        for fact in case.facts:
            assert len(fact.id) == 8 and int(fact.id, 16) > 0
            assert fact.version >= 1
            assert fact.created_at == fact.last_accessed or fact.version == 2
            assert fact.created_at <= AS_OF and fact.valid_from <= AS_OF
            serialized = (fact.content, fact.created_at, fact.version, fact.qualifiers)
            assert seen.setdefault(fact.id, serialized) == serialized
