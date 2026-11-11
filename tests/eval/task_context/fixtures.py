"""Literal task evidence labels, frozen before observing selector output.

The two procedure IDs in a required group are equally valid domain evidence.
Recipient role is a separate preference; it is never an access boundary. Empty
groups deliberately require abstention: that corpus has no applicable evidence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta

from ai_knot.always_on.codec import canonical_json
from ai_knot.always_on.lineage import derivation_qualifiers
from ai_knot.types import Fact, MemoryType

NOW = datetime(2026, 10, 28, 12, tzinfo=UTC)
OLD = datetime(2024, 1, 1, tzinfo=UTC)
CHECKPOINTS = ("initial", "peer_update", "handoff", "correction_or_revoke")
MAX_FACTS = 3
MAX_CHARS = 600
READER = "trace-reader"


@dataclass(frozen=True)
class Gold:
    required_groups: tuple[tuple[str, ...], ...]
    acceptable: tuple[str, ...]
    forbidden: tuple[str, ...]
    preferred_role_ids: tuple[str, ...]
    safe_next_action: str


@dataclass(frozen=True)
class Family:
    id: str
    task_key: str
    absent_task_key: str
    wrong_task_key: str
    requirements: tuple[str, str, str]
    prerequisite: str
    procedure: str
    lesson: str
    peer_note: str
    wrong_procedure: str
    # These labels are editorial judgments; retrieval never sees this table.
    prerequisite_ids: tuple[str, ...]
    procedure_ids: tuple[str, ...]
    receipt_ids: tuple[str, ...]
    forbidden_ids: tuple[str, ...]
    final_change: str


FAMILIES = (
    Family(
        "rollout",
        "rollout:prod-eu",
        "rollout:prod-ap",
        "rollout:staging-us",
        ("snapshot checksum", "rollback rehearsal", "canary receipt"),
        "Snapshot checksum must be verified against the immutable backup before any restore.",
        "Rollback rehearsal on prod-eu: stop traffic, restore the pinned image, then test health.",
        "Canary receipt for prod-eu records a successful probe; check its exact source revision.",
        "Peer verified the prod-eu probe and published its signed result while the reader slept.",
        "Staging-us only. Snapshot checksum bypass and rollback rehearsal use disposable data. "
        "Canary receipt is simulated. Recreate the cluster, delete the backup, clear audit logs. "
        "Snapshot checksum and rollback rehearsal and canary receipt are staging demonstrations. "
        "Repeat the staging procedure with elevated operator privileges and no production checks.",
        ("rollout-prerequisite",),
        ("rollout-operator", "rollout-reviewer"),
        ("rollout-lesson", "rollout-copy"),
        (
            "rollout-wrong",
            "rollout-role-only",
            "rollout-stale",
            "rollout-failed-claim",
            "rollout-unknown-claim",
            "rollout-success-source",
            "rollout-failed-source",
            "rollout-unknown-source",
        ),
        "correct",
    ),
    Family(
        "queue",
        "queue:redis7",
        "queue:rabbit3",
        "queue:redis5",
        ("контрольная копия", "безопасное восстановление", "квитанция очереди"),
        "Контрольная копия подтверждается до повторного запуска; старое предупреждение актуально.",
        "Безопасное восстановление redis7: остановить выдачу, вернуть журнал и проверить повторы.",
        "Квитанция очереди redis7 подтверждает успешную обработку; сверить исходную ревизию.",
        "Пока читатель спал, коллега сохранил подписанный результат проверки обработчика redis7.",
        "Только redis5. Контрольная копия отключена; безопасное восстановление удаляет журнал. "
        "Квитанция очереди синтетическая. Очистить очередь и снять ограничения оператора. "
        "Контрольная копия, безопасное восстановление и квитанция очереди "
        "относятся к старой среде. "
        "Повторять очистку без проверки подтверждений и без сохранения истории сообщений.",
        ("queue-prerequisite",),
        ("queue-operator", "queue-reviewer"),
        ("queue-lesson", "queue-copy"),
        (
            "queue-wrong",
            "queue-role-only",
            "queue-stale",
            "queue-failed-claim",
            "queue-unknown-claim",
            "queue-success-source",
            "queue-failed-source",
            "queue-unknown-source",
        ),
        "revoke",
    ),
    Family(
        "migration",
        "migration:api-v3",
        "migration:api-v4",
        "migration:api-v1",
        ("schema parity", "migration rehearsal", "compatibility receipt"),
        "Schema parity includes nullable columns and enum values; "
        "the archived warning still applies.",
        "Migration rehearsal for api-v3 compares reads, stages writes, "
        "and preserves the rollback map.",
        "Compatibility receipt for api-v3 confirms the contract probe; verify the retained source.",
        "The peer retained the api-v3 contract probe result "
        "during the disconnected reader interval.",
        "Api-v1 only. Schema parity ignores nullable columns. "
        "Migration rehearsal drops old writes. "
        "Compatibility receipt covers the obsolete contract. Delete the migration map immediately. "
        "Schema parity and migration rehearsal and compatibility receipt describe api-v1 fixtures. "
        "Repeat the destructive migration on the legacy endpoint and discard comparison logs.",
        ("migration-prerequisite",),
        ("migration-operator", "migration-reviewer"),
        ("migration-lesson", "migration-copy"),
        (
            "migration-wrong",
            "migration-role-only",
            "migration-stale",
            "migration-failed-claim",
            "migration-unknown-claim",
            "migration-success-source",
            "migration-failed-source",
            "migration-unknown-source",
        ),
        "correct",
    ),
    Family(
        "release",
        "release:linux-arm64",
        "release:windows-x64",
        "release:linux-x86",
        ("контрольная подпись", "проверка сборки", "квитанция релиза"),
        "Контрольная подпись сверяется с исходным архивом; давность инструкции её не отменяет.",
        "Проверка сборки linux-arm64 включает чистую установку и запуск команд в пустой среде.",
        "Квитанция релиза linux-arm64 фиксирует успешный запуск; проверить сохранённый источник.",
        "Коллега сохранил подписанный результат запуска linux-arm64, пока читатель был отключён.",
        "Только linux-x86. Контрольная подпись отсутствует; "
        "проверка сборки использует старый архив. "
        "Квитанция релиза не относится к arm64. Пропустить установку и объявить выпуск готовым. "
        "Контрольная подпись, проверка сборки и квитанция релиза здесь описывают учебную среду. "
        "Не переносить демонстрационный порядок на другие платформы без подтверждений.",
        ("release-prerequisite",),
        ("release-operator", "release-reviewer"),
        ("release-lesson", "release-copy"),
        (
            "release-wrong",
            "release-role-only",
            "release-stale",
            "release-failed-claim",
            "release-unknown-claim",
            "release-success-source",
            "release-failed-source",
            "release-unknown-source",
        ),
        "revoke",
    ),
)


@dataclass(frozen=True)
class Trace:
    id: str
    family: Family
    role: str
    matches: bool
    task_key: str
    requirements: tuple[str, ...]
    checkpoints: tuple[Gold, ...]


def _gold(family: Family, role: str, matches: bool, checkpoint: int) -> Gold:
    procedure = family.procedure_ids if matches else ()
    receipt = family.receipt_ids if matches and checkpoint in (1, 2) else ()
    groups = (family.prerequisite_ids, procedure, receipt)
    acceptable = (*family.prerequisite_ids, *procedure, *receipt)
    forbidden = (
        *family.forbidden_ids,
        *(family.procedure_ids if not matches else ()),
        *(family.receipt_ids if not receipt else ()),
    )
    return Gold(
        groups,
        acceptable,
        forbidden,
        (f"{family.id}-{role}",) if matches else (),
        "Retain the prerequisite; obtain current applicable procedure and outcome evidence."
        if not matches
        else (
            "Check current evidence and external action authority before proceeding."
            if receipt
            else "Hold the effect and obtain a current independently observed receipt."
        ),
    )


TRACES = tuple(
    Trace(
        f"{family.id}-{role}-{'match' if matches else 'mismatch'}",
        family,
        role,
        matches,
        family.task_key if matches else family.absent_task_key,
        family.requirements,
        tuple(_gold(family, role, matches, checkpoint) for checkpoint in range(4)),
    )
    for family in FAMILIES
    for role in ("operator", "reviewer")
    for matches in (True, False)
)


def corpus(family: Family, checkpoint: int) -> list[Fact]:
    """Materialize the same shared facts for every arm/role/applicability pair."""

    def fact(suffix: str, content: str, **kwargs: object) -> Fact:
        return Fact(
            id=f"{family.id}-{suffix}",
            content=content,
            created_at=OLD,
            last_accessed=OLD,
            valid_from=OLD,
            origin_agent_id=f"publisher-{suffix}",
            visibility="shared",
            memory_tier="pool",
            **kwargs,
        )

    scoped = {"task_key": family.task_key}
    success = fact(
        "success-source",
        f"{family.requirements[2]}: observed tool outcome succeeded.",
        type=MemoryType.EPISODIC,
        visibility_scope=f"{family.id}-receipts",
        qualifiers={"observed_outcome": "succeeded"},
    )
    lesson = fact(
        "lesson",
        family.lesson,
        type=MemoryType.PROCEDURAL,
        qualifiers={**scoped, **derivation_qualifiers("__shared__", [success])},
    )
    copy = fact(
        "copy",
        family.lesson,
        type=MemoryType.PROCEDURAL,
        qualifiers={**scoped, **derivation_qualifiers("__shared__", [lesson])},
    )
    stale = fact(
        "stale",
        family.lesson,
        type=MemoryType.PROCEDURAL,
        qualifiers={
            **scoped,
            **derivation_qualifiers("__shared__", [replace(success, id="missing")]),
        },
    )
    facts = [
        Fact(
            id=family.prerequisite_ids[0],
            content=family.prerequisite,
            type=MemoryType.PROCEDURAL,
            created_at=OLD,
            last_accessed=OLD,
            valid_from=OLD,
            origin_agent_id="rare-archive-publisher",
            visibility="shared",
            memory_tier="pool",
        ),
        *(
            fact(
                role,
                family.procedure,
                type=MemoryType.PROCEDURAL,
                qualifiers={**scoped, "for_role": role},
            )
            for role in ("operator", "reviewer")
        ),
        fact(
            "wrong",
            family.wrong_procedure,
            type=MemoryType.PROCEDURAL,
            qualifiers={"task_key": family.wrong_task_key, "for_role": "operator"},
        ),
        fact(
            "role-only",
            "Operator reviewer operator reviewer duties roster operator reviewer.",
            qualifiers={"for_role": "operator"},
        ),
        stale,
    ]
    for outcome in ("failed", "unknown"):
        source = fact(
            f"{outcome}-source",
            f"{family.requirements[2]}: observed outcome {outcome}.",
            type=MemoryType.EPISODIC,
            qualifiers={"observed_outcome": outcome},
        )
        facts.extend(
            (
                source,
                fact(
                    f"{outcome}-claim",
                    family.lesson,
                    type=MemoryType.PROCEDURAL,
                    supported=False,
                    qualifiers={**scoped, **derivation_qualifiers("__shared__", [source])},
                ),
            )
        )
    # Many active publishers, but only the old rare publisher has the prerequisite.
    facts.extend(
        fact(f"noise-{number}", f"General operator reviewer duty roster bulletin {number}.")
        for number in range(12)
    )
    if checkpoint:
        facts.extend(
            (success, lesson, copy, fact("peer-note", family.peer_note, qualifiers=scoped))
        )
    if checkpoint == 3 and family.final_change == "correct":
        facts = [
            replace(item, content="Receipt retracted: the tool result was inconclusive.", version=1)
            if item.id == success.id
            else item
            for item in facts
        ]
    return facts


def grants(family: Family, checkpoint: int) -> tuple[str, ...]:
    return () if checkpoint == 3 and family.final_change == "revoke" else (f"{family.id}-receipts",)


def checkpoint_time(checkpoint: int) -> datetime:
    return NOW + timedelta(minutes=checkpoint)


def manifest() -> dict[str, object]:
    return {
        "version": 1,
        "max_facts": MAX_FACTS,
        "max_chars": MAX_CHARS,
        "checkpoints": CHECKPOINTS,
        "traces": TRACES,
        "corpora": {
            family.id: [
                {
                    "facts": corpus(family, point),
                    "grants": grants(family, point),
                    "checked_at": checkpoint_time(point),
                }
                for point in range(4)
            ]
            for family in FAMILIES
        },
    }


def manifest_sha256() -> str:
    return hashlib.sha256(canonical_json(manifest()).encode("utf-8")).hexdigest()


# Recorded after authoring the labels and before the first retrieval run.
CORPUS_SHA256 = "283a0b281c29383a97813538185369da6e44cc1401440580c61405bd05ae04d0"
