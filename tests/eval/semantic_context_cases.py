"""Editorial held-out cases; authored and frozen before any retrieval/model run.

All records belong in ``__shared__``. ``read_scopes`` contains visibility grants
for READER; task applicability, source support, active dates and lineage still
apply. Gold groups are ANDed, with alternative source IDs ORed inside a group.
Choices are public decisions, and ABSTAIN means the readable current sources do
not establish a listed action. No source text contains evaluation annotations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from ai_knot.always_on.codec import canonical_json
from ai_knot.always_on.lineage import derivation_qualifiers
from ai_knot.types import Fact

READER = "semantic-eval"
AS_OF = datetime(2026, 10, 28, 12, tzinfo=UTC)
STAMP = datetime(2026, 10, 26, 9, tzinfo=UTC)
CHANGED = datetime(2026, 10, 27, 10, tzinfo=UTC)
RULE = (
    "Use supported, current source records applicable to this task and readable "
    "with the supplied scopes. Select ABSTAIN if they do not establish a listed action."
)
ABSTAIN = ("ABSTAIN", "Insufficient readable current evidence.")


@dataclass(frozen=True)
class Case:
    id: str
    query: str
    task_key: str
    facts: tuple[Fact, ...]
    read_scopes: tuple[str, ...]
    required_groups: tuple[tuple[str, ...], ...]
    choices: tuple[tuple[str, str], ...]
    expected_choice: str


def _fact(
    fact_id: str,
    task: str,
    text: str,
    *,
    scope: str = "global",
    supported: bool = True,
    slot: str = "",
    version: int = 1,
) -> Fact:
    return Fact(
        id=fact_id,
        content=text,
        source_snippets=[text],
        source_spans=[f"fixture://source/{fact_id}#p1"],
        created_at=STAMP,
        last_accessed=STAMP,
        valid_from=STAMP,
        event_time=STAMP,
        version=version,
        supported=supported,
        origin_agent_id="source-editor",
        visibility="public",
        visibility_scope=scope,
        memory_tier="pool",
        qualifiers={"task_key": task} if task else {},
        slot_key=slot,
        value_text=text if slot else "",
    )


def _stale_copy(fact_id: str, current: Fact, old_text: str) -> Fact:
    previous = replace(
        current,
        content=old_text,
        source_snippets=[old_text],
        value_text=old_text,
        version=current.version - 1,
        valid_from=STAMP,
    )
    return replace(
        _fact(fact_id, current.qualifiers["task_key"], old_text),
        verification_source="aggregate",
        qualifiers={**current.qualifiers, **derivation_qualifiers("__shared__", [previous])},
    )


def _topic_records(prefix: str, task: str, texts: tuple[str, ...]) -> tuple[Fact, ...]:
    return tuple(_fact(f"{prefix}{number:02x}", task, text) for number, text in enumerate(texts, 1))


MAINTENANCE = (
    (
        _fact(
            "a1000001",
            "chiller-rotor",
            "Before breaking the coolant circuit, drain trapped "
            "pressure and padlock the inlet valve.",
        ),
        _fact(
            "a1000002",
            "chiller-rotor",
            "Opening the fluid loop requires zero stored pressure and a locked supply gate.",
        ),
        _fact(
            "a1000003",
            "chiller-window",
            "A shutdown exceeding ninety minutes requires "
            "the duty engineer Mira to sign the work permit.",
            scope="maintenance",
        ),
        _fact(
            "a1000004",
            "chiller-renewal",
            "Осмотр теплообменника завершён; инженер поставил "
            "подпись в акте технического состояния.",
        ),
        _fact(
            "a1000005",
            "chiller-renewal",
            "Запасной комплект уплотнений доставлен на склад и выдан ремонтной бригаде.",
        ),
        _fact(
            "a1000006",
            "chiller-solvent",
            "Use only solvent K17 to flush the seal housing.",
            scope="restricted-maintenance",
        ),
        _fact("a1000007", "chiller-solvent", "The replacement identification plate has arrived."),
        _fact(
            "a1000008",
            "chiller-training",
            "A shutdown exceeding ninety minutes requires "
            "the trainer Pavel to sign the work permit.",
        ),
    )
    + _topic_records(
        "a11000",
        "chiller-rotor",
        (
            "The replacement rotating assembly weighs forty kilograms and has a dedicated cradle.",
            "Housing fasteners require eighty newton-metres during final reassembly.",
            "The outer casing coating must cure for six hours after the repair has finished.",
        ),
    )
    + _topic_records(
        "a12000",
        "chiller-window",
        (
            "Brief visual checks take fifteen minutes and require no extended outage permit.",
            "Replacement consumables for this repair are collected from the north storeroom.",
            "The outage calendar displays local time; the workshop wall clock displays UTC.",
        ),
    )
)

QUEUE = (
    (
        _fact(
            "b2000001",
            "parcel-concurrency",
            "At most half a dozen consignments may be in flight on the sorting lane at once.",
        ),
        _fact(
            "b2000002",
            "parcel-replay",
            "A durable acceptance receipt makes a consignment "
            "terminal. Never enqueue it again, even after the connection drops.",
        ),
        _fact(
            "b2000003",
            "parcel-damage",
            "Keep a torn consignment in quarantine until its "
            "stored fingerprint agrees with the newly computed digest.",
            scope="parcel-ops",
        ),
        _fact(
            "b2000004",
            "parcel-demo",
            "A torn consignment can enter the next sorting stage "
            "immediately; no fingerprint check is needed for this demonstration.",
        ),
        _fact(
            "b2000005",
            "parcel-recovery",
            "Recover lost delivery receipts by running the repair-receipts command.",
            supported=False,
        ),
        _fact("b2000006", "", "The sorting dashboard refreshes its status light every minute."),
        _fact(
            "b2000007",
            "parcel-concurrency",
            "At most twelve consignments may be in flight on the sorting lane at once.",
            scope="queue-lab",
        ),
    )
    + _topic_records(
        "b21000",
        "parcel-concurrency",
        (
            "The conveyor contains twenty-four physical bins, including empty return containers.",
            "Priority labels have three colours; the colour does not change handling sequence.",
            "The lane supervisor records completed consignment totals at the end of every hour.",
        ),
    )
    + _topic_records(
        "b22000",
        "parcel-damage",
        (
            "Fingerprint comparison results are retained for thirty days in the audit archive.",
            "The digest display uses hexadecimal notation and truncates only its visual preview.",
            "Photographs of damaged packaging are attached to the carrier claim after inspection.",
        ),
    )
)

_route = replace(
    _fact(
        "c3000003",
        "db-routing",
        "The primary write endpoint has moved to the western "
        "cluster. Direct all new mutations there.",
        slot="ledger::write-route",
        version=2,
    ),
    valid_from=CHANGED,
    created_at=CHANGED,
)
INCIDENT = (
    (
        _fact(
            "c3000001",
            "db-pressure",
            "When replica lag outruns the safe margin, stop "
            "accepting mutations; reads may continue while followers catch up.",
        ),
        _fact(
            "c3000002",
            "db-restart",
            "После трёх неудачных запусков автоматика блокируется. "
            "Дальнейшее восстановление выполняет дежурный вручную.",
            scope="incident-duty",
        ),
        _route,
        replace(
            _fact(
                "c3000004",
                "db-routing",
                "The primary write endpoint is the eastern cluster.",
                slot="ledger::write-route",
            ),
            valid_until=CHANGED,
        ),
        _stale_copy("c3000005", _route, "Keep sending all new mutations to the eastern cluster."),
        _fact(
            "c3000006",
            "db-threshold",
            "The alarm log records recurring storage timeouts; "
            "the numeric activation limit has not been recorded.",
        ),
        _fact(
            "c3000007",
            "db-sandbox",
            "Enable the storage-timeout alarm at seven events per minute in the sandbox exercise.",
        ),
    )
    + _topic_records(
        "c31000",
        "db-pressure",
        (
            "The replica-lag dashboard refreshes every fifteen seconds during an incident.",
            "Follower hostnames identify their geographical cell rather than their current rank.",
            "Request audit records remain available for ninety days after the incident is closed.",
        ),
    )
    + _topic_records(
        "c32000",
        "db-restart",
        (
            "После успешного восстановления журнал запуска архивируется на отдельном узле.",
            "Вне аварии штатный запуск сервиса занимает приблизительно сорок секунд.",
            "Номер текущей сборки отображается в нижней строке консоли дежурной смены.",
        ),
    )
)

_release = replace(
    _fact(
        "d4000003",
        "atlas-gate",
        "The signing fault has been fixed and the release "
        "manager has lifted the distribution embargo.",
        slot="atlas::gate",
        version=2,
    ),
    valid_from=CHANGED,
    created_at=CHANGED,
)
RELEASE = (
    (
        _fact(
            "d4000001",
            "atlas-canary",
            "In the observation window, 997 of every thousand requests completed successfully.",
        ),
        _fact(
            "d4000002",
            "atlas-canary",
            "Both geographical cells passed their readiness probes and are serving normally.",
        ),
        _release,
        _fact(
            "d4000004",
            "atlas-notice",
            "Send a customer bulletin for cryptographic "
            "compatibility changes. Cosmetic wording edits do not require one.",
            scope="release-comms",
        ),
        replace(
            _fact(
                "d4000005",
                "atlas-gate",
                "Distribution is embargoed because package signatures are invalid.",
                slot="atlas::gate",
            ),
            valid_until=CHANGED,
        ),
        _stale_copy("d4000006", _release, "Distribution remains embargoed until signatures work."),
        replace(
            _fact(
                "d4000007",
                "atlas-attestation",
                "The vulnerability attestation for artifact 8.4 is approved.",
            ),
            valid_until=CHANGED,
        ),
        _fact(
            "d4000008",
            "atlas-attestation",
            "Artifact 8.4 has been uploaded to staging; "
            "no new vulnerability attestation is recorded after the old approval ended.",
        ),
    )
    + _topic_records(
        "d41000",
        "atlas-canary",
        (
            "The observation dashboard retains per-request traces for twenty-four hours.",
            "The release archive contains one thousand signed packages from prior versions.",
            "Readiness probes use the internal health endpoint and do not send customer traffic.",
        ),
    )
    + _topic_records(
        "d42000",
        "atlas-notice",
        (
            "Customer bulletins are translated into the four languages offered by the product.",
            "The release notes editor wraps interface labels in quotation marks for readability.",
            "The distribution list is refreshed weekly by the customer operations team.",
        ),
    )
)

CASES = (
    Case(
        "maintenance-isolation",
        RULE + " A technician will unbolt the impeller housing. Which preparation is established?",
        "chiller-rotor",
        MAINTENANCE,
        (),
        (("a1000001", "a1000002"),),
        (
            ("A", "Bleed the circuit and secure its supply shut."),
            ("B", "Keep the circuit pressurized during disassembly."),
            ABSTAIN,
        ),
        "A",
    ),
    Case(
        "maintenance-permit-ru",
        RULE + " Окно ремонта продлится два часа. Кто должен разрешить такую остановку?",
        "chiller-window",
        MAINTENANCE,
        ("maintenance",),
        (("a1000003",),),
        (("A", "Pavel"), ("B", "Mira"), ABSTAIN),
        "B",
    ),
    Case(
        "maintenance-prerequisites-ru",
        RULE + " Разрешить замену можно, только если "
        "есть подписанный результат проверки и материалы для герметизации. Что делать?",
        "chiller-renewal",
        MAINTENANCE,
        (),
        (("a1000004",), ("a1000005",)),
        (
            ("A", "Отложить: хотя бы одно условие не выполнено."),
            ("B", "Разрешить: оба условия выполнены."),
            ABSTAIN,
        ),
        "B",
    ),
    Case(
        "maintenance-unreadable",
        RULE + " Which chemical is authorized for washing the seal enclosure?",
        "chiller-solvent",
        MAINTENANCE,
        (),
        (),
        (("A", "Use K17."), ("B", "Use plain water."), ABSTAIN),
        "ABSTAIN",
    ),
    Case(
        "queue-capacity",
        RULE + " What ceiling should the dispatcher place on simultaneously unfinished work?",
        "parcel-concurrency",
        QUEUE,
        (),
        (("b2000001",),),
        (("A", "Twelve."), ("B", "Six."), ABSTAIN),
        "B",
    ),
    Case(
        "queue-idempotence-ru",
        RULE + " Приём подтверждён сохранённой квитанцией, "
        "но связь оборвалась. Нужно ли повторно отправить ту же посылку в обработку?",
        "parcel-replay",
        QUEUE,
        (),
        (("b2000002",),),
        (("A", "Нет, считать приём завершённым."), ("B", "Да, поставить ещё раз."), ABSTAIN),
        "A",
    ),
    Case(
        "queue-task-boundary",
        RULE + " A package arrived ripped. Should its next "
        "processing step start now or wait for an integrity comparison?",
        "parcel-damage",
        QUEUE,
        ("parcel-ops",),
        (("b2000003",),),
        (
            ("A", "Start now without validation."),
            ("B", "Wait until the recorded and recomputed checksums match."),
            ABSTAIN,
        ),
        "B",
    ),
    Case(
        "queue-unsupported",
        RULE + " Which approved procedure reconstructs missing acknowledgements?",
        "parcel-recovery",
        QUEUE,
        (),
        (),
        (("A", "Run repair-receipts."), ("B", "Delete the queue."), ABSTAIN),
        "ABSTAIN",
    ),
    Case(
        "incident-load-shedding",
        RULE + " Followers cannot keep pace with updates. "
        "Which immediate service restriction is supported?",
        "db-pressure",
        INCIDENT,
        (),
        (("c3000001",),),
        (
            ("A", "Disable reads but continue writes."),
            ("B", "Pause writes while keeping reads available."),
            ABSTAIN,
        ),
        "B",
    ),
    Case(
        "incident-manual-recovery-ru",
        RULE + " Сервис уже трижды не смог подняться. Кому передать следующий шаг?",
        "db-restart",
        INCIDENT,
        ("incident-duty",),
        (("c3000002",),),
        (
            ("A", "Дежурному оператору для ручного восстановления."),
            ("B", "Автоматике для бесконечных повторов."),
            ABSTAIN,
        ),
        "A",
    ),
    Case(
        "incident-source-update",
        RULE + " Where should the client send fresh state-changing operations now?",
        "db-routing",
        INCIDENT,
        (),
        (("c3000003",),),
        (("A", "The eastern cluster."), ("B", "The western cluster."), ABSTAIN),
        "B",
    ),
    Case(
        "incident-unrecorded-limit",
        RULE + " At how many storage errors in sixty seconds should the production alert activate?",
        "db-threshold",
        INCIDENT,
        (),
        (),
        (("A", "Seven."), ("B", "Ten."), ABSTAIN),
        "ABSTAIN",
    ),
    Case(
        "release-two-prerequisites",
        RULE + " Promote only when failures are below "
        "one percent and both regions are healthy. Which action follows?",
        "atlas-canary",
        RELEASE,
        (),
        (("d4000001",), ("d4000002",)),
        (
            ("A", "Promote: both conditions hold."),
            ("B", "Hold: at least one condition fails."),
            ABSTAIN,
        ),
        "A",
    ),
    Case(
        "release-notification-ru",
        RULE + " В патче исправлены только формулировки "
        "интерфейса. Нужна ли рассылка пользователям по этому правилу?",
        "atlas-notice",
        RELEASE,
        ("release-comms",),
        (("d4000004",),),
        (
            ("A", "Да, отправить уведомление."),
            ("B", "Нет, для такой правки оно не требуется."),
            ABSTAIN,
        ),
        "B",
    ),
    Case(
        "release-source-update",
        RULE + " Is shipping presently permitted after the package authentication repair?",
        "atlas-gate",
        RELEASE,
        (),
        (("d4000003",),),
        (("A", "Resume distribution."), ("B", "Keep distribution blocked."), ABSTAIN),
        "A",
    ),
    Case(
        "release-expired-approval",
        RULE + " Can artifact 8.4 ship on the strength of a currently valid security attestation?",
        "atlas-attestation",
        RELEASE,
        (),
        (),
        (
            ("A", "Yes, a current approval is recorded."),
            ("B", "No, a current rejection is recorded."),
            ABSTAIN,
        ),
        "ABSTAIN",
    ),
)


def manifest_json() -> str:
    return canonical_json({"version": 1, "reader": READER, "as_of": AS_OF, "cases": CASES})


def manifest_sha256() -> str:
    return hashlib.sha256(manifest_json().encode("utf-8")).hexdigest()
