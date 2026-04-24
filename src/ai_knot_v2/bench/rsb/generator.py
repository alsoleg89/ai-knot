"""RSB v1 — Reliability Scenario Bench generator.

Generates synthetic conversations for medical and scheduling domains.
Each scenario tests a specific memory reliability property.

Sprint 15-17: 2 domains × 3 scenarios each = 6 scenarios minimum.
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class RSBTurn:
    speaker: str
    text: str
    session_id: str = "rsb-session-1"
    timestamp: int = 1_700_000_000


@dataclasses.dataclass(frozen=True, slots=True)
class RSBQuestion:
    text: str
    expected_predicates: tuple[str, ...]  # predicates that MUST appear in recalled atoms
    expected_objects: tuple[str, ...]  # object_value substrings that must appear
    must_not_recall: tuple[str, ...] = ()  # object_value substrings that must NOT appear


@dataclasses.dataclass(frozen=True, slots=True)
class RSBScenario:
    name: str
    domain: str
    description: str
    turns: tuple[RSBTurn, ...]
    questions: tuple[RSBQuestion, ...]
    user_id: str = "rsb-user"
    agent_id: str = "rsb-agent"


# ---------------------------------------------------------------------------
# Noise turns (filler conversation to test survival over many turns)
# ---------------------------------------------------------------------------

_NOISE_TURNS: list[tuple[str, str]] = [
    ("user", "The weather has been really nice lately."),
    ("user", "I watched a good movie last night."),
    ("user", "My neighbor got a new dog."),
    ("user", "I tried a new recipe for pasta."),
    ("user", "Work has been busy this week."),
    ("user", "I went for a walk in the park."),
    ("user", "My friend called me yesterday."),
    ("user", "I finished reading a book about history."),
    ("user", "The kids had a school event today."),
    ("user", "I ordered some books online."),
]


def _noise(session_id: str = "rsb-session-1", base_ts: int = 1_700_000_000) -> list[RSBTurn]:
    return [
        RSBTurn(speaker=t[0], text=t[1], session_id=session_id, timestamp=base_ts + i * 60)
        for i, t in enumerate(_NOISE_TURNS)
    ]


# ---------------------------------------------------------------------------
# Medical domain scenarios
# ---------------------------------------------------------------------------


def _medical_scenario_1() -> RSBScenario:
    """RSB-M1: Critical medication must survive over many turns."""
    early_turns = [
        RSBTurn(
            "user",
            "I take aspirin 325mg daily for my heart condition.",
            timestamp=1_700_000_000,
        ),
        RSBTurn(
            "user",
            "My cardiologist prescribed it last year.",
            timestamp=1_700_000_060,
        ),
    ]
    noise = _noise(base_ts=1_700_001_000)
    return RSBScenario(
        name="RSB-M1",
        domain="medical",
        description="Critical medication fact must be recalled after many unrelated turns",
        turns=tuple(early_turns + noise),
        questions=(
            RSBQuestion(
                text="What medication do I take daily?",
                expected_predicates=("prefers", "has", "is", "takes"),
                expected_objects=("aspirin",),
            ),
        ),
    )


def _medical_scenario_2() -> RSBScenario:
    """RSB-M2: Allergy information must not be overridden by stale positive data."""
    turns = [
        RSBTurn("user", "I'm allergic to penicillin.", timestamp=1_700_000_000),
        RSBTurn("user", "My doctor knows about my penicillin allergy.", timestamp=1_700_000_060),
        RSBTurn(
            "user",
            "Actually, I think I used to take penicillin fine years ago.",
            timestamp=1_700_001_000,
        ),
    ]
    return RSBScenario(
        name="RSB-M2",
        domain="medical",
        description="Allergy information persists despite contradictory historical references",
        turns=tuple(turns),
        questions=(
            RSBQuestion(
                text="Do I have any drug allergies?",
                expected_predicates=("has", "is"),
                expected_objects=("penicillin",),
            ),
        ),
    )


def _medical_scenario_3() -> RSBScenario:
    """RSB-M3: Newer diagnosis replaces older one (temporal update test)."""
    turns = [
        RSBTurn("user", "I was diagnosed with pre-diabetes.", timestamp=1_700_000_000),
        RSBTurn("user", "I've been managing my diet.", timestamp=1_700_000_060),
        RSBTurn(
            "user",
            "My latest test shows I no longer have pre-diabetes — my blood sugar is normal.",
            timestamp=1_700_001_000,
        ),
    ]
    return RSBScenario(
        name="RSB-M3",
        domain="medical",
        description="Temporal diagnosis update — newer negative result supersedes older positive",
        turns=tuple(turns),
        questions=(
            RSBQuestion(
                text="What is my current diabetes status?",
                expected_predicates=("is", "has"),
                expected_objects=("normal", "blood sugar", "no longer"),
                must_not_recall=(),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Scheduling domain scenarios
# ---------------------------------------------------------------------------


def _scheduling_scenario_1() -> RSBScenario:
    """RSB-S1: Appointment must be recalled after confirmation."""
    turns = [
        RSBTurn(
            "user",
            "I have a dentist appointment on Thursday at 3pm.",
            timestamp=1_700_000_000,
        ),
        RSBTurn("user", "I confirmed the appointment yesterday.", timestamp=1_700_000_060),
    ]
    noise = _noise(base_ts=1_700_001_000)
    return RSBScenario(
        name="RSB-S1",
        domain="scheduling",
        description="Confirmed appointment must be recalled after unrelated turns",
        turns=tuple(turns + noise),
        questions=(
            RSBQuestion(
                text="Do I have any upcoming appointments?",
                expected_predicates=("has", "is", "scheduled"),
                expected_objects=("dentist", "appointment", "Thursday"),
            ),
        ),
    )


def _scheduling_scenario_2() -> RSBScenario:
    """RSB-S2: Cancellation must override original appointment."""
    turns = [
        RSBTurn(
            "user",
            "I scheduled a doctor's appointment for Friday at 2pm.",
            timestamp=1_700_000_000,
        ),
        RSBTurn(
            "user",
            "I cancelled the Friday doctor appointment.",
            timestamp=1_700_001_000,
        ),
    ]
    return RSBScenario(
        name="RSB-S2",
        domain="scheduling",
        description="Cancellation event must be captured alongside original appointment",
        turns=tuple(turns),
        questions=(
            RSBQuestion(
                text="What happened with my Friday appointment?",
                expected_predicates=("has", "is", "cancelled"),
                expected_objects=("cancelled", "appointment", "Friday"),
            ),
        ),
    )


def _scheduling_scenario_3() -> RSBScenario:
    """RSB-S3: Commitment/promise must be recalled accurately."""
    turns = [
        RSBTurn(
            "user",
            "I promised my sister I would visit her next weekend.",
            timestamp=1_700_000_000,
        ),
        RSBTurn(
            "user",
            "I also need to call mom on Sunday.",
            timestamp=1_700_000_060,
        ),
    ]
    return RSBScenario(
        name="RSB-S3",
        domain="scheduling",
        description="Social commitment must be recalled as a high-priority fact",
        turns=tuple(turns),
        questions=(
            RSBQuestion(
                text="What commitments do I have next weekend?",
                expected_predicates=("has", "is", "promised"),
                expected_objects=("sister", "visit"),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ALL_SCENARIOS: list[RSBScenario] = [
    _medical_scenario_1(),
    _medical_scenario_2(),
    _medical_scenario_3(),
    _scheduling_scenario_1(),
    _scheduling_scenario_2(),
    _scheduling_scenario_3(),
]


def load_scenarios(
    domain: str | None = None,
    names: list[str] | None = None,
) -> list[RSBScenario]:
    """Return RSB scenarios, optionally filtered by domain or name list."""
    scenarios = _ALL_SCENARIOS
    if domain:
        scenarios = [s for s in scenarios if s.domain == domain]
    if names:
        scenarios = [s for s in scenarios if s.name in names]
    return scenarios


def scenario_as_dict(scenario: RSBScenario) -> dict[str, Any]:
    """Serialize scenario to dict (for YAML export)."""
    return {
        "name": scenario.name,
        "domain": scenario.domain,
        "description": scenario.description,
        "user_id": scenario.user_id,
        "turns": [
            {"speaker": t.speaker, "text": t.text, "session_id": t.session_id}
            for t in scenario.turns
        ],
        "questions": [
            {
                "text": q.text,
                "expected_predicates": list(q.expected_predicates),
                "expected_objects": list(q.expected_objects),
                "must_not_recall": list(q.must_not_recall),
            }
            for q in scenario.questions
        ],
    }
