"""Synthetic session generator — medical domain first.

Generates synthetic conversation sessions with ground-truth atom labels
for use in the metrics harness. RSB-Light variant: no human annotation.

Usage:
    sessions = generate_sessions(n=700, domain="medical", seed=42)
    for session in sessions:
        learn_resp = api.learn(session.learn_request)
        recall_resp = api.recall(session.recall_request)
        sc = compute_scorecard(recall_resp.atoms, ..., session.gold_atom_ids, ...)
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

from ai_knot_v2.api.sdk import EpisodeIn, LearnRequest, RecallRequest

# ---------------------------------------------------------------------------
# Domain templates — medical first, then scheduling, preference, identity
# ---------------------------------------------------------------------------

_MEDICAL_TEMPLATES = [
    ("Alice has been diagnosed with {condition}.", "Alice", "has", "{condition}", "medical", 0.9),
    ("I have {condition} the doctor monitors.", "user", "has", "{condition}", "medical", 0.8),
    ("Bob takes {medication} every day.", "Bob", "has", "{medication}", "medical", 0.8),
    ("I have a doctor appointment on {date}.", "user", "has", "appt {date}", "scheduling", 0.6),
    ("Alice is allergic to {allergen}.", "Alice", "has", "{allergen} allergy", "medical", 0.9),
    ("I was prescribed {medication}.", "user", "has", "{medication}", "medical", 0.8),
    ("Bob was referred to a specialist.", "Bob", "has", "{condition}", "medical", 0.7),
]

_SCHEDULING_TEMPLATES = [
    ("I have a meeting with {person} on {date}.", "user", "has", "meeting", "scheduling", 0.5),
    ("Alice scheduled a call for {date}.", "Alice", "has", "call on {date}", "scheduling", 0.5),
    ("Bob's project deadline is {date}.", "Bob", "has", "deadline {date}", "scheduling", 0.6),
]

_PREFERENCE_TEMPLATES = [
    ("I love {activity}.", "user", "prefers", "{activity}", "preference", 0.2),
    ("Alice hates {food}.", "Alice", "dislikes", "{food}", "preference", 0.2),
    ("Bob enjoys {activity} on weekends.", "Bob", "prefers", "{activity}", "preference", 0.2),
]

_IDENTITY_TEMPLATES = [
    ("My name is {name}.", "user", "is", "{name}", "identity", 0.4),
    ("Alice works at {company}.", "Alice", "works_at", "{company}", "identity", 0.4),
    ("I live in {city}.", "user", "lives_in", "{city}", "identity", 0.4),
]

_DOMAIN_TEMPLATES = {
    "medical": _MEDICAL_TEMPLATES,
    "scheduling": _SCHEDULING_TEMPLATES,
    "preference": _PREFERENCE_TEMPLATES,
    "identity": _IDENTITY_TEMPLATES,
}

_CONDITIONS = ["diabetes", "hypertension", "asthma", "migraines", "arthritis", "anemia"]
_MEDICATIONS = ["metformin", "aspirin", "lisinopril", "atorvastatin", "albuterol"]
_ALLERGENS = ["penicillin", "peanuts", "shellfish", "latex", "sulfa drugs"]
_ACTIVITIES = ["hiking", "swimming", "reading", "cooking", "gardening", "cycling"]
_FOODS = ["spicy food", "dairy", "gluten", "seafood", "red meat"]
_PEOPLE = ["Dr. Smith", "the team", "the client", "management"]
_DATES = ["Monday", "next week", "tomorrow", "next Tuesday", "March 15th"]
_NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve"]
_COMPANIES = ["Google", "Acme Corp", "MedTech", "StartupX", "Hospital North"]
_CITIES = ["New York", "London", "Berlin", "Tokyo", "Sydney"]
_QUERIES = {
    "medical": [
        "What medical conditions does the user have?",
        "What medications is the user taking?",
        "Does the user have any upcoming medical appointments?",
        "What allergies does the user have?",
    ],
    "scheduling": [
        "What events are scheduled this week?",
        "When is the next important deadline?",
        "What meetings does the user have?",
    ],
    "preference": [
        "What does the user like?",
        "What activities does the user enjoy?",
        "What does the user avoid or dislike?",
    ],
    "identity": [
        "Who is the user?",
        "Where does the user live?",
        "Where does the user work?",
    ],
}


def _fill(template: str, rng: random.Random) -> str:
    subs = {
        "{condition}": rng.choice(_CONDITIONS),
        "{medication}": rng.choice(_MEDICATIONS),
        "{allergen}": rng.choice(_ALLERGENS),
        "{activity}": rng.choice(_ACTIVITIES),
        "{food}": rng.choice(_FOODS),
        "{person}": rng.choice(_PEOPLE),
        "{date}": rng.choice(_DATES),
        "{name}": rng.choice(_NAMES),
        "{company}": rng.choice(_COMPANIES),
        "{city}": rng.choice(_CITIES),
    }
    result = template
    for k, v in subs.items():
        result = result.replace(k, v)
    return result


@dataclass
class SyntheticSession:
    session_id: str
    domain: str
    learn_request: LearnRequest
    recall_request: RecallRequest
    gold_texts: list[str] = field(default_factory=list)
    gold_episode_ids: set[str] = field(default_factory=set)
    expected_risk_class: str = "medical"


def generate_sessions(
    n: int = 700,
    domain: str = "medical",
    seed: int = 42,
    episodes_per_session: int = 5,
    noise_fraction: float = 0.2,
) -> list[SyntheticSession]:
    """Generate n synthetic sessions with ground-truth labels.

    Each session:
    - Has `episodes_per_session` episodes from the target domain
    - Plus `noise_fraction * episodes` off-topic noise episodes
    - Has a recall query from the target domain

    Returns list of SyntheticSession with gold_texts populated.
    """
    rng = random.Random(seed)
    sessions: list[SyntheticSession] = []

    domain_templates = _DOMAIN_TEMPLATES.get(domain, _MEDICAL_TEMPLATES)
    domain_queries = _QUERIES.get(domain, _QUERIES["medical"])

    for i in range(n):
        session_id = f"synthetic-{domain}-{i:04d}"
        ts_base = int(time.time()) - (n - i) * 3600

        episodes: list[EpisodeIn] = []
        gold_texts: list[str] = []

        # Target-domain episodes
        n_target = max(1, episodes_per_session - int(episodes_per_session * noise_fraction))
        for j in range(n_target):
            tmpl_row = rng.choice(domain_templates)
            text = _fill(tmpl_row[0], rng)
            episodes.append(
                EpisodeIn(
                    text=text,
                    session_id=session_id,
                    timestamp=ts_base + j * 60,
                )
            )
            gold_texts.append(text)

        # Noise episodes (off-domain)
        noise_domains = [d for d in _DOMAIN_TEMPLATES if d != domain]
        for j in range(episodes_per_session - n_target):
            noise_dom = rng.choice(noise_domains)
            tmpl_row = rng.choice(_DOMAIN_TEMPLATES[noise_dom])
            text = _fill(tmpl_row[0], rng)
            episodes.append(
                EpisodeIn(
                    text=text,
                    session_id=session_id,
                    timestamp=ts_base + (n_target + j) * 60,
                )
            )
            # Noise episodes are NOT in gold_texts

        rng.shuffle(episodes)
        query = rng.choice(domain_queries)

        sessions.append(
            SyntheticSession(
                session_id=session_id,
                domain=domain,
                learn_request=LearnRequest(episodes=episodes),
                recall_request=RecallRequest(query=query),
                gold_texts=gold_texts,
                expected_risk_class=domain,
            )
        )

    return sessions
