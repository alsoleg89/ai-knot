"""Probe query templates per risk class.

Probes are verification questions used to assess atom relevance against a query.
Templates use {entity} as a placeholder for the resolved entity surface form.
"""

from __future__ import annotations

PROBE_TEMPLATES: dict[str, list[str]] = {
    "medical": [
        "What medical conditions does {entity} have?",
        "What medications does {entity} take?",
        "Has {entity} seen a doctor recently?",
        "Does {entity} have any upcoming medical appointments?",
        "Does {entity} have any allergies?",
        "What health issues has {entity} mentioned?",
    ],
    "scheduling": [
        "What events does {entity} have scheduled?",
        "When is {entity}'s next appointment?",
        "What meetings does {entity} have this week?",
        "Are there any upcoming deadlines for {entity}?",
        "What is on {entity}'s calendar?",
    ],
    "preference": [
        "What does {entity} like or prefer?",
        "What activities does {entity} enjoy?",
        "What does {entity} want to avoid?",
        "What are {entity}'s interests?",
        "What is {entity}'s favorite activity?",
    ],
    "identity": [
        "Who is {entity}?",
        "Where does {entity} live?",
        "What is {entity}'s occupation?",
        "How old is {entity}?",
        "What is {entity}'s background?",
    ],
    "finance": [
        "What is {entity}'s income or salary?",
        "What are {entity}'s financial commitments?",
        "What budget does {entity} have?",
    ],
    "safety": [
        "Are there any safety concerns related to {entity}?",
        "What risks does {entity} face?",
    ],
    "legal": [
        "What legal commitments does {entity} have?",
        "What contracts has {entity} signed?",
    ],
    "commitment": [
        "What has {entity} committed to?",
        "What promises has {entity} made?",
        "What obligations does {entity} have?",
    ],
    "ambient": [
        "What general information do we know about {entity}?",
        "What has {entity} mentioned in conversation?",
    ],
}


def get_probes(risk_class: str, entity: str) -> list[str]:
    """Return formatted probe queries for a given risk class and entity."""
    templates = PROBE_TEMPLATES.get(risk_class, PROBE_TEMPLATES["ambient"])
    return [t.format(entity=entity) for t in templates]


def probe_matches_atom_object(probe: str, object_value: str | None) -> bool:
    """Lightweight check: does probe question domain match atom object content?"""
    if object_value is None:
        return False
    probe_lower = probe.lower()
    obj_lower = object_value.lower()
    # Check for any word overlap (>3 chars to skip noise words)
    probe_words = {w for w in probe_lower.split() if len(w) > 3}
    obj_words = {w for w in obj_lower.split() if len(w) > 3}
    return bool(probe_words & obj_words)
