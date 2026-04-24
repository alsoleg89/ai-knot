"""Action Fingerprint Calculus.

Deterministic computation of action-affect bitmaps, canonical signatures,
distance metrics, and predictions over MemoryAtom collections.

No LLM. All operations are rule-based and reproducible.
"""

from __future__ import annotations

from ai_knot_v2.core.action_taxonomy import (
    DOMAIN_ACTION_CLASSES,
    ActionClass,
    DomainName,
)
from ai_knot_v2.core.atom import MemoryAtom

# Canonical action signature: sorted tuple of domain tokens + action class names
ActionSignature = tuple[str, ...]

# ---------------------------------------------------------------------------
# Query-to-domain keyword index
# ---------------------------------------------------------------------------

_QUERY_DOMAIN_KEYWORDS: list[tuple[str, DomainName]] = [
    ("diagnos", "medical"),
    ("prescription", "medical"),
    ("medication", "medical"),
    ("symptom", "medical"),
    ("doctor", "medical"),
    ("hospital", "medical"),
    ("treatment", "medical"),
    ("allergy", "medical"),
    ("health", "medical"),
    ("nurse", "medical"),
    ("schedule", "scheduling"),
    ("appointment", "scheduling"),
    ("meeting", "scheduling"),
    ("reminder", "scheduling"),
    ("event", "scheduling"),
    ("calendar", "scheduling"),
    ("deadline", "scheduling"),
    ("session", "scheduling"),
    ("prefer", "preference"),
    ("like", "preference"),
    ("enjoy", "preference"),
    ("want", "preference"),
    ("avoid", "preference"),
    ("interest", "preference"),
    ("favorite", "preference"),
    ("dislike", "preference"),
    ("name", "identity"),
    ("age", "identity"),
    ("address", "identity"),
    ("identity", "identity"),
    ("profile", "identity"),
    ("lives", "identity"),
    ("born", "identity"),
    ("gender", "identity"),
    ("nationality", "identity"),
]

# ---------------------------------------------------------------------------
# compute_action_affect_mask
# ---------------------------------------------------------------------------

_MEDICAL_DIAGNOSE_OBJS = frozenset(
    ["condition", "disease", "illness", "symptom", "diagnos", "sick", "injured", "pain", "cancer"]
)
_MEDICAL_PRESCRIBE_OBJS = frozenset(
    ["medication", "prescription", "drug", "treatment", "therapy", "medicine"]
)
_MEDICAL_APPT_OBJS = frozenset(
    ["appointment", "doctor", "hospital", "nurse", "clinic", "specialist"]
)
_SCHEDULING_CANCEL_OBJS = frozenset(["cancel", "cancell", "cancelled"])
_SCHEDULING_RESCHEDULE_OBJS = frozenset(["reschedule", "rescheduled", "postpone"])


def compute_action_affect_mask(atom: MemoryAtom) -> int:
    """Compute action affect bitmap from atom fields deterministically."""
    mask = ActionClass.NONE
    rc = atom.risk_class
    pred = atom.predicate
    obj_lower = (atom.object_value or "").lower()

    if rc == "medical":
        mask |= ActionClass.MONITOR
        if any(kw in obj_lower for kw in _MEDICAL_DIAGNOSE_OBJS):
            mask |= ActionClass.DIAGNOSE
        if any(kw in obj_lower for kw in _MEDICAL_PRESCRIBE_OBJS):
            mask |= ActionClass.PRESCRIBE
        if any(kw in obj_lower for kw in _MEDICAL_APPT_OBJS):
            mask |= ActionClass.SCHEDULE_APPT
        if "referral" in obj_lower or "specialist" in obj_lower or "consult" in obj_lower:
            mask |= ActionClass.REFER

    elif rc == "scheduling":
        if any(kw in obj_lower for kw in _SCHEDULING_CANCEL_OBJS):
            mask |= ActionClass.CANCEL_EVENT
        elif any(kw in obj_lower for kw in _SCHEDULING_RESCHEDULE_OBJS):
            mask |= ActionClass.RESCHEDULE
        else:
            mask |= ActionClass.CREATE_EVENT
        mask |= ActionClass.REMIND

    elif rc == "preference":
        if atom.polarity == "neg":
            mask |= ActionClass.AVOID
        else:
            mask |= ActionClass.RECOMMEND
        mask |= ActionClass.PERSONALIZE

    elif rc == "identity":
        mask |= ActionClass.UPDATE_PROFILE
        if pred in ("is", "has") and ("name" in obj_lower or "called" in obj_lower):
            mask |= ActionClass.VERIFY_IDENTITY
        if pred in ("lives_in", "moved_to"):
            mask |= ActionClass.LINK_ENTITY

    elif rc == "finance":
        # Finance not yet in first-pass domains — maps to PERSONALIZE as neutral
        mask |= ActionClass.PERSONALIZE

    return int(mask)


# ---------------------------------------------------------------------------
# canonical_action_signature
# ---------------------------------------------------------------------------


def canonical_action_signature(atoms: list[MemoryAtom], query: str) -> ActionSignature:
    """Build a canonical sorted-tuple signature from atoms and query.

    The signature combines inferred domain tokens (from query keywords) with
    action class names (from atom masks). Used as a cache key and for distance.
    """
    q_lower = query.lower()
    intent_domains: set[DomainName] = set()
    for keyword, domain in _QUERY_DOMAIN_KEYWORDS:
        if keyword in q_lower:
            intent_domains.add(domain)

    action_names: set[str] = set()
    for atom in atoms:
        mask = ActionClass(atom.action_affect_mask)
        for cls in ActionClass:
            if cls != ActionClass.NONE and (mask & cls) and cls.name:
                action_names.add(cls.name)

    return tuple(sorted(intent_domains | action_names))


# ---------------------------------------------------------------------------
# action_distance
# ---------------------------------------------------------------------------


def action_distance(action_a: int, action_b: int) -> float:
    """Normalized Hamming distance between two action masks ∈ [0.0, 1.0].

    0.0 = identical, 1.0 = maximally different.
    """
    xor = action_a ^ action_b
    bits_diff = bin(xor).count("1")
    union = action_a | action_b
    total_bits = bin(union).count("1")
    if total_bits == 0:
        return 0.0
    return bits_diff / total_bits


# ---------------------------------------------------------------------------
# predict_action
# ---------------------------------------------------------------------------


def predict_action(signature: ActionSignature) -> ActionClass:
    """Deterministic action class prediction from a canonical signature.

    Returns the highest-scoring ActionClass, or ActionClass.NONE if ambiguous/empty.

    Scoring:
    - Direct match to ActionClass name → +3
    - Domain name match → +1 per domain-class
    - Ties → prefer higher-priority class (lower enum value wins)
    """
    scores: dict[ActionClass, int] = {}

    for token in signature:
        # Try direct ActionClass name match
        try:
            cls = ActionClass[token]
            if cls != ActionClass.NONE:
                scores[cls] = scores.get(cls, 0) + 3
                continue
        except KeyError:
            pass

        # Domain name → boost all domain action classes
        domain_classes = DOMAIN_ACTION_CLASSES.get(token, [])
        for cls in domain_classes:
            scores[cls] = scores.get(cls, 0) + 1

    if not scores:
        return ActionClass.NONE

    max_score = max(scores.values())
    candidates = [cls for cls, s in scores.items() if s == max_score]
    # Deterministic tie-break: lowest integer value (highest priority)
    return min(candidates, key=lambda c: c.value)
