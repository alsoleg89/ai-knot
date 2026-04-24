"""Action class taxonomy for 4 initial domains: medical, scheduling, preference, identity."""

from __future__ import annotations

from enum import IntFlag, auto

DomainName = str  # "medical" | "scheduling" | "preference" | "identity"


class ActionClass(IntFlag):
    """Bitmap-compatible action class enum covering 4 initial domains."""

    NONE = 0
    # Medical domain
    DIAGNOSE = auto()
    PRESCRIBE = auto()
    SCHEDULE_APPT = auto()
    REFER = auto()
    MONITOR = auto()
    # Scheduling domain
    CREATE_EVENT = auto()
    CANCEL_EVENT = auto()
    RESCHEDULE = auto()
    REMIND = auto()
    # Preference domain
    RECOMMEND = auto()
    AVOID = auto()
    PERSONALIZE = auto()
    # Identity domain
    UPDATE_PROFILE = auto()
    VERIFY_IDENTITY = auto()
    LINK_ENTITY = auto()


DOMAIN_ACTION_CLASSES: dict[DomainName, list[ActionClass]] = {
    "medical": [
        ActionClass.DIAGNOSE,
        ActionClass.PRESCRIBE,
        ActionClass.SCHEDULE_APPT,
        ActionClass.REFER,
        ActionClass.MONITOR,
    ],
    "scheduling": [
        ActionClass.CREATE_EVENT,
        ActionClass.CANCEL_EVENT,
        ActionClass.RESCHEDULE,
        ActionClass.REMIND,
        ActionClass.SCHEDULE_APPT,
    ],
    "preference": [
        ActionClass.RECOMMEND,
        ActionClass.AVOID,
        ActionClass.PERSONALIZE,
    ],
    "identity": [
        ActionClass.UPDATE_PROFILE,
        ActionClass.VERIFY_IDENTITY,
        ActionClass.LINK_ENTITY,
    ],
}

# Reverse map: action class → domain(s)
_ACTION_TO_DOMAINS: dict[ActionClass, list[DomainName]] = {}
for _domain, _classes in DOMAIN_ACTION_CLASSES.items():
    for _cls in _classes:
        _ACTION_TO_DOMAINS.setdefault(_cls, []).append(_domain)


def action_domains(mask: int) -> list[DomainName]:
    """Return domains touched by a given action mask (deduplicated, sorted)."""
    domains: set[DomainName] = set()
    cls_val = ActionClass(mask)
    for cls in ActionClass:
        if cls != ActionClass.NONE and (cls_val & cls):
            domains.update(_ACTION_TO_DOMAINS.get(cls, []))
    return sorted(domains)
