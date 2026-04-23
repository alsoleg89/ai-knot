"""Deterministic rule-based materializer — converts RawEpisode → list[AtomicClaim].

Design constraints (enforced by tests/test_materialization_kind_whitelist.py):
  * Only emits ClaimKind.STATE / RELATION / EVENT / DURATION / TRANSITION.
  * Never calls any LLM or network service.
  * Fully deterministic: same RawEpisode → same AtomicClaim set every time.
  * Every produced claim has a non-empty source_episode_id and source_spans.

LLM-enriched DESCRIPTOR / INTENT claims are handled by ai_knot.enrichment (offline
optional pass) and must NEVER be produced here.

Current materialization version.  Increment when the extraction logic changes
so that rebuild_materialized() can detect stale claims.
"""

from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime

from ai_knot._text_guards import is_deictic_subject, is_evaluative_predicate
from ai_knot.query_types import (
    DETERMINISTIC_CLAIM_KINDS,
    AtomicClaim,
    ClaimKind,
    DirtyKey,
    RawEpisode,
)

MATERIALIZATION_VERSION: int = 6

# ---------------------------------------------------------------------------
# Regex patterns for deterministic extraction
# ---------------------------------------------------------------------------

# Sentence boundary splitter (handles .!? followed by whitespace or end).
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Date/time patterns — used to detect EVENT and DURATION claims.
_DATE_RE = re.compile(
    r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[./]\d{1,2}[./]\d{2,4}"
    r"|(?:january|february|march|april|may|june|july|august|september"
    r"|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,\s*\d{4})?)\b",
    re.IGNORECASE,
)

_DURATION_RE = re.compile(
    r"\bfor\s+(?:about\s+)?(\d+\s+(?:second|minute|hour|day|week|month|year)s?)\b",
    re.IGNORECASE,
)

# Entity-state patterns: "X is/was/are/were Y", "X has/had Y".
_STATE_RE = re.compile(
    r"^([A-Z][^,\.]{1,40}?)\s+(?:is|was|are|were|has|had|have|becomes?|became|"
    r"remains?|remained|seems?|seemed|looks?|looked)\s+(.{3,80})$",
    re.IGNORECASE,
)

# Occupation/role patterns: "X works as Y", "X serves as Y", "X acts as Y".
_ROLE_RE = re.compile(
    r"^([A-Z][^,\.]{1,40}?)\s+"
    r"(?:works?|worked|serves?|served|acts?|acted|functions?|functioned)\s+"
    r"(?:as\s+(?:a\s+|an\s+)?|as\s+)(.{3,60})$",
    re.IGNORECASE,
)

# Relation patterns: "X [verb] Y", "X and Y [verb]"
_RELATION_RE = re.compile(
    r"^([A-Z][^,\.]{1,40}?)\s+"
    r"(?:knows?|knew|met|meets?|helped|helps?|loves?|loved|married|marries?|"
    r"hired|hires?|trained|trains?|supported|supports?|founded|co-founded?)\s+"
    r"([A-Z][^,\.]{1,40})$",
    re.IGNORECASE,
)

# Transition patterns: "X moved to Y", "X changed from A to B", "X became Y".
_TRANSITION_RE = re.compile(
    r"^([A-Z][^,\.]{1,40}?)\s+"
    r"(?:moved?|relocated?|transferred?|promoted?|switched?|changed?|transitioned?)\s+"
    r"(?:to|from|into|toward)\s+(.{3,60})$",
    re.IGNORECASE,
)

# Death / passing pattern: "X passed away" or "X died [...]".
_DEATH_RE = re.compile(
    r"^([A-Z][a-zA-Z' ]{1,50}?)\s+(?:passed\s+away|died)(?:\s+.{0,60})?\.?\s*$",
    re.IGNORECASE,
)

# Session-date prefix extracted by dated-learn ingest mode.
_DATE_PREFIX_RE = re.compile(r"^\[([^\]]+)\]\s*")

# Speaker-prefix pattern: "Dave: " or "Alice - " at start of turn text.
_SPEAKER_PREFIX_RE = re.compile(r"^([A-Z][a-zA-Z]+)\s*[:–\-]\s*")

# Name-like token (two or more capitalized words).
_PROPER_NAME_RE = re.compile(r"\b([A-Z][a-z]+ (?:[A-Z][a-z]+ )?[A-Z][a-z]+)\b")

# Sentence openers that signal questions, imperatives, or conversational filler —
# sentences starting with these should not be materialized as facts.
_SENT_GARBAGE_OPENERS: frozenset[str] = frozenset(
    {
        "Do",
        "Does",
        "Did",
        "Have",
        "Has",
        "What",
        "When",
        "Where",
        "How",
        "Why",
        "Would",
        "Could",
        "Should",
        "Is",
        "Are",
        "Was",
        "Were",
        "Can",
        "Will",
        "Take",
        "Let",
        "Glad",
        "Thanks",
        "Thank",
        "Oh",
        "Hmm",
        "Yeah",
        "Sure",
        "Right",
        "Actually",
        "Well",
        "So",
        "OK",
        "Okay",
    }
)

# Pronoun-like tokens that should not be treated as named subjects.
_PRONOUN_SUBJECTS: frozenset[str] = frozenset(
    {
        "I",
        "You",
        "We",
        "They",
        "It",
        "This",
        "That",
        "He",
        "She",
        "My",
        "Your",
        "Our",
        "Their",
        "Its",
        "Me",
        "Him",
        "Her",
        "Us",
        "Them",
    }
)

# First-person preference / sentiment patterns (used when speaker is known).
_FP_LIKES_RE = re.compile(
    r"^I\s+(?:really\s+)?(?:love|enjoy|like|adore|prefer|appreciate)\s+(.+?)\.?\s*$",
    re.IGNORECASE,
)
_FP_DISLIKES_RE = re.compile(
    r"^I\s+(?:really\s+)?"
    r"(?:hate|dislike|can't stand|cannot stand|don't like|do not like)"
    r"\s+(.+?)\.?\s*$",
    re.IGNORECASE,
)
_FP_SATISFYING_RE = re.compile(
    r"(?:It(?:'s| is)(?: so| really)? satisfying to\s+(.+?)\.?\s*$"
    r"|I\s+(?:really\s+)?find\s+(.+?)\s+(?:so |really )?satisfying\.?\s*$)",
    re.IGNORECASE,
)
_FP_SELF_STATE_RE = re.compile(
    r"^I\s+(?:am|'m)\s+(.+?)\.?\s*$",
    re.IGNORECASE,
)

# First-person action patterns (used when speaker is known).
_FP_DRIVES_RE = re.compile(
    r"^I\s+(?:drive|own|have)\s+(?:a|an|the)\s+(.+?)\.?\s*$",
    re.IGNORECASE,
)
_FP_WORK_RE = re.compile(
    r"^I\s+(?:joined?|(?:started?|work(?:ed)?) at|am at)\s+(.+?)\.?\s*$",
    re.IGNORECASE,
)
_FP_WORK_LEFT_RE = re.compile(
    r"^I\s+(?:left|retired?\s+from)\s+(.+?)\.?\s*$",
    re.IGNORECASE,
)
_FP_MOVE_RE = re.compile(
    r"^I\s+(?:moved?|relocated?|transferred?)\s+to\s+(.+?)\.?\s*$",
    re.IGNORECASE,
)
_FP_STARTED_RE = re.compile(
    r"^I\s+(?:started?|began?|took\s+up|picked\s+up)\s+(?:to\s+)?(.+?)\.?\s*$",
    re.IGNORECASE,
)
_FP_BECAME_RE = re.compile(
    r"^I\s+(?:became?|turned\s+into)\s+(?:a\s+|an\s+)?(.+?)\.?\s*$",
    re.IGNORECASE,
)
_FP_ACTIVITY_RE = re.compile(
    r"^I(?:'ve|'ve|\s+have)\s+been\s+(.+?)(?:\s+(?:lately|recently|these\s+days|a\s+lot))?\.?\s*$",
    re.IGNORECASE,
)

# Leading adverbial prefixes that hide the "I <verb>" opener required by the
# first-person patterns above (e.g. "Last weekend I joined ..."). Stripped only
# when the residue still begins with "I <word>" — so non-FP sentences are left
# untouched.
_LEADING_ADV_RE = re.compile(
    r"^(?:"
    r"Last\s+\w+"
    r"|Yesterday"
    r"|Today"
    r"|Tonight"
    r"|Recently"
    r"|Lately"
    r"|This\s+(?:morning|afternoon|evening|weekend|week|month|year)"
    r"|A\s+(?:few|couple\s+of)\s+(?:days|weeks|months|years)\s+ago"
    r"|Earlier(?:\s+(?:today|this\s+\w+))?"
    r"|Just\s+(?:now|recently)"
    r"|Since\s+we\s+last\s+(?:spoke|talked|chatted)"
    r"|Over\s+the\s+(?:weekend|week|past\s+\w+)"
    r")\s*,?\s+(?=I\s+\w+)",
    re.IGNORECASE,
)

# First-person event/action patterns (used when speaker is known).
# Each entry: relation_name → compiled regex.
_FP_EVENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("attended", re.compile(r"^I\s+(?:attended?|went\s+to)\s+(.+?)\.?\s*$", re.IGNORECASE)),
    (
        "joined",
        re.compile(
            r"^I\s+(?:joined?|became\s+(?:a\s+|an\s+)?member\s+of)\s+(.+?)\.?\s*$", re.IGNORECASE
        ),
    ),
    (
        "signed_up_for",
        re.compile(
            r"^I\s+(?:signed?\s+up\s+for|registered\s+for|enrolled\s+in)\s+(.+?)\.?\s*$",
            re.IGNORECASE,
        ),
    ),
    ("applied_to", re.compile(r"^I\s+(?:applied?\s+(?:to|for))\s+(.+?)\.?\s*$", re.IGNORECASE)),
    (
        "visited",
        re.compile(
            r"^I\s+(?:visited?|stopped\s+by|went\s+to)\s+(.+?)\.?\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "met_with",
        re.compile(
            r"^I\s+(?:met\s+(?:with\s+)?|had\s+a\s+meeting\s+with\s+|caught\s+up\s+with\s+)\s*(.+?)\.?\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "bought",
        re.compile(r"^I\s+(?:bought?|purchased?|picked\s+up)\s+(.+?)\.?\s*$", re.IGNORECASE),
    ),
    ("read", re.compile(r"^I\s+(?:read|finished\s+(?:reading\s+)?)\s+(.+?)\.?\s*$", re.IGNORECASE)),
    (
        "ran",
        re.compile(
            r"^I\s+(?:ran?|completed?\s+(?:a\s+|the\s+)?(?:run|race|event|course|circuit))\s+(.+?)\.?\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "created",
        re.compile(
            r"^I\s+(?:created?|built?|made?|launched?|released?|published?|wrote|written)\s+(.+?)\.?\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "spoke_at",
        re.compile(
            r"^I\s+(?:spoke?\s+at|presented?\s+at|gave\s+a\s+(?:talk|presentation|lecture)\s+at)\s+(.+?)\.?\s*$",
            re.IGNORECASE,
        ),
    ),
    (
        "acquired",
        re.compile(
            r"^I\s+(?:got|received|was\s+given|was\s+awarded)\s+(?:a\s+|an\s+|the\s+)?(.+?)\.?\s*$",
            re.IGNORECASE,
        ),
    ),
]

# Relative-time markers used in event qualifier extraction.
_RELATIVE_TIME_RE = re.compile(r"\b(today|yesterday|tomorrow)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def materialize_episode(raw: RawEpisode) -> list[AtomicClaim]:
    """Convert one RawEpisode into zero or more AtomicClaims.

    Deterministic: same input always produces the same output.
    Only emits the five DETERMINISTIC_CLAIM_KINDS.
    Raises AssertionError if a claim with DESCRIPTOR or INTENT kind would be
    produced (guard against future regressions).
    """
    now = datetime.now(UTC)
    text, session_date = _strip_date_prefix(raw.raw_text)
    session_date = session_date or raw.session_date

    # Extract and strip speaker prefix ("Dave: ...") for first-person claim mapping.
    speaker: str | None = None
    m_speaker = _SPEAKER_PREFIX_RE.match(text)
    if m_speaker:
        speaker = m_speaker.group(1)
        text = text[m_speaker.end() :]

    sentences = _split_sentences(text)
    claims: list[AtomicClaim] = []

    for sent in sentences:
        sent = sent.strip()
        if len(sent) < 6:
            continue
        if _is_garbage_sentence(sent):
            continue
        extracted = _extract_from_sentence(sent, raw, session_date, now, speaker=speaker)
        claims.extend(extracted)

    # Invariant guard — fail fast during testing.
    for c in claims:
        assert c.kind in DETERMINISTIC_CLAIM_KINDS, (
            f"Materializer emitted forbidden kind {c.kind!r} for episode {raw.id!r}. "
            f"DESCRIPTOR/INTENT are enrichment-only."
        )

    return claims


def rebuild_claims_from_raw(
    episodes: list[RawEpisode],
    *,
    version: int = MATERIALIZATION_VERSION,
) -> list[AtomicClaim]:
    """Rebuild all claims from a list of raw episodes.

    Deterministic: calling this twice with the same episodes + same version
    produces an identical claim set.
    """
    all_claims: list[AtomicClaim] = []
    for ep in episodes:
        claims = materialize_episode(ep)
        # Stamp requested version on all claims.
        if version != MATERIALIZATION_VERSION:
            claims = [_with_version(c, version) for c in claims]
        all_claims.extend(claims)
    return all_claims


def dirty_keys_for_claims(claims: list[AtomicClaim]) -> list[DirtyKey]:
    """Compute the minimal set of DirtyKeys for a batch of new/updated claims.

    Uses the finest-grained key level possible:
    - slot_key set → DirtyKey.for_slot(subject, relation)
    - no slot_key  → DirtyKey.for_subject(subject)
    Deduplicates keys before returning.
    """
    seen: set[tuple[str, str | None]] = set()
    keys: list[DirtyKey] = []
    for c in claims:
        if c.subject and c.relation:
            sk: tuple[str, str | None] = (c.subject, c.relation)
            if sk not in seen:
                seen.add(sk)
                keys.append(DirtyKey.for_slot(c.subject, c.relation))
        elif c.subject:
            sk = (c.subject, None)
            if sk not in seen:
                seen.add(sk)
                keys.append(DirtyKey.for_subject(c.subject))
    return keys


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _strip_date_prefix(text: str) -> tuple[str, datetime | None]:
    """Remove a leading [DATE] prefix and return (cleaned_text, parsed_date)."""
    m = _DATE_PREFIX_RE.match(text)
    if not m:
        return text, None
    date_str = m.group(1)
    clean = text[m.end() :]
    parsed = _parse_date_str(date_str)
    return clean, parsed


def _parse_date_str(s: str) -> datetime | None:
    """Try a few common date formats; return None on failure."""
    for fmt in (
        "%d %B, %Y",
        "%d %B %Y",
        "%B %d, %Y",
        "%B %d %Y",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d.%m.%Y",
    ):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences; also split on newlines."""
    parts: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts.extend(_SENT_SPLIT.split(line))
    return [p for p in parts if p]


def _make_claim_id(raw_id: str, suffix: str) -> str:
    """Deterministic claim ID from episode_id + disambiguating suffix."""
    key = f"{raw_id}|{suffix}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _is_garbage_sentence(sent: str) -> bool:
    """Return True for question, imperative, or conversational-filler sentences.

    Catches:
    - Sentences that end with '?' (questions without a capitalized opener).
    - Sentences whose first word is in _SENT_GARBAGE_OPENERS.
    """
    stripped = sent.strip().rstrip(".!,;:")
    if stripped.endswith("?"):
        return True
    parts = stripped.split()
    first = parts[0].rstrip(",;:!?") if parts else ""
    return first in _SENT_GARBAGE_OPENERS


def _strip_leading_adverbial(sent: str) -> str:
    """Drop a leading temporal adverbial so the remainder starts with "I <verb>".

    Returns the original sentence when the residue would not start with an
    I-opener — leaves non-FP sentences untouched.
    """
    m = _LEADING_ADV_RE.match(sent)
    if not m:
        return sent
    return sent[m.end() :]


def _extract_from_sentence(
    sent: str,
    raw: RawEpisode,
    session_date: datetime | None,
    now: datetime,
    *,
    speaker: str | None = None,
) -> list[AtomicClaim]:
    """Try all extraction patterns on a single sentence; return matched claims."""
    results: list[AtomicClaim] = []

    # --- FIRST-PERSON PREFERENCE (requires named speaker) ----------------
    if speaker and speaker not in _PRONOUN_SUBJECTS:
        # Strip leading temporal adverbials ("Last weekend, I ...") so patterns
        # that require "^I\s+<verb>" still match. Original `sent` is retained
        # for qualifiers, spans, and claim IDs so source attribution is stable.
        fp_sent = _strip_leading_adverbial(sent)
        m = _FP_LIKES_RE.match(fp_sent)
        if m:
            claim_id = _make_claim_id(raw.id, f"likes:{sent[:30]}")
            results.append(
                _make_claim(
                    claim_id=claim_id,
                    raw=raw,
                    kind=ClaimKind.STATE,
                    subject=speaker,
                    relation="likes",
                    value_text=m.group(1).strip(),
                    qualifiers={"source_sentence": sent[:120]},
                    event_time=None,
                    session_date=session_date,
                    now=now,
                    span=(0, len(sent)),
                    slot_key=f"{speaker}::likes",
                )
            )
            return results
        m = _FP_DISLIKES_RE.match(fp_sent)
        if m:
            claim_id = _make_claim_id(raw.id, f"dislikes:{sent[:30]}")
            results.append(
                _make_claim(
                    claim_id=claim_id,
                    raw=raw,
                    kind=ClaimKind.STATE,
                    subject=speaker,
                    relation="dislikes",
                    value_text=m.group(1).strip(),
                    qualifiers={"source_sentence": sent[:120]},
                    event_time=None,
                    session_date=session_date,
                    now=now,
                    span=(0, len(sent)),
                    slot_key=f"{speaker}::dislikes",
                )
            )
            return results
        m = _FP_SATISFYING_RE.search(fp_sent)
        if m:
            value = (m.group(1) or m.group(2) or "").strip()
            if value:
                claim_id = _make_claim_id(raw.id, f"satisfying:{sent[:30]}")
                results.append(
                    _make_claim(
                        claim_id=claim_id,
                        raw=raw,
                        kind=ClaimKind.STATE,
                        subject=speaker,
                        relation="finds_satisfying",
                        value_text=value,
                        qualifiers={"source_sentence": sent[:120]},
                        event_time=None,
                        session_date=session_date,
                        now=now,
                        span=(0, len(sent)),
                        slot_key=f"{speaker}::finds_satisfying",
                    )
                )
                return results

        # --- FIRST-PERSON SELF-STATE (requires named speaker) -----------------------
        m = _FP_SELF_STATE_RE.match(fp_sent)
        if m and speaker and speaker not in _PRONOUN_SUBJECTS:
            value = m.group(1).strip()
            claim_id = _make_claim_id(raw.id, f"self_state:{sent[:30]}")
            results.append(
                _make_claim(
                    claim_id=claim_id,
                    raw=raw,
                    kind=ClaimKind.STATE,
                    subject=speaker,
                    relation="state",
                    value_text=value,
                    qualifiers={"source_sentence": sent[:120]},
                    event_time=None,
                    session_date=session_date,
                    now=now,
                    span=(0, len(sent)),
                    slot_key=f"{speaker}::state",
                )
            )
            return results

        # --- FIRST-PERSON ACTIONS (requires named speaker) ---------------
        m = _FP_DRIVES_RE.match(fp_sent)
        if m:
            claim_id = _make_claim_id(raw.id, f"drives:{sent[:30]}")
            results.append(
                _make_claim(
                    claim_id=claim_id,
                    raw=raw,
                    kind=ClaimKind.STATE,
                    subject=speaker,
                    relation="drives",
                    value_text=m.group(1).strip(),
                    qualifiers={"source_sentence": sent[:120]},
                    event_time=None,
                    session_date=session_date,
                    now=now,
                    span=(0, len(sent)),
                    slot_key=f"{speaker}::drives",
                )
            )
            return results
        m = _FP_WORK_RE.match(fp_sent)
        if m:
            claim_id = _make_claim_id(raw.id, f"works_at:{sent[:30]}")
            results.append(
                _make_claim(
                    claim_id=claim_id,
                    raw=raw,
                    kind=ClaimKind.RELATION,
                    subject=speaker,
                    relation="works_at",
                    value_text=m.group(1).strip(),
                    qualifiers={"source_sentence": sent[:120]},
                    event_time=session_date,
                    session_date=session_date,
                    now=now,
                    span=(0, len(sent)),
                    slot_key=f"{speaker}::works_at",
                )
            )
            return results
        m = _FP_WORK_LEFT_RE.match(fp_sent)
        if m:
            claim_id = _make_claim_id(raw.id, f"left_job:{sent[:30]}")
            results.append(
                _make_claim(
                    claim_id=claim_id,
                    raw=raw,
                    kind=ClaimKind.TRANSITION,
                    subject=speaker,
                    relation="left_job",
                    value_text=m.group(1).strip(),
                    qualifiers={"source_sentence": sent[:120]},
                    event_time=session_date,
                    session_date=session_date,
                    now=now,
                    span=(0, len(sent)),
                    slot_key=f"{speaker}::left_job",
                )
            )
            return results
        m = _FP_MOVE_RE.match(fp_sent)
        if m:
            claim_id = _make_claim_id(raw.id, f"moved_to:{sent[:30]}")
            results.append(
                _make_claim(
                    claim_id=claim_id,
                    raw=raw,
                    kind=ClaimKind.TRANSITION,
                    subject=speaker,
                    relation="moved_to",
                    value_text=m.group(1).strip(),
                    qualifiers={"source_sentence": sent[:120]},
                    event_time=session_date,
                    session_date=session_date,
                    now=now,
                    span=(0, len(sent)),
                    slot_key=f"{speaker}::moved_to",
                )
            )
            return results
        m = _FP_STARTED_RE.match(fp_sent)
        if m:
            claim_id = _make_claim_id(raw.id, f"started:{sent[:30]}")
            results.append(
                _make_claim(
                    claim_id=claim_id,
                    raw=raw,
                    kind=ClaimKind.STATE,
                    subject=speaker,
                    relation="started",
                    value_text=m.group(1).strip(),
                    qualifiers={"source_sentence": sent[:120]},
                    event_time=session_date,
                    session_date=session_date,
                    now=now,
                    span=(0, len(sent)),
                    slot_key=f"{speaker}::started",
                )
            )
            return results
        m = _FP_BECAME_RE.match(fp_sent)
        if m:
            claim_id = _make_claim_id(raw.id, f"became:{sent[:30]}")
            results.append(
                _make_claim(
                    claim_id=claim_id,
                    raw=raw,
                    kind=ClaimKind.TRANSITION,
                    subject=speaker,
                    relation="became",
                    value_text=m.group(1).strip(),
                    qualifiers={"source_sentence": sent[:120]},
                    event_time=session_date,
                    session_date=session_date,
                    now=now,
                    span=(0, len(sent)),
                    slot_key=f"{speaker}::became",
                )
            )
            return results
        m = _FP_ACTIVITY_RE.match(fp_sent)
        if m:
            claim_id = _make_claim_id(raw.id, f"activity:{sent[:30]}")
            results.append(
                _make_claim(
                    claim_id=claim_id,
                    raw=raw,
                    kind=ClaimKind.STATE,
                    subject=speaker,
                    relation="activity_ongoing",
                    value_text=m.group(1).strip(),
                    qualifiers={"source_sentence": sent[:120]},
                    event_time=None,
                    session_date=session_date,
                    now=now,
                    span=(0, len(sent)),
                    slot_key=f"{speaker}::activity_ongoing",
                )
            )
            return results

        # --- FIRST-PERSON EVENTS (requires named speaker) ----------------
        for relation, fp_re in _FP_EVENT_PATTERNS:
            m = fp_re.match(fp_sent)
            if m:
                value = m.group(1).strip() if m.lastindex and m.group(1) else sent[:80]
                qualifiers: dict[str, str] = {
                    "time_anchor": "session_date",
                    "source_sentence": sent[:120],
                }
                rel_m = _RELATIVE_TIME_RE.search(sent)
                if rel_m:
                    qualifiers["relative_time"] = rel_m.group(1).lower()
                claim_id = _make_claim_id(raw.id, f"fp_event_{relation}:{sent[:30]}")
                results.append(
                    _make_claim(
                        claim_id=claim_id,
                        raw=raw,
                        kind=ClaimKind.EVENT,
                        subject=speaker,
                        relation=relation,
                        value_text=value,
                        qualifiers=qualifiers,
                        event_time=None,  # resolved later by time_resolve()
                        session_date=session_date,
                        now=now,
                        span=(0, len(sent)),
                        slot_key=f"{speaker}::{relation}",
                    )
                )
                return results

    # --- DURATION --------------------------------------------------------
    m = _DURATION_RE.search(sent)
    if m:
        duration_str = m.group(1)
        # Find subject if present.
        subject = _extract_simple_subject(sent[: m.start()]) or "unknown"
        claim_id = _make_claim_id(raw.id, f"duration:{sent[:30]}")
        span = (m.start(), m.end())
        results.append(
            _make_claim(
                claim_id=claim_id,
                raw=raw,
                kind=ClaimKind.DURATION,
                subject=subject,
                relation="duration",
                value_text=duration_str,
                qualifiers={"source_sentence": sent[:120]},
                event_time=session_date,
                session_date=session_date,
                now=now,
                span=span,
            )
        )
        return results  # one claim per sentence max for duration

    # --- DEATH / PASSING -------------------------------------------------
    m = _DEATH_RE.match(sent)
    if m:
        subject = m.group(1).strip()
        claim_id = _make_claim_id(raw.id, f"passed_away:{sent[:30]}")
        results.append(
            _make_claim(
                claim_id=claim_id,
                raw=raw,
                kind=ClaimKind.TRANSITION,
                subject=subject,
                relation="passed_away",
                value_text="deceased",
                qualifiers={"source_sentence": sent[:120]},
                event_time=session_date,
                session_date=session_date,
                now=now,
                span=(m.start(), m.end()),
                slot_key=f"{subject}::passed_away",
            )
        )
        return results

    # --- TRANSITION ------------------------------------------------------
    m = _TRANSITION_RE.match(sent)
    if m:
        subject, dest = m.group(1).strip(), m.group(2).strip()
        claim_id = _make_claim_id(raw.id, f"transition:{sent[:30]}")
        results.append(
            _make_claim(
                claim_id=claim_id,
                raw=raw,
                kind=ClaimKind.TRANSITION,
                subject=subject,
                relation="moved_to",
                value_text=dest,
                qualifiers={},
                event_time=session_date,
                session_date=session_date,
                now=now,
                span=(m.start(), m.end()),
            )
        )
        return results

    # --- RELATION --------------------------------------------------------
    m = _RELATION_RE.match(sent)
    if m:
        subject, obj = m.group(1).strip(), m.group(2).strip()
        verb_m = re.search(
            r"\b(knows?|knew|met|meets?|helped|helps?|loves?|loved|"
            r"married|marries?|hired|hires?|trained|trains?|"
            r"supported|supports?|founded|co-founded?)\b",
            sent,
            re.IGNORECASE,
        )
        relation = verb_m.group(1).lower().rstrip("s") if verb_m else "related_to"
        claim_id = _make_claim_id(raw.id, f"relation:{sent[:30]}")
        results.append(
            _make_claim(
                claim_id=claim_id,
                raw=raw,
                kind=ClaimKind.RELATION,
                subject=subject,
                relation=relation,
                value_text=obj,
                qualifiers={},
                event_time=session_date,
                session_date=session_date,
                now=now,
                span=(m.start(), m.end()),
            )
        )
        return results

    # --- ROLE (subtype of STATE) -----------------------------------------
    # Belt-and-suspenders: skip if the would-be subject is a pronoun.
    m = _ROLE_RE.match(sent)
    if m:
        subject, role = m.group(1).strip(), m.group(2).strip()
        if subject in _PRONOUN_SUBJECTS:
            return results
        claim_id = _make_claim_id(raw.id, f"state_role:{sent[:30]}")
        slot_key = f"{subject}::role"
        results.append(
            _make_claim(
                claim_id=claim_id,
                raw=raw,
                kind=ClaimKind.STATE,
                subject=subject,
                relation="role",
                value_text=role,
                qualifiers={},
                event_time=None,
                session_date=session_date,
                now=now,
                span=(m.start(), m.end()),
                slot_key=slot_key,
            )
        )
        return results

    # --- STATE -----------------------------------------------------------
    # Belt-and-suspenders: skip if the would-be subject is a pronoun.
    m = _STATE_RE.match(sent)
    if m:
        subject, predicate = m.group(1).strip(), m.group(2).strip()
        if subject in _PRONOUN_SUBJECTS:
            return results
        # Discourse guard: skip deictic subject + evaluative predicate combos.
        if is_deictic_subject(subject) and is_evaluative_predicate(predicate):
            return results
        if len(subject) >= 2 and len(predicate) >= 2:
            claim_id = _make_claim_id(raw.id, f"state:{sent[:30]}")
            results.append(
                _make_claim(
                    claim_id=claim_id,
                    raw=raw,
                    kind=ClaimKind.STATE,
                    subject=subject,
                    relation="state",
                    value_text=predicate,
                    qualifiers={},
                    event_time=None,
                    session_date=session_date,
                    now=now,
                    span=(m.start(), m.end()),
                )
            )
            return results

    # --- EVENT (fallback: sentence has a date and a subject-like token) --
    date_m = _DATE_RE.search(sent)
    if date_m:
        event_date = _parse_date_str(date_m.group(0)) or session_date
        subject = _extract_simple_subject(sent) or "unknown"
        # Discourse guard: skip deictic + evaluative noise.
        if is_deictic_subject(subject) and is_evaluative_predicate(sent[:80]):
            return results
        claim_id = _make_claim_id(raw.id, f"event:{sent[:30]}")
        results.append(
            _make_claim(
                claim_id=claim_id,
                raw=raw,
                kind=ClaimKind.EVENT,
                subject=subject,
                relation="occurred",
                value_text=sent[:200],
                qualifiers={"date_token": date_m.group(0)},
                event_time=event_date,
                session_date=session_date,
                now=now,
                span=(0, len(sent)),
            )
        )
        return results

    return []


def _extract_simple_subject(text: str) -> str | None:
    """Extract the first proper-name-like token from text.

    Excludes pronouns and question-opener words that cannot be subjects of
    factual claims (e.g. "Do", "What", "I", "You").
    """
    m = _PROPER_NAME_RE.search(text)
    if m:
        candidate = m.group(1)
        first_word = candidate.split()[0]
        if first_word not in _PRONOUN_SUBJECTS and first_word not in _SENT_GARBAGE_OPENERS:
            return candidate
    # Fall back to first capitalized word.
    parts = text.split()
    for p in parts:
        p = p.strip(".,;:!?")
        if (
            p
            and p[0].isupper()
            and len(p) > 2
            and p not in _PRONOUN_SUBJECTS
            and p not in _SENT_GARBAGE_OPENERS
        ):
            return p
    return None


def _make_claim(
    *,
    claim_id: str,
    raw: RawEpisode,
    kind: ClaimKind,
    subject: str,
    relation: str,
    value_text: str,
    qualifiers: dict[str, str],
    event_time: datetime | None,
    session_date: datetime | None,
    now: datetime,
    span: tuple[int, int],
    slot_key: str = "",
) -> AtomicClaim:
    """Construct a single AtomicClaim with all required fields populated."""
    from ai_knot.tokenizer import tokenize as _tokenize

    tokens = tuple(_tokenize(value_text))
    if not slot_key and subject and relation:
        slot_key = f"{subject}::{relation}"
    return AtomicClaim(
        id=claim_id,
        agent_id=raw.agent_id,
        kind=kind,
        subject=subject,
        relation=relation,
        value_text=value_text,
        value_tokens=tokens,
        qualifiers=qualifiers,
        polarity="support",
        event_time=event_time,
        observed_at=raw.observed_at,
        valid_from=session_date or raw.observed_at,
        valid_until=None,
        confidence=0.85,
        salience=1.0,
        source_episode_id=raw.id,
        source_spans=(span,),
        materialization_version=MATERIALIZATION_VERSION,
        materialized_at=now,
        slot_key=slot_key,
        version=0,
        origin_agent_id=raw.agent_id,
    )


def _with_version(claim: AtomicClaim, version: int) -> AtomicClaim:
    """Return a copy of claim with materialization_version set to version."""
    return AtomicClaim(
        id=claim.id,
        agent_id=claim.agent_id,
        kind=claim.kind,
        subject=claim.subject,
        relation=claim.relation,
        value_text=claim.value_text,
        value_tokens=claim.value_tokens,
        qualifiers=claim.qualifiers,
        polarity=claim.polarity,
        event_time=claim.event_time,
        observed_at=claim.observed_at,
        valid_from=claim.valid_from,
        valid_until=claim.valid_until,
        confidence=claim.confidence,
        salience=claim.salience,
        source_episode_id=claim.source_episode_id,
        source_spans=claim.source_spans,
        materialization_version=version,
        materialized_at=claim.materialized_at,
        slot_key=claim.slot_key,
        version=claim.version,
        origin_agent_id=claim.origin_agent_id,
    )
