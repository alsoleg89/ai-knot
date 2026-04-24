"""Deterministic atomizer: RawEpisode → list[MemoryAtom].

No LOCOMO-specific patterns. No LLM calls.
All extraction is rule-based and reproducible.

Sprint 7 changes:
- Speaker name used as subject for first-person resolution (not agent_id)
- Event/action verb pattern (went, visited, attended, received, ...)
- Subject length guard: reject subjects > 40 chars or < 2 chars
- Skip question-word subjects (how, what, who, when, why)
- Strip speaker-prefix format "Name: text" before extraction
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal

from ai_knot_v2.core._ulid import new_ulid
from ai_knot_v2.core.atom import MemoryAtom
from ai_knot_v2.core.episode import RawEpisode
from ai_knot_v2.core.groupoid import EntityGroupoid, resolve_speaker_entity
from ai_knot_v2.core.risk import classify_risk
from ai_knot_v2.core.temporal import resolve_temporal


@dataclass(frozen=True, slots=True)
class ClauseCandidate:
    subject_raw: str
    predicate_raw: str
    object_raw: str | None
    polarity: Literal["pos", "neg"]
    temporal_expr: str | None
    source_span: tuple[int, int]


# ---------------------------------------------------------------------------
# Regex extraction patterns
# ---------------------------------------------------------------------------

_NEGATION = re.compile(
    r"\b(not|never|no|don't|doesn't|didn't|won't|isn't|aren't|wasn't|weren't|can't)\b", re.I
)

# Question-word subjects are noise
_QUESTION_SUBJ = re.compile(r"^(how|what|when|where|who|why|which)$", re.I)

# Generic/trivial subjects
_TRIVIAL_SUBJ = re.compile(r"^(that|this|it|there|here|and|but|so)$", re.I)

# Past-tense and present action verbs (event/action pattern)
_EVENT_VERBS = (
    r"went|visited|attended|saw|met|received|got|found|made|created|started|joined|left"
    r"|moved|tried|began|finished|completed|took|gave|brought|sent|used|bought|learned"
    r"|studied|built|wrote|read|played|ran|walked|cooked|painted|drew|sang|won"
    r"|lost|helped|told|asked|showed|decided|realized|mentioned|said|talked|spoke"
    r"|called|texted|reached|heard|felt|thought|knew|remembered|forgot|wanted"
    r"|needed|liked|loved|hated|preferred|chose|picked|noticed|watched"
    r"|graduated|married|divorced|retired|hired|adopted|rescued|traveled|flew|drove"
    r"|sailed|competed|opened|closed|launched|released|published|submitted|applied"
    r"|baked|cleaned|fixed|repaired|hiked|swam|biked|cycled|broke|hurt|injured"
    r"|celebrated|organized|planned|booked|reserved|cancelled|scheduled|rescheduled"
    r"|signed|agreed|promised|confirmed|announced|discovered|invented|designed"
    r"|developed|deployed|shipped|sold|purchased|rented|leased|donated|volunteered"
    r"|take|takes|use|uses|wear|wears|drive|drives|run|runs|eat|eats|drink|drinks"
)

_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Alice's salary is 120k  /  My job is engineering
    (
        "possession",
        re.compile(
            r"([\w]+(?:\s+[\w]+)?)'s\s+([\w\s]+?)\s+(is|are|was|were)\s+(not\s+)?([\w\s,.$€£\d]+)",
            re.I,
        ),
    ),
    # I work(s/ed) at/for Company
    (
        "work_at",
        re.compile(
            r"([\w]+(?:\s+[\w]+)?)\s+(work(?:s|ed|ing)?)\s+(?:at|for|in)\s+([\w\s&,.-]+)",
            re.I,
        ),
    ),
    # Alice lives/lived/moved in/to City
    (
        "location",
        re.compile(
            r"([\w]+(?:\s+[\w]+)?)\s+(live[sd]?s?|lives?|moved?|stay(?:s|ed)?|reside[sd]?)\s+"
            r"(?:in|to|at|near|from)?\s*([\w\s,.-]+)",
            re.I,
        ),
    ),
    # I like/love/prefer/enjoy/hate X
    (
        "preference",
        re.compile(
            r"([\w]+(?:\s+[\w]+)?)\s+"
            r"(like[sd]?|love[sd]?|prefer[sd]?|enjoy[sd]?|hate[sd]?|dislike[sd]?|adore[sd]?)\s+"
            r"([\w\s,.!'-]+)",
            re.I,
        ),
    ),
    # Alice has/had a dog / I have a meeting
    (
        "has_obj",
        re.compile(
            r"([\w]+(?:\s+[\w]+)?)\s+(have|has|had)\s+(?:a\s+|an\s+|the\s+)?([\w\s]+)",
            re.I,
        ),
    ),
    # Alice is/was a doctor / I am a nurse
    (
        "copula",
        re.compile(
            r"([\w]+(?:\s+[\w]+)?)\s+(is|am|are|was|were)\s+(not\s+)?"
            r"(?:a\s+|an\s+|the\s+)?([\w\s,.-]+)",
            re.I,
        ),
    ),
    # I earn/make/get X per (year|month|week)
    (
        "income",
        re.compile(
            r"([\w]+(?:\s+[\w]+)?)\s+(?:earn|make|get|receive[sd]?)\s+"
            r"([\w\s$€£,.\d]+(?:per|a|each)\s+(?:year|month|week|day))",
            re.I,
        ),
    ),
    # I went to X / Caroline attended X / She visited X
    (
        "event",
        re.compile(
            rf"([\w]+(?:\s+[\w]+)?)\s+(?:just\s+|recently\s+|also\s+)?({_EVENT_VERBS})\s+"
            r"(?:to\s+|from\s+|at\s+|for\s+|a\s+|an\s+|the\s+)?([^\.\!\?]{3,60})",
            re.I,
        ),
    ),
]

_STRIP_RE = re.compile(r"[.!?,;:]+$")

# Detect "Speaker: text" prefix format
_SPEAKER_PREFIX_RE = re.compile(r"^([\w][\w\s]{0,30}):\s+(.+)$", re.S)

# Contraction expansion table
_CONTRACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bI'm\b", re.I), "I am"),
    (re.compile(r"\bI've\b", re.I), "I have"),
    (re.compile(r"\bI'd\b", re.I), "I had"),
    (re.compile(r"\bI'll\b", re.I), "I will"),
    (re.compile(r"\bI'm\b", re.I), "I am"),
    (re.compile(r"\bshe's\b", re.I), "she is"),
    (re.compile(r"\bhe's\b", re.I), "he is"),
    (re.compile(r"\bit's\b", re.I), "it is"),
    (re.compile(r"\bthey've\b", re.I), "they have"),
    (re.compile(r"\bwe've\b", re.I), "we have"),
    (re.compile(r"\bwon't\b", re.I), "will not"),
    (re.compile(r"\bcan't\b", re.I), "cannot"),
    (re.compile(r"\bdon't\b", re.I), "do not"),
    (re.compile(r"\bisn't\b", re.I), "is not"),
    (re.compile(r"\baren't\b", re.I), "are not"),
    (re.compile(r"\bwasn't\b", re.I), "was not"),
    (re.compile(r"\bweren't\b", re.I), "were not"),
]

# Subjects that are clause fragments (subordinating conjunctions + pronoun)
_SUBORD_SUBJ = re.compile(
    r"^(since|when|after|before|if|although|because|while|as|though|once|until|and|but|or|so)\b",
    re.I,
)

# Subjects ending with particle words indicate clause capture
_PARTICLE_END = re.compile(
    r"\b(to|of|or|and|but|so|just|only|also|then|yet|for|nor|as)\s*$",
    re.I,
)

# Gerund-start pattern: "Researching X", "Attending Y", "Working on Z"
_GERUND_START = re.compile(
    r"^(\w+ing)\s+(?:a\s+|an\s+|the\s+|about\s+|for\s+|on\s+|at\s+|with\s+)?([\w\s,'-]{2,50}?)"
    r"(?:\s*[-—–,\.!?]|$)",
    re.I,
)


def _expand_contractions(text: str) -> str:
    """Expand English contractions so pattern verbs can match."""
    for pat, replacement in _CONTRACTIONS:
        text = pat.sub(replacement, text)
    return text


def _clean(s: str) -> str:
    return _STRIP_RE.sub("", s).strip()


def _strip_speaker_prefix(text: str) -> tuple[str, str | None]:
    """Strip 'Speaker: text' prefix. Returns (clean_text, speaker_from_prefix|None)."""
    m = _SPEAKER_PREFIX_RE.match(text)
    if m:
        prefix = m.group(1).strip()
        # Only treat as speaker prefix if it looks like a name (≤ 3 words, no verb)
        if len(prefix.split()) <= 3 and not re.search(
            r"\b(is|are|was|have|had|do|did)\b", prefix, re.I
        ):
            return m.group(2), prefix
    return text, None


def _detect_polarity(group: str | None, surrounding: str) -> Literal["pos", "neg"]:
    if group and re.search(r"\bnot\b", group, re.I):
        return "neg"
    if _NEGATION.search(surrounding[:30]):
        return "neg"
    return "pos"


def _sentence_split(text: str) -> list[str]:
    """Rough sentence splitter (no NLTK dependency)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _subject_ok(subj: str) -> bool:
    """Return True if subject is plausible (not trivial, not too long)."""
    s = subj.strip()
    if not s or len(s) > 40:
        return False
    if _TRIVIAL_SUBJ.match(s):
        return False
    if _QUESTION_SUBJ.match(s):
        return False
    # Reject subjects starting with subordinating conjunctions
    if _SUBORD_SUBJ.match(s):
        return False
    # Reject subjects ending with particle/conjunction words (clause capture)
    if _PARTICLE_END.search(s):
        return False
    # Skip pure punctuation or digits (but allow single chars like "I")
    return not (len(s) > 1 and re.match(r"^[\d\W]+$", s))


def _gerund_clauses(text: str, implied_subject: str) -> list[ClauseCandidate]:
    """Extract S-V-O from gerund-start sentences using implied_subject as agent."""
    candidates: list[ClauseCandidate] = []
    for sent in _sentence_split(text):
        m = _GERUND_START.match(sent)
        if not m:
            continue
        verb_ing = m.group(1)  # e.g. "Researching"
        obj = _clean(m.group(2)) if m.group(2) else None
        if not obj or len(obj) < 3:
            continue
        # Canonicalize gerund → base form heuristic
        verb_base = re.sub(r"ing$", "", verb_ing).rstrip("e") or verb_ing
        candidates.append(
            ClauseCandidate(
                subject_raw=implied_subject,
                predicate_raw=verb_base,
                object_raw=obj,
                polarity="pos",
                temporal_expr=sent,
                source_span=(0, len(sent)),
            )
        )
    return candidates


def _extract_clauses(text: str) -> list[ClauseCandidate]:
    candidates: list[ClauseCandidate] = []
    seen_predicates: set[str] = set()

    for sent in _sentence_split(text):
        for pattern_name, pattern in _PATTERNS:
            for m in pattern.finditer(sent):
                if pattern_name == "possession":
                    subj = f"{_clean(m.group(1))}'s {_clean(m.group(2))}"
                    pred = m.group(3).lower()
                    neg_grp = m.group(4)
                    obj = _clean(m.group(5)) if (m.lastindex or 0) >= 5 else None
                    polarity = _detect_polarity(neg_grp, sent)
                elif pattern_name in ("work_at", "location", "preference", "income"):
                    subj = _clean(m.group(1))
                    pred = _clean(m.group(2)).lower()
                    obj = _clean(m.group(3)) if (m.lastindex or 0) >= 3 else None
                    polarity = _detect_polarity(None, sent)
                elif pattern_name == "has_obj":
                    subj = _clean(m.group(1))
                    pred = "has"
                    obj = _clean(m.group(3))
                    polarity = _detect_polarity(None, sent)
                elif pattern_name == "copula":
                    subj = _clean(m.group(1))
                    pred = "is"
                    neg_grp = m.group(3)
                    obj = _clean(m.group(4)) if (m.lastindex or 0) >= 4 else None
                    polarity = _detect_polarity(neg_grp, sent)
                elif pattern_name == "event":
                    subj = _clean(m.group(1))
                    pred = _clean(m.group(2)).lower()
                    obj = _clean(m.group(3)) if (m.lastindex or 0) >= 3 else None
                    polarity = _detect_polarity(None, sent)
                else:
                    continue

                if not subj or not pred or not obj:
                    continue

                if not _subject_ok(subj):
                    continue

                # Deduplicate within sentence
                key = f"{subj.lower()}:{pred}:{(obj or '').lower()}"
                if key in seen_predicates:
                    continue
                seen_predicates.add(key)

                span = (m.start(), m.end())
                candidates.append(
                    ClauseCandidate(
                        subject_raw=subj,
                        predicate_raw=pred,
                        object_raw=obj,
                        polarity=polarity,
                        temporal_expr=sent if sent != m.group(0) else None,
                        source_span=span,
                    )
                )

    return candidates


def _canonical_predicate(predicate_raw: str, pattern_name: str | None = None) -> str:
    """Normalize predicate to a snake_case canonical form."""
    p = predicate_raw.lower().strip()
    mapping = {
        "is": "is",
        "am": "is",
        "are": "is",
        "was": "is",
        "were": "is",
        "has": "has",
        "have": "has",
        "had": "has",
        "likes": "prefers",
        "like": "prefers",
        "loved": "prefers",
        "love": "prefers",
        "prefers": "prefers",
        "prefer": "prefers",
        "enjoyed": "prefers",
        "enjoy": "prefers",
        "hates": "dislikes",
        "hate": "dislikes",
        "dislikes": "dislikes",
        "dislike": "dislikes",
        "works": "works_at",
        "work": "works_at",
        "worked": "works_at",
        "lives": "lives_in",
        "live": "lives_in",
        "lived": "lives_in",
        "moved": "moved_to",
    }
    return mapping.get(p, re.sub(r"[^a-z0-9_]", "_", p))


_FIRST_PERSON = re.compile(r"^(i|me|my|mine|myself)$", re.I)

# Third-person pronouns that can be resolved via within-session coreference
_THIRD_SG = re.compile(r"^(she|he|her|him|his|hers)$", re.I)
_THIRD_PL = re.compile(r"^(they|them|their|theirs)$", re.I)


class Atomizer:
    """Converts RawEpisode objects into MemoryAtom lists.

    Maintains within-session coreference state: after a named entity appears
    as subject, subsequent third-person pronouns in the same session resolve
    to that entity's orbit (Sprint 12 holonomy + coreference activation).
    """

    def __init__(self, groupoid: EntityGroupoid | None = None) -> None:
        self._groupoid = groupoid or EntityGroupoid()
        # Within-session coreference anchor: (session_id, orbit_id, display_name)
        self._coref: tuple[str, str, str] | None = None

    def atomize(
        self,
        episode: RawEpisode,
        session_date: date,
    ) -> list[MemoryAtom]:
        """Extract MemoryAtom list from a single episode."""
        speaker_orbit = resolve_speaker_entity(episode.speaker, episode.user_id, episode.agent_id)

        # Resolve speaker display name: user_id carries real name when provided
        speaker_name: str = episode.user_id or episode.agent_id

        # Strip "Speaker: text" prefix, then expand contractions for better coverage
        clean_text, _prefix_speaker = _strip_speaker_prefix(episode.text)
        expanded = _expand_contractions(clean_text)

        clauses = _extract_clauses(expanded)
        # Gerund-start sentences have no explicit subject → use speaker as agent
        clauses = clauses + _gerund_clauses(expanded, speaker_name)
        atoms: list[MemoryAtom] = []

        for clause in clauses:
            subj_raw = clause.subject_raw

            # Entity resolution: first-person → speaker name
            if _FIRST_PERSON.match(subj_raw):
                entity_orbit_id = speaker_orbit
                subject: str = speaker_name
            elif (_THIRD_SG.match(subj_raw) or _THIRD_PL.match(subj_raw)) and (
                self._coref is not None and self._coref[0] == episode.session_id
            ):
                # Within-session coreference: resolve pronoun to last named entity
                _, entity_orbit_id, coref_name = self._coref
                subject = coref_name  # use resolved name for display
            else:
                subject = subj_raw
                entity_orbit_id = self._groupoid.resolve(subj_raw)
                # Update coreference: named entity (capitalized, non-pronoun)
                if (
                    subj_raw
                    and subj_raw[0].isupper()
                    and not _FIRST_PERSON.match(subj_raw)
                    and not _THIRD_SG.match(subj_raw)
                    and not _THIRD_PL.match(subj_raw)
                ):
                    self._coref = (episode.session_id, entity_orbit_id, subject)

            # Temporal resolution
            source_text = clause.temporal_expr or episode.text
            valid_from, valid_until, granularity = resolve_temporal(source_text, session_date)

            # Canonical predicate
            canon_pred = _canonical_predicate(clause.predicate_raw)

            # Risk classification — include subject for better context
            risk_class, risk_severity = classify_risk(
                f"{subject.lower()} {canon_pred}", clause.object_raw
            )

            # Protection energy heuristic: high-risk facts get higher initial energy
            protection_energy = min(1.0, risk_severity * 2.0)

            # Regret charge: risk × irreducibility (Sprint 1 placeholder: 1.0)
            regret_charge = risk_severity * 1.0

            atoms.append(
                MemoryAtom(
                    atom_id=new_ulid(),
                    agent_id=episode.agent_id,
                    user_id=episode.user_id,
                    variables=(subject.lower().replace(" ", "_"),),
                    causal_graph=(),
                    kernel_kind="point",
                    kernel_payload={},
                    intervention_domain=(subject.lower().replace(" ", "_"),),
                    predicate=canon_pred,
                    subject=subject,
                    object_value=clause.object_raw,
                    polarity=clause.polarity,
                    valid_from=valid_from,
                    valid_until=valid_until,
                    observation_time=episode.timestamp,
                    belief_time=episode.timestamp,
                    granularity=granularity,  # type: ignore[arg-type]
                    entity_orbit_id=entity_orbit_id,
                    transport_provenance=(episode.session_id,),
                    depends_on=(),
                    depended_by=(),
                    risk_class=risk_class,
                    risk_severity=risk_severity,
                    regret_charge=regret_charge,
                    irreducibility_score=1.0,
                    protection_energy=protection_energy,
                    action_affect_mask=0,
                    credence=0.9,
                    evidence_episodes=(episode.episode_id,),
                    synthesis_method="regex",
                    validation_tests=(),
                    contradiction_events=(),
                )
            )

        return atoms
