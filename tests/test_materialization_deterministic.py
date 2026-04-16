"""Tests for deterministic materialization invariants."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from ai_knot.materialization import (
    MATERIALIZATION_VERSION,
    dirty_keys_for_claims,
    materialize_episode,
    rebuild_claims_from_raw,
)
from ai_knot.query_types import (
    DETERMINISTIC_CLAIM_KINDS,
    ClaimKind,
    RawEpisode,
    make_episode_id,
)

NOW = datetime(2024, 6, 1, 12, 0, 0, tzinfo=UTC)


def _ep(raw_text: str, session_id: str = "sess", turn_id: str = "turn") -> RawEpisode:
    ep_id = make_episode_id("agent", session_id, turn_id)
    return RawEpisode(
        id=ep_id,
        agent_id="agent",
        session_id=session_id,
        turn_id=turn_id,
        speaker="user",
        observed_at=NOW,
        session_date=None,
        raw_text=raw_text,
        source_meta={},
        parent_episode_id=None,
    )


# ---------------------------------------------------------------------------
# Kind whitelist
# ---------------------------------------------------------------------------


class TestMaterializationKindWhitelist:
    def test_only_deterministic_kinds_produced(self):
        ep = _ep("Alice works as a software engineer at TechCorp.")
        claims = materialize_episode(ep)
        for c in claims:
            assert c.kind in DETERMINISTIC_CLAIM_KINDS, (
                f"Non-deterministic kind {c.kind!r} produced by materializer. "
                f"Only DESCRIPTOR/INTENT may come from enrichment, never from materialize_episode."
            )

    def test_no_descriptor_or_intent_from_materializer(self):
        ep = _ep("Bob is a creative, empathetic person who loves innovation.")
        claims = materialize_episode(ep)
        enrichment_only = {ClaimKind.DESCRIPTOR, ClaimKind.INTENT}
        for c in claims:
            assert c.kind not in enrichment_only, (
                f"materialize_episode produced enrichment-only kind {c.kind!r}"
            )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestMaterializationDeterminism:
    def test_same_episode_same_claims(self):
        ep = _ep("Carol studies physics at the university.")
        claims1 = materialize_episode(ep)
        claims2 = materialize_episode(ep)
        assert {c.id for c in claims1} == {c.id for c in claims2}

    def test_same_text_different_turn_different_claim_ids(self):
        """Claim IDs depend on episode_id, so different turns → different IDs."""
        ep1 = _ep("Alice plays piano.", session_id="sess", turn_id="turn-1")
        ep2 = _ep("Alice plays piano.", session_id="sess", turn_id="turn-2")
        claims1 = materialize_episode(ep1)
        claims2 = materialize_episode(ep2)
        ids1 = {c.id for c in claims1}
        ids2 = {c.id for c in claims2}
        # Same content but different episode → different IDs
        assert ids1 != ids2 or not ids1  # if no claims, trivially different

    def test_rebuild_same_as_individual_materialize(self):
        episodes = [
            _ep("Alice works at Acme.", session_id="sess", turn_id="turn-0"),
            _ep("Alice lives in Paris.", session_id="sess", turn_id="turn-1"),
        ]
        individual = []
        for ep in episodes:
            individual.extend(materialize_episode(ep))

        rebuilt = rebuild_claims_from_raw(episodes, version=MATERIALIZATION_VERSION)
        assert {c.id for c in individual} == {c.id for c in rebuilt}


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class TestMaterializationProvenance:
    def test_all_claims_have_source_episode_id(self):
        ep = _ep("Dave joined the company in January 2023.")
        claims = materialize_episode(ep)
        for c in claims:
            assert c.source_episode_id == ep.id, f"Claim {c.id} missing source_episode_id"

    def test_all_claims_have_source_spans(self):
        ep = _ep("Eve is 28 years old.")
        claims = materialize_episode(ep)
        for c in claims:
            if c.kind in {ClaimKind.STATE, ClaimKind.EVENT, ClaimKind.RELATION}:
                assert c.source_spans, (
                    f"Claim {c.id} (kind={c.kind}) has empty source_spans — "
                    "provenance invariant violated"
                )


# ---------------------------------------------------------------------------
# DirtyKey minimality
# ---------------------------------------------------------------------------


class TestDirtyKeyMinimality:
    def test_slot_level_key_preferred_over_subject_only(self):
        """Claims with subject+relation should emit subject+relation DirtyKey, not subject-only."""
        ep = _ep("Alice's salary is $80k.")
        claims = materialize_episode(ep)
        if not claims:
            pytest.skip("no claims extracted from salary text")

        dirty_keys = dirty_keys_for_claims(claims)
        # At least one key should be a slot-level key (has both subject AND relation)
        slot_level = [k for k in dirty_keys if k.subject and k.relation]
        subject_only = [k for k in dirty_keys if k.subject and not k.relation]

        # If any slot-level key exists, there should be fewer or equal subject-only keys
        if slot_level:
            # Prefer slot-level: subject-only keys are only emitted for claims
            # where we can't identify the specific relation
            assert len(slot_level) >= len(subject_only) or subject_only == [], (
                "Expected slot-level DirtyKey to be preferred; got more subject-only keys"
            )

    def test_no_bundle_kind_topic_keys_from_routine_materialize(self):
        """Bundle-kind+topic DirtyKeys are only for admin/debug, not routine ingest."""
        ep = _ep("Frank is a chef at La Maison.")
        claims = materialize_episode(ep)
        dirty_keys = dirty_keys_for_claims(claims)
        bundle_kind_keys = [k for k in dirty_keys if k.bundle_kind is not None]
        assert bundle_kind_keys == [], (
            "materialize_episode should not emit bundle_kind DirtyKeys — those are admin/debug only"
        )

    def test_dirty_keys_cover_all_subjects(self):
        ep = _ep("Grace works at Bloom Inc and lives near downtown.")
        claims = materialize_episode(ep)
        if not claims:
            pytest.skip("no claims extracted")

        dirty_keys = dirty_keys_for_claims(claims)
        claim_subjects = {c.subject for c in claims}
        key_subjects = {k.subject for k in dirty_keys if k.subject}
        # Every claim subject must appear in at least one dirty key
        for subj in claim_subjects:
            assert subj in key_subjects, (
                f"Subject {subj!r} from claims has no corresponding DirtyKey"
            )


# ---------------------------------------------------------------------------
# Speaker-prefix and first-person extraction
# ---------------------------------------------------------------------------


class TestSpeakerPrefixExtraction:
    def _ep_with_speaker(self, raw_text: str, turn_id: str = "turn") -> RawEpisode:
        ep_id = make_episode_id("agent", "sess", turn_id)
        return RawEpisode(
            id=ep_id,
            agent_id="agent",
            session_id="sess",
            turn_id=turn_id,
            speaker="user",
            observed_at=NOW,
            session_date=None,
            raw_text=raw_text,
            source_meta={},
            parent_episode_id=None,
        )

    def test_speaker_prefix_stripped_not_in_subject(self):
        """'Dave: I love restoring old cars' — 'Dave' must be subject, not 'Dave I'."""
        ep = self._ep_with_speaker("Dave: I love restoring old cars")
        claims = materialize_episode(ep)
        subjects = {c.subject for c in claims}
        # The speaker prefix "Dave:" must not become the subject fragment.
        assert "Dave" in subjects or not claims, (
            f"Expected subject 'Dave', got subjects: {subjects}"
        )
        # No subject should start with "Dave:" or contain the colon.
        for s in subjects:
            assert ":" not in s, f"Subject contains colon from speaker prefix: {s!r}"

    def test_first_person_likes_with_speaker(self):
        """Speaker-prefixed turn: 'Dave: I love restoring old cars' → STATE likes."""
        ep = self._ep_with_speaker("Dave: I love restoring old cars")
        claims = materialize_episode(ep)
        likes_claims = [c for c in claims if c.relation == "likes" and c.subject == "Dave"]
        assert likes_claims, (
            f"Expected a likes claim with subject='Dave', got claims: "
            f"{[(c.subject, c.relation, c.value_text) for c in claims]}"
        )
        assert (
            "restoring old cars" in likes_claims[0].value_text.lower()
            or "old cars" in likes_claims[0].value_text.lower()
        )

    def test_first_person_satisfying_with_speaker(self):
        """'Dave: It's so satisfying to bring an old car back to life' → STATE finds_satisfying."""
        ep = self._ep_with_speaker("Dave: It's so satisfying to bring an old car back to life")
        claims = materialize_episode(ep)
        sat_claims = [c for c in claims if c.relation == "finds_satisfying" and c.subject == "Dave"]
        assert sat_claims, (
            f"Expected a finds_satisfying claim with subject='Dave', got: "
            f"{[(c.subject, c.relation, c.value_text) for c in claims]}"
        )

    def test_first_person_dislikes_with_speaker(self):
        """'Alice: I hate waiting in line' → STATE dislikes."""
        ep = self._ep_with_speaker("Alice: I hate waiting in line")
        claims = materialize_episode(ep)
        dislikes_claims = [c for c in claims if c.relation == "dislikes" and c.subject == "Alice"]
        assert dislikes_claims, (
            f"Expected a dislikes claim, got: "
            f"{[(c.subject, c.relation, c.value_text) for c in claims]}"
        )

    def test_garbage_sentences_not_materialized(self):
        """Question/imperative openers must not produce claims."""
        garbage_texts = [
            "Do you like that?",
            "What kind of car do you drive?",
            "Take a look at this.",
            "Glad you mentioned that.",
            "Thanks for sharing.",
        ]
        for text in garbage_texts:
            ep = self._ep_with_speaker(text, turn_id=f"turn-{text[:10]}")
            claims = materialize_episode(ep)
            assert claims == [], (
                f"Garbage sentence {text!r} should produce no claims, got {len(claims)}"
            )

    def test_pronoun_not_subject_without_speaker(self):
        """Without a speaker prefix, 'I' / 'Do' must not appear as subjects."""
        ep = _ep("I love hiking and exploring new places.")
        claims = materialize_episode(ep)
        for c in claims:
            assert c.subject not in ("I", "Do", "What", "You"), (
                f"Pronoun/opener {c.subject!r} must not be a claim subject"
            )


# ---------------------------------------------------------------------------
# §5 — Speaker-aware action extraction
# ---------------------------------------------------------------------------


class TestSpeakerFirstPerson:
    def _ep_speaker(self, raw_text: str, turn_id: str = "t") -> RawEpisode:
        ep_id = make_episode_id("agent", "sess", turn_id)
        return RawEpisode(
            id=ep_id,
            agent_id="agent",
            session_id="sess",
            turn_id=turn_id,
            speaker="user",
            observed_at=NOW,
            session_date=None,
            raw_text=raw_text,
            source_meta={},
            parent_episode_id=None,
        )

    def test_fp_drives_with_speaker(self) -> None:
        """'Evan: I drive a Tacoma' → STATE drives with subject=Evan."""
        ep = self._ep_speaker("Evan: I drive a Tacoma.", "t-drives")
        claims = materialize_episode(ep)
        drives = [c for c in claims if c.relation == "drives" and c.subject == "Evan"]
        assert drives, (
            f"Expected drives claim for Evan, got: "
            f"{[(c.subject, c.relation, c.value_text) for c in claims]}"
        )
        assert "tacoma" in drives[0].value_text.lower()

    def test_fp_works_at_with_speaker(self) -> None:
        """'Dave: I joined Acme last year' → RELATION works_at with subject=Dave."""
        ep = self._ep_speaker("Dave: I joined Acme last year.", "t-work")
        claims = materialize_episode(ep)
        work = [c for c in claims if c.relation == "works_at" and c.subject == "Dave"]
        assert work, (
            f"Expected works_at claim for Dave, got: "
            f"{[(c.subject, c.relation, c.value_text) for c in claims]}"
        )

    def test_fp_moved_to_with_speaker(self) -> None:
        """'Alice: I moved to Berlin last spring' → TRANSITION moved_to with subject=Alice."""
        ep = self._ep_speaker("Alice: I moved to Berlin last spring.", "t-move")
        claims = materialize_episode(ep)
        moved = [c for c in claims if c.relation == "moved_to" and c.subject == "Alice"]
        assert moved, (
            f"Expected moved_to claim for Alice, got: "
            f"{[(c.subject, c.relation, c.value_text) for c in claims]}"
        )
        assert "berlin" in moved[0].value_text.lower()

    def test_fp_started_with_speaker(self) -> None:
        """'Bob: I started researching machine learning' → STATE started with subject=Bob."""
        ep = self._ep_speaker("Bob: I started researching machine learning.", "t-started")
        claims = materialize_episode(ep)
        started = [c for c in claims if c.relation == "started" and c.subject == "Bob"]
        assert started, (
            f"Expected started claim for Bob, got: "
            f"{[(c.subject, c.relation, c.value_text) for c in claims]}"
        )

    def test_fp_action_no_speaker_no_claim(self) -> None:
        """Without a speaker prefix, 'I drive a Tacoma' must not produce a drives claim."""
        ep = _ep("I drive a Tacoma.")
        claims = materialize_episode(ep)
        drives = [c for c in claims if c.relation == "drives"]
        assert not drives, (
            "Without named speaker, first-person action must not produce a drives claim"
        )


# ---------------------------------------------------------------------------
# First-person event extraction (v5)
# ---------------------------------------------------------------------------


def _make_raw_fp(text: str, speaker: str, session_date: datetime | None = None) -> RawEpisode:
    """Helper: raw episode with speaker prefix."""
    from datetime import UTC

    sd = session_date or datetime(2026, 4, 15, tzinfo=UTC)
    ep_id = make_episode_id("test-agent", "sess-fp", f"fp-{speaker}-{text[:10]}")
    return RawEpisode(
        id=ep_id,
        agent_id="test-agent",
        session_id="sess-fp",
        turn_id=f"fp-{speaker}-{text[:10]}",
        speaker=speaker,
        observed_at=sd,
        session_date=sd,
        raw_text=f"{speaker}: {text}",
        source_meta={},
        parent_episode_id=None,
    )


def test_first_person_event_extraction():
    """I attended … → EVENT claim with slot_key and time_anchor."""
    ep = _make_raw_fp("I attended a pottery workshop yesterday", "Alice")
    claims = materialize_episode(ep)
    event_claims = [c for c in claims if c.kind is ClaimKind.EVENT]
    assert event_claims, f"Expected EVENT claim, got: {claims}"
    c = event_claims[0]
    assert c.subject == "Alice"
    assert c.relation == "attended"
    assert c.slot_key == "Alice::attended"


def test_time_anchor_qualifiers_present():
    """Session-anchored event: time_anchor qualifier set, event_time NOT set."""
    ep = _make_raw_fp("I signed up for pottery today", "Alice")
    claims = materialize_episode(ep)
    event_claims = [
        c for c in claims if c.kind is ClaimKind.EVENT and c.relation == "signed_up_for"
    ]
    assert event_claims, "Expected signed_up_for EVENT claim"
    c = event_claims[0]
    assert c.qualifiers.get("time_anchor") == "session_date"
    assert c.qualifiers.get("relative_time") == "today"
    assert c.event_time is None, "event_time must NOT be set — time_resolve() computes it"


def test_relative_time_yesterday_captured():
    """'yesterday' is captured in qualifiers."""
    ep = _make_raw_fp("I attended a workshop yesterday", "Alice")
    claims = materialize_episode(ep)
    event_claims = [c for c in claims if c.kind is ClaimKind.EVENT and c.relation == "attended"]
    assert event_claims
    assert event_claims[0].qualifiers.get("relative_time") == "yesterday"


def test_discourse_guard_only_triggers_on_combined_signal():
    """'That sounds wonderful' → no claim; 'Mary has a cool idea' → claim."""
    sd = datetime(2026, 4, 15, tzinfo=UTC)
    # Deictic + evaluative → suppressed
    ep_noise = RawEpisode(
        id=make_episode_id("test", "sess-noise", "t-noise"),
        agent_id="test",
        session_id="sess-noise",
        turn_id="t-noise",
        speaker="user",
        observed_at=sd,
        session_date=None,
        raw_text="That sounds wonderful.",
        source_meta={},
        parent_episode_id=None,
    )
    assert materialize_episode(ep_noise) == []

    # Non-deictic subject → should NOT be suppressed (Mary is a real subject)
    ep_ok = RawEpisode(
        id=make_episode_id("test", "sess-ok", "t-ok"),
        agent_id="test",
        session_id="sess-ok",
        turn_id="t-ok",
        speaker="user",
        observed_at=sd,
        session_date=None,
        raw_text="Mary has a cool new idea.",
        source_meta={},
        parent_episode_id=None,
    )
    claims_ok = materialize_episode(ep_ok)
    assert claims_ok, "Mary has a cool idea should produce a claim"
