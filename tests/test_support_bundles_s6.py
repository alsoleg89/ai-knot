"""S6 regression tests — entity-topic filter and stable bundle IDs."""

from __future__ import annotations

from datetime import UTC, datetime

from ai_knot.materialization import materialize_episode
from ai_knot.query_types import (
    AtomicClaim,
    BundleKind,
    ClaimKind,
    RawEpisode,
    stable_bundle_id,
)
from ai_knot.support_bundles import build_entity_topic_bundles


def _make_raw(text: str, agent_id: str = "test") -> RawEpisode:
    return RawEpisode(
        id="ep1",
        agent_id=agent_id,
        session_id="s1",
        turn_id="t1",
        speaker="user",
        observed_at=datetime.now(UTC),
        raw_text=text,
        session_date=None,
    )


def _make_claim(subject: str, relation: str = "state", agent_id: str = "test") -> AtomicClaim:
    now = datetime.now(UTC)
    return AtomicClaim(
        id=f"{subject}:{relation}",
        agent_id=agent_id,
        kind=ClaimKind.STATE,
        subject=subject,
        relation=relation,
        value_text=f"{subject} {relation} value",
        value_tokens=(),
        qualifiers={},
        polarity="support",
        event_time=None,
        observed_at=now,
        valid_from=now,
        valid_until=None,
        confidence=0.85,
        salience=1.0,
        source_episode_id="ep1",
        source_spans=((0, 10),),
        materialization_version=4,
        materialized_at=now,
        slot_key=f"{subject}::{relation}",
        version=0,
        origin_agent_id=agent_id,
    )


class TestEntityTopicFilter:
    def test_proper_noun_single_claim_survives(self) -> None:
        """Single claim about a proper noun should create a bundle."""
        claims = [_make_claim("Alice")]
        bundles, _ = build_entity_topic_bundles(
            claims, agent_id="test", materialization_version=4
        )
        subjects = [b.topic for b in bundles]
        assert "Alice" in subjects

    def test_multi_claim_noise_subject_survives(self) -> None:
        """Subject with 2+ claims survives even if it looks like a filler."""
        claims = [_make_claim("My dream"), _make_claim("My dream", "role")]
        bundles, _ = build_entity_topic_bundles(
            claims, agent_id="test", materialization_version=4
        )
        subjects = [b.topic for b in bundles]
        assert "My dream" in subjects

    def test_single_claim_noise_subject_filtered(self) -> None:
        """Single claim with a discourse subject (starts with 'My') should be filtered."""
        # "My" is in the discourse blocklist, so a single-claim bundle is skipped
        claims = [_make_claim("My new car")]
        bundles, _ = build_entity_topic_bundles(
            claims, agent_id="test", materialization_version=4
        )
        subjects = [b.topic for b in bundles]
        # "My" is in the blocklist — filtered unless there are 2+ claims
        assert "My new car" not in subjects

    def test_discourse_pronoun_filtered(self) -> None:
        """'It' and 'This' should be filtered as discourse subjects."""
        claims = [_make_claim("It")]
        bundles, _ = build_entity_topic_bundles(
            claims, agent_id="test", materialization_version=4
        )
        subjects = [b.topic for b in bundles]
        assert "It" not in subjects

    def test_proper_noun_two_words_survives(self) -> None:
        """Two-word proper noun with one claim should create a bundle."""
        claims = [_make_claim("John Smith")]
        bundles, _ = build_entity_topic_bundles(
            claims, agent_id="test", materialization_version=4
        )
        subjects = [b.topic for b in bundles]
        assert "John Smith" in subjects

    def test_stable_bundle_id_deterministic(self) -> None:
        """Same (kind, topic) always yields the same bundle ID."""
        id1 = stable_bundle_id(BundleKind.ENTITY_TOPIC, "Alice")
        id2 = stable_bundle_id(BundleKind.ENTITY_TOPIC, "Alice")
        assert id1 == id2

    def test_stable_bundle_id_different_kinds_differ(self) -> None:
        """Different bundle kinds for the same topic yield different IDs."""
        id1 = stable_bundle_id(BundleKind.ENTITY_TOPIC, "Alice")
        id2 = stable_bundle_id(BundleKind.STATE_TIMELINE, "Alice")
        assert id1 != id2

    def test_rebuild_produces_same_bundle_id(self) -> None:
        """Two builds from the same claims produce the same bundle ID."""
        claims = [_make_claim("Alice")]
        bundles1, _ = build_entity_topic_bundles(
            claims, agent_id="test", materialization_version=4
        )
        bundles2, _ = build_entity_topic_bundles(
            claims, agent_id="test", materialization_version=4
        )
        assert bundles1[0].id == bundles2[0].id


class TestMaterializationSpeakerAware:
    def test_speaker_prefix_stripped(self) -> None:
        """Speaker prefix 'Alice: ' is stripped before extraction."""
        raw = _make_raw("Alice: I like hiking and cooking.")
        claims = materialize_episode(raw)
        # Should extract claims attributed to Alice (speaker), not "I"
        subjects = [c.subject for c in claims]
        # "I" should NOT appear as a subject (pronoun guard)
        assert "I" not in subjects

    def test_fp_likes_generates_speaker_claim(self) -> None:
        """'I like X' with speaker prefix generates a likes claim for the speaker."""
        raw = _make_raw("Alice: I love hiking.")
        claims = materialize_episode(raw)
        like_claims = [c for c in claims if c.relation == "likes"]
        assert any(c.subject == "Alice" for c in like_claims), (
            f"Expected Alice::likes claim, got: {[(c.subject, c.relation) for c in claims]}"
        )

    def test_garbage_sentence_filtered(self) -> None:
        """Questions and filler sentences should not produce claims."""
        raw = _make_raw("What do you think about hiking?")
        claims = materialize_episode(raw)
        # Question should not produce any claims
        assert len(claims) == 0

    def test_materialization_version_is_4(self) -> None:
        """MATERIALIZATION_VERSION should be 4 for S6."""
        from ai_knot.materialization import MATERIALIZATION_VERSION

        assert MATERIALIZATION_VERSION == 4


class TestQueryContractTemporalRouting:
    def test_temporal_question_routes_to_time_resolve(self) -> None:
        """A 'when' question should produce time_axis=EVENT and route to time_resolve."""
        from ai_knot.query_contract import analyze_query, derive_answer_contract

        frame = analyze_query("When did Alice start working at TechCorp?")
        contract = derive_answer_contract(frame)
        from ai_knot.query_types import TimeAxis

        assert contract.time_axis is TimeAxis.EVENT

    def test_temporal_query_geometry_precedes_relation(self) -> None:
        """Temporal axis is set even when a relation verb is present."""
        from ai_knot.query_contract import analyze_query, derive_answer_contract
        from ai_knot.query_types import TimeAxis

        frame = analyze_query("When did Bob start working at Acme?")
        contract = derive_answer_contract(frame)
        assert contract.time_axis is TimeAxis.EVENT
