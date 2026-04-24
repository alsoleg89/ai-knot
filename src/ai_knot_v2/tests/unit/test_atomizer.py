"""Sprint 3 — atomizer unit tests (no LOCOMO data, synthetic only)."""

from __future__ import annotations

from datetime import date

from ai_knot_v2.core._ulid import new_ulid
from ai_knot_v2.core.episode import RawEpisode
from ai_knot_v2.ops.atomizer import Atomizer


def _ep(text: str, speaker: str = "user", user_id: str = "user-1") -> RawEpisode:
    return RawEpisode(
        episode_id=new_ulid(),
        agent_id="agent-1",
        user_id=user_id,
        session_id="session-1",
        turn_index=0,
        speaker=speaker,  # type: ignore[arg-type]
        text=text,
        timestamp=1_700_000_000,
    )


SESSION_DATE = date(2024, 1, 15)


class TestAtomizerBasic:
    def test_simple_copula_extracts_atom(self) -> None:
        atoms = Atomizer().atomize(_ep("Alice is a doctor."), SESSION_DATE)
        assert len(atoms) >= 1
        predicates = {a.predicate for a in atoms}
        assert "is" in predicates

    def test_preference_extracts_atom(self) -> None:
        atoms = Atomizer().atomize(_ep("I love hiking."), SESSION_DATE)
        assert any(a.predicate == "prefers" for a in atoms)

    def test_possession_extracts_atom(self) -> None:
        atoms = Atomizer().atomize(_ep("Alice's salary is 120k."), SESSION_DATE)
        assert any("salary" in (a.subject or "").lower() for a in atoms)

    def test_work_at_extracts_atom(self) -> None:
        atoms = Atomizer().atomize(_ep("Alice works at Google."), SESSION_DATE)
        assert any(a.predicate == "works_at" for a in atoms)

    def test_first_person_resolves_to_user(self) -> None:
        atoms = Atomizer().atomize(_ep("I am a software engineer."), SESSION_DATE)
        assert len(atoms) >= 1
        user_atoms = [a for a in atoms if a.user_id == "user-1"]
        assert user_atoms, "first-person atom should have user_id set"

    def test_negation_detected(self) -> None:
        atoms = Atomizer().atomize(_ep("Alice is not a doctor."), SESSION_DATE)
        neg_atoms = [a for a in atoms if a.polarity == "neg"]
        assert neg_atoms, "negation should produce polarity=neg atom"

    def test_risk_classification_finance(self) -> None:
        atoms = Atomizer().atomize(_ep("My salary is 80000 per year."), SESSION_DATE)
        finance_atoms = [a for a in atoms if a.risk_class == "finance"]
        assert finance_atoms

    def test_risk_classification_medical(self) -> None:
        atoms = Atomizer().atomize(_ep("I have a doctor appointment tomorrow."), SESSION_DATE)
        # either scheduling or medical
        assert all(a.risk_class in ("scheduling", "medical", "has") for a in atoms)

    def test_temporal_yesterday_resolved(self) -> None:
        atoms = Atomizer().atomize(_ep("Alice was at work yesterday."), SESSION_DATE)
        temporal_atoms = [a for a in atoms if a.valid_from is not None]
        assert temporal_atoms, "yesterday should set valid_from"

    def test_empty_text_returns_empty(self) -> None:
        atoms = Atomizer().atomize(_ep(""), SESSION_DATE)
        assert atoms == []

    def test_multiple_facts_in_one_episode(self) -> None:
        atoms = Atomizer().atomize(_ep("Alice is a teacher. She loves hiking."), SESSION_DATE)
        assert len(atoms) >= 1

    def test_atoms_have_evidence_episode(self) -> None:
        ep = _ep("Alice is a doctor.")
        atoms = Atomizer().atomize(ep, SESSION_DATE)
        for atom in atoms:
            assert ep.episode_id in atom.evidence_episodes

    def test_atoms_have_valid_ulid(self) -> None:
        atoms = Atomizer().atomize(_ep("Alice is a teacher."), SESSION_DATE)
        for atom in atoms:
            assert len(atom.atom_id) == 26


class TestAtomizerRiskClassification:
    def test_preference_predicate_risk(self) -> None:
        from ai_knot_v2.core.risk import classify_risk

        cls, sev = classify_risk("prefers", "hiking")
        assert cls == "preference"
        assert sev == 0.2

    def test_finance_predicate_risk(self) -> None:
        from ai_knot_v2.core.risk import classify_risk

        cls, sev = classify_risk("has_salary", "120000")
        assert cls == "finance"
        assert sev > 0.5

    def test_ambient_predicate_risk(self) -> None:
        from ai_knot_v2.core.risk import classify_risk

        cls, sev = classify_risk("is", "happy")
        assert sev <= 0.2


class TestAtomizerSprint7:
    """Sprint 7 regression tests: event pattern, contractions, gerund, subject quality."""

    def test_event_pattern_went(self) -> None:
        atoms = Atomizer().atomize(_ep("I went to a doctor appointment yesterday."), SESSION_DATE)
        event_atoms = [a for a in atoms if "went" in a.predicate or a.predicate == "went"]
        assert event_atoms, "event pattern should extract 'went' atom"

    def test_event_pattern_visited(self) -> None:
        atoms = Atomizer().atomize(_ep("Alice visited the hospital last week."), SESSION_DATE)
        assert any(a.predicate in ("visited", "visit") for a in atoms)

    def test_contraction_expansion_am(self) -> None:
        atoms = Atomizer().atomize(_ep("I'm a software engineer."), SESSION_DATE)
        assert len(atoms) >= 1, "I'm should expand to I am for copula match"
        assert any(a.subject == "user-1" for a in atoms)

    def test_contraction_expansion_have(self) -> None:
        atoms = Atomizer().atomize(_ep("I've had diabetes for years."), SESSION_DATE)
        assert len(atoms) >= 1

    def test_gerund_start_extracts_with_speaker(self) -> None:
        atoms = Atomizer().atomize(
            _ep("Researching adoption agencies.", user_id="Caroline"), SESSION_DATE
        )
        gerund_atoms = [a for a in atoms if a.subject == "Caroline"]
        assert gerund_atoms, "gerund-start should use speaker as subject"

    def test_first_person_uses_speaker_name(self) -> None:
        atoms = Atomizer().atomize(_ep("I love painting.", user_id="Melanie"), SESSION_DATE)
        assert any(a.subject == "Melanie" for a in atoms)

    def test_question_word_subject_filtered(self) -> None:
        atoms = Atomizer().atomize(_ep("How have you been?"), SESSION_DATE)
        question_subj = [
            a for a in atoms if a.subject and a.subject.lower() in ("how", "what", "when", "where")
        ]
        assert not question_subj, "question-word subjects should be filtered"

    def test_conjunction_start_subject_filtered(self) -> None:
        atoms = Atomizer().atomize(_ep("And it was amazing."), SESSION_DATE)
        bad_subj = [a for a in atoms if a.subject and a.subject.lower().startswith("and")]
        assert not bad_subj

    def test_speaker_prefix_stripped(self) -> None:
        atoms = Atomizer().atomize(_ep("Caroline: I love hiking."), SESSION_DATE)
        # Should still extract atoms despite "Caroline: " prefix
        assert len(atoms) >= 1

    def test_per_episode_user_id_in_product_api(self) -> None:
        """MemoryAPI must use per-EpisodeIn user_id for first-person resolution."""
        from ai_knot_v2.api.product import MemoryAPI
        from ai_knot_v2.api.sdk import EpisodeIn, LearnRequest

        api = MemoryAPI(db_path=":memory:")
        api.learn(
            LearnRequest(
                episodes=[
                    EpisodeIn(text="I love hiking.", user_id="Alice"),
                    EpisodeIn(text="I hate spicy food.", user_id="Bob"),
                ]
            )
        )
        all_atoms = api._library.all_atoms()  # noqa: SLF001
        alice_atoms = [a for a in all_atoms if a.user_id == "Alice"]
        bob_atoms = [a for a in all_atoms if a.user_id == "Bob"]
        assert alice_atoms, "Alice's episode atoms should have user_id=Alice"
        assert bob_atoms, "Bob's episode atoms should have user_id=Bob"


class TestTemporalResolution:
    def test_yesterday(self) -> None:
        from ai_knot_v2.core.temporal import resolve_temporal

        vf, vu, gran = resolve_temporal("I did it yesterday", SESSION_DATE)
        assert vf is not None and vu is not None
        assert gran == "day"

    def test_last_week(self) -> None:
        from ai_knot_v2.core.temporal import resolve_temporal

        vf, vu, gran = resolve_temporal("last week we went hiking", SESSION_DATE)
        assert vf is not None
        assert gran == "interval"

    def test_in_2023(self) -> None:
        from ai_knot_v2.core.temporal import resolve_temporal

        vf, vu, gran = resolve_temporal("I started working in 2023", SESSION_DATE)
        assert vf is not None
        assert gran == "year"

    def test_no_temporal_returns_none(self) -> None:
        from ai_knot_v2.core.temporal import resolve_temporal

        vf, vu, gran = resolve_temporal("I am a doctor", SESSION_DATE)
        assert vf is None
        assert vu is None
        assert gran == "instant"
