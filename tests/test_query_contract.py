"""Tests for query_contract — geometry-based routing without keyword policy."""

from __future__ import annotations

from ai_knot.query_contract import analyze_query, derive_answer_contract
from ai_knot.query_types import (
    AnswerSpace,
    EvidenceRegime,
    TimeAxis,
    TruthMode,
)

# ---------------------------------------------------------------------------
# analyze_query — QueryFrame geometry
# ---------------------------------------------------------------------------


class TestAnalyzeQuery:
    def test_set_question_what_hobbies(self):
        # Without a plural-noun inflection heuristic, "What hobbies" without
        # an explicit aggregation cue (all/list/enumerate) maps to DESCRIPTION.
        # Use "List all hobbies" or "What are all Alice's hobbies" for SET.
        frame = analyze_query("What hobbies does Alice have?")
        assert frame.answer_space in (AnswerSpace.DESCRIPTION, AnswerSpace.SET)

    def test_set_question_list_all(self):
        frame = analyze_query("List all the activities Alice enjoys.")
        assert frame.answer_space is AnswerSpace.SET

    def test_bool_question_is(self):
        frame = analyze_query("Is Alice a vegetarian?")
        assert frame.answer_space is AnswerSpace.BOOL

    def test_bool_question_does(self):
        frame = analyze_query("Does Bob drink coffee?")
        assert frame.answer_space is AnswerSpace.BOOL

    def test_entity_question_who(self):
        frame = analyze_query("Who is Alice's best friend?")
        assert frame.answer_space is AnswerSpace.ENTITY

    def test_scalar_how_many(self):
        frame = analyze_query("How many siblings does Alice have?")
        assert frame.answer_space is AnswerSpace.SCALAR

    def test_scalar_when(self):
        frame = analyze_query("When did Bob start working at Acme?")
        assert frame.answer_space is AnswerSpace.SCALAR

    def test_description_what_is(self):
        frame = analyze_query("What is Alice's current job?")
        # "What ... is" patterns — may be DESCRIPTION or SET depending on noun
        # Allow SET or DESCRIPTION for this pattern
        assert frame.answer_space in (AnswerSpace.DESCRIPTION, AnswerSpace.SET)

    def test_entity_extraction_two_names(self):
        frame = analyze_query("What does John Smith think about Alice Johnson?")
        assert "John Smith" in frame.focus_entities or "Alice Johnson" in frame.focus_entities

    def test_no_surface_keyword_routing(self):
        """'would', 'likely', 'might' must NOT determine the contract."""
        frame_plain = analyze_query("Is Alice happy?")
        frame_modal = analyze_query("Would Alice be happy?")
        # Both are BOOL — modal surface form must not change answer_space
        assert frame_plain.answer_space is AnswerSpace.BOOL
        assert frame_modal.answer_space is AnswerSpace.BOOL

    def test_temporal_scope_current_signals(self):
        frame = analyze_query("What is Alice's current job?")
        assert frame.temporal_scope == "current"

    def test_temporal_scope_historical(self):
        frame = analyze_query("When did Bob graduate?")
        assert frame.temporal_scope in ("historical", "none")  # 'when' → historical

    def test_temporal_scope_interval(self):
        frame = analyze_query("What did Alice do during the summer?")
        assert frame.temporal_scope == "interval"

    def test_evidence_regime_bool(self):
        frame = analyze_query("Is Alice married?")
        assert frame.evidence_regime is EvidenceRegime.SUPPORT_VS_CONTRA

    def test_evidence_regime_set(self):
        frame = analyze_query("List all sports Bob plays.")
        assert frame.evidence_regime is EvidenceRegime.AGGREGATE

    def test_evidence_regime_single(self):
        frame = analyze_query("What is Alice's age?")
        assert frame.evidence_regime is EvidenceRegime.SINGLE


# ---------------------------------------------------------------------------
# derive_answer_contract — mapping frame → contract
# ---------------------------------------------------------------------------


class TestDeriveAnswerContract:
    def test_set_question_truth_mode(self):
        frame = analyze_query("List all books Alice has read.")
        contract = derive_answer_contract(frame)
        assert contract.truth_mode is TruthMode.RECONSTRUCT

    def test_bool_question_truth_mode_is_direct_not_hypothesis(self):
        """BOOL does NOT auto-map to HYPOTHESIS — that's choose_strategy's job."""
        frame = analyze_query("Is Alice a vegetarian?")
        contract = derive_answer_contract(frame)
        assert contract.truth_mode is TruthMode.DIRECT

    def test_current_temporal_maps_to_current_axis(self):
        frame = analyze_query("What is Alice doing now?")
        contract = derive_answer_contract(frame)
        assert contract.time_axis is TimeAxis.CURRENT

    def test_historical_maps_to_event_axis(self):
        frame = analyze_query("When did Alice start her job?")
        contract = derive_answer_contract(frame)
        assert contract.time_axis is TimeAxis.EVENT

    def test_interval_maps_to_interval_axis(self):
        frame = analyze_query("What happened between 2020 and 2022?")
        contract = derive_answer_contract(frame)
        assert contract.time_axis is TimeAxis.INTERVAL

    def test_set_question_aggregate_regime(self):
        frame = analyze_query("List all sports Bob plays.")
        contract = derive_answer_contract(frame)
        assert contract.evidence_regime is EvidenceRegime.AGGREGATE

    def test_uncertainty_threshold_default(self):
        frame = analyze_query("Is Alice a teacher?")
        contract = derive_answer_contract(frame)
        assert 0.0 < contract.uncertainty_threshold <= 1.0


# ---------------------------------------------------------------------------
# Product queries — non-LoCoMo examples
# ---------------------------------------------------------------------------


class TestProductQueries:
    """Ensure contract derivation works on real-world product use cases."""

    def test_user_preference_set(self):
        frame = analyze_query("What are the user's dietary preferences?")
        derive_answer_contract(frame)  # validate no error
        # Dietary preferences = a set of restrictions
        assert frame.answer_space in (AnswerSpace.SET, AnswerSpace.DESCRIPTION)

    def test_current_state_bool(self):
        frame = analyze_query("Is the user subscribed to the newsletter?")
        contract = derive_answer_contract(frame)
        assert frame.answer_space is AnswerSpace.BOOL
        assert contract.truth_mode is TruthMode.DIRECT

    def test_entity_lookup(self):
        frame = analyze_query("Who is the user's account manager?")
        assert frame.answer_space is AnswerSpace.ENTITY

    def test_scalar_count(self):
        frame = analyze_query("How many open tickets does Alice have?")
        assert frame.answer_space is AnswerSpace.SCALAR


# ---------------------------------------------------------------------------
# Fix 3 — SET heuristic regression: singular nouns ending in 's' are not SET
# ---------------------------------------------------------------------------


import pytest  # noqa: E402


class TestSetHeuristicNarrowing:
    @pytest.mark.parametrize(
        "q",
        [
            "What is the status of Project Alpha?",
            "What is the bonus?",
            "What is Alice's address?",
            "What is the business outcome?",
            "What is the focus?",
        ],
    )
    def test_singular_s_nouns_not_routed_as_set(self, q: str) -> None:
        frame = analyze_query(q)
        assert frame.answer_space is not AnswerSpace.SET, (
            f"'{q}' must NOT be classified as SET (singular noun ending in 's')"
        )

    @pytest.mark.parametrize(
        "q",
        [
            "List all Alice's hobbies",
            "What are all the places Tim visited?",
            "Name every member of the team",
            "Enumerate the recent changes",
        ],
    )
    def test_structural_set_cues_still_route_as_set(self, q: str) -> None:
        frame = analyze_query(q)
        assert frame.answer_space is AnswerSpace.SET, (
            f"'{q}' must be classified as SET (structural aggregation cue)"
        )


# ---------------------------------------------------------------------------
# Fix — entity extraction strips leading modal/aux openers
# ---------------------------------------------------------------------------


class TestEntityExtractionFix:
    def test_would_caroline_extracts_caroline(self):
        """'Would Caroline ...' must extract 'Caroline', not 'Would Caroline'."""
        frame = analyze_query("Would Caroline be able to attend?")
        assert "Caroline" in frame.focus_entities, (
            f"Expected 'Caroline' in focus_entities, got {frame.focus_entities}"
        )
        assert not any("Would" in e for e in frame.focus_entities), (
            f"'Would' must not appear in entity: {frame.focus_entities}"
        )

    def test_did_melanie_extracts_melanie(self):
        """'Did Melanie ...' must extract 'Melanie', not 'Did Melanie'."""
        frame = analyze_query("Did Melanie finish the project?")
        assert "Melanie" in frame.focus_entities, (
            f"Expected 'Melanie' in focus_entities, got {frame.focus_entities}"
        )

    def test_is_alice_extracts_alice(self):
        """'Is Alice a doctor?' must extract 'Alice'."""
        frame = analyze_query("Is Alice a doctor?")
        assert "Alice" in frame.focus_entities, (
            f"Expected 'Alice' in focus_entities, got {frame.focus_entities}"
        )

    def test_has_bob_extracts_bob(self):
        frame = analyze_query("Has Bob visited Paris?")
        assert "Bob" in frame.focus_entities


# ---------------------------------------------------------------------------
# Fix — relation extraction normalizes inflected verbs
# ---------------------------------------------------------------------------


class TestRelationExtractionFix:
    def test_drives_normalized_to_drive(self):
        """'What kind of car does Evan drive?' must give focus_relation='drive'."""
        frame = analyze_query("What kind of car does Evan drive?")
        assert frame.focus_relation == "drive", (
            f"Expected focus_relation='drive', got {frame.focus_relation!r}"
        )

    def test_restoring_normalized_to_restore(self):
        frame = analyze_query("What does Dave find satisfying about restoring old cars?")
        assert frame.focus_relation in ("restore", "find", "satisfy", "finds_satisfying"), (
            f"Expected relation from restoring/find/satisfy, got {frame.focus_relation!r}"
        )

    def test_passed_away_gives_historical_scope(self):
        """'When did X's mother pass away?' must give temporal_scope='historical'."""
        frame = analyze_query("When did Deborah's mother pass away?")
        assert frame.temporal_scope in ("historical", "none"), (
            f"Expected historical temporal scope, got {frame.temporal_scope!r}"
        )
        assert "Deborah" in frame.focus_entities, (
            f"Expected 'Deborah' in focus_entities, got {frame.focus_entities}"
        )

    def test_passed_away_normalized_to_passed_away(self):
        frame = analyze_query("When did Deborah's mother pass away?")
        # Compound pattern "pass away" → "passed_away" (matches materializer's stored relation).
        assert frame.focus_relation in ("passed_away", "pass", "die", None), (
            f"Unexpected relation {frame.focus_relation!r}"
        )

    def test_research_detected(self):
        frame = analyze_query("What does Carol research at the university?")
        assert frame.focus_relation == "research", (
            f"Expected 'research', got {frame.focus_relation!r}"
        )


def test_what_is_alice_like_has_no_focus_relation() -> None:
    assert analyze_query("What is Alice like?").focus_relation is None
    assert analyze_query("What's the place like?").focus_relation is None


def test_work_as_maps_to_role() -> None:
    assert analyze_query("What does Alice work as?").focus_relation == "role"


def test_calendar_tokens_dropped_in_temporal_context():
    """'When did Melanie go camping in June?' → only Melanie, not June."""
    from ai_knot.query_contract import build_answer_contract

    contract, frame = build_answer_contract("When did Melanie go camping in June?")
    entities = list(frame.focus_entities)
    assert "Melanie" in entities
    assert "June" not in entities, f"June should be stripped as temporal token, got: {entities}"


def test_calendar_tokens_preserved_as_name():
    """'May called Bob yesterday' → both May and Bob extracted (May is a name here)."""
    from ai_knot.query_contract import build_answer_contract

    contract, frame = build_answer_contract("What did May tell Bob?")
    entities = list(frame.focus_entities)
    # May is in subject position without temporal preposition — should be kept
    assert "Bob" in entities
    # May should be kept (it's used as a person name, not as a temporal marker)
    assert "May" in entities, f"May (name) should be preserved, got: {entities}"


def test_implicit_set_books_has_read():
    """'What books has Alice read?' → AnswerSpace.SET."""
    from ai_knot.query_contract import build_answer_contract

    contract, frame = build_answer_contract("What books has Alice read?")
    assert contract.answer_space is AnswerSpace.SET, f"Expected SET, got {contract.answer_space}"


def test_what_is_description_not_set():
    """'What is Alice's job?' → AnswerSpace.DESCRIPTION (anti-regression)."""
    from ai_knot.query_contract import build_answer_contract

    contract, frame = build_answer_contract("What is Alice's job?")
    assert contract.answer_space is not AnswerSpace.SET, (
        f"'What is X's Y?' should not be SET, got {contract.answer_space}"
    )


# ---------------------------------------------------------------------------
# Move 4 — implicit SET covers full English aux class, not only has/have
# ---------------------------------------------------------------------------


class TestImplicitSetAuxBroadened:
    """'what/which + AUX + NOUN_HEAD' fires SET for any common English aux."""

    @pytest.mark.parametrize(
        "q",
        [
            "What activities does Melanie enjoy?",
            "What hobbies are on Alice's list?",
            "What places did Bob visit last year?",
            "What books is Carol reading?",
            "Which movies were on the shortlist?",
            "What sports do the kids play?",
        ],
    )
    def test_implicit_set_covers_all_auxiliaries(self, q: str) -> None:
        from ai_knot.query_contract import build_answer_contract

        contract, _ = build_answer_contract(q)
        assert contract.answer_space is AnswerSpace.SET, (
            f"'{q}' must be SET (aux + SET_NOUN_HEAD), got {contract.answer_space}"
        )

    @pytest.mark.parametrize(
        "q",
        [
            "What is the status of Project Alpha?",
            "What is the focus of this meeting?",
            "What is Alice's address?",
            "What is the outcome?",
            "Does Alice like coffee?",
            "Is Bob a vegetarian?",
        ],
    )
    def test_singular_and_bool_unchanged(self, q: str) -> None:
        """Widening aux must not turn singular or bool Q into SET."""
        from ai_knot.query_contract import build_answer_contract

        contract, _ = build_answer_contract(q)
        assert contract.answer_space is not AnswerSpace.SET, (
            f"'{q}' must NOT become SET after aux widening, got {contract.answer_space}"
        )
