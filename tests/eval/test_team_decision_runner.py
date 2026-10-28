"""Untrusted reader output must be validated before simulated effects."""

import pytest

from tests.eval.team_decisions.runner import PROTOCOL_HASH, _proposal, decision_schema


def test_structured_response_schema_only_enumerates_visible_citation_indices():
    assert decision_schema(3)["properties"]["basis"]["items"]["enum"] == [0, 1, 2]
    assert decision_schema(0)["properties"]["basis"]["maxItems"] == 0
    assert "enum" not in decision_schema(3)["properties"]["target"]
    assert decision_schema(3)["additionalProperties"] is False


def test_reader_protocol_was_frozen_before_live_execution():
    assert PROTOCOL_HASH == "f3ddea2a84a999d1e850f3f7de5c9918ea0f920430f48b169e736cff80f15d80"


def test_hold_without_a_known_target_is_a_valid_no_effect_proposal():
    value, refs = _proposal('{"action":"hold","target":"","basis":[]}', [])
    assert value["action"] == "hold" and refs == ()


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"action":"execute","target":"x","basis":[true]}',
        '{"action":"execute","target":"x","basis":[0]}',
        '{"action":"execute","target":"","basis":[]}',
        '{"action":"success","target":"x","basis":[]}',
        '{"action":"hold","target":"","basis":[],"oracle":true}',
    ],
)
def test_invalid_or_out_of_context_proposal_is_rejected(response):
    with pytest.raises((ValueError, TypeError)):
        _proposal(response, [])
