"""Integration tests for lexical bridge in recall pipeline."""

import pytest

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage


@pytest.fixture
def kb(tmp_path):  # type: ignore[no-untyped-def]
    return KnowledgeBase(agent_id="lex-test", storage=YAMLStorage(base_dir=str(tmp_path)))


class TestLexicalBridgeIntegration:
    def test_trace_present_when_flag_on(
        self, kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        monkeypatch.setenv("AI_KNOT_LEXICAL_BRIDGE", "1")
        kb.add("Sarah practices tennis every Saturday morning")
        _, trace = kb.recall_facts_with_trace("What sport does Sarah play?")
        # stage0_lexical_bridge must be present in trace when flag is on
        assert "stage0_lexical_bridge" in trace

    def test_trace_absent_when_flag_off(
        self, kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        monkeypatch.delenv("AI_KNOT_LEXICAL_BRIDGE", raising=False)
        kb.add("Sarah practices tennis every Saturday morning")
        _, trace = kb.recall_facts_with_trace("What sport does Sarah play?")
        assert trace.get("stage0_lexical_bridge") is None

    def test_navigational_trace_shows_zero_terms(
        self, kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        monkeypatch.setenv("AI_KNOT_LEXICAL_BRIDGE", "1")
        kb.add("Meeting notes from January board session")
        _, trace = kb.recall_facts_with_trace("find meeting notes from January")
        bridge = trace.get("stage0_lexical_bridge")
        if bridge is not None:
            assert bridge["terms_added"] == 0

    def test_bridge_on_activity_query_adds_terms(
        self, kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        monkeypatch.setenv("AI_KNOT_LEXICAL_BRIDGE", "1")
        kb.add("Sarah goes to tennis practice every week")
        _, trace = kb.recall_facts_with_trace("What sport does Sarah play?")
        bridge = trace.get("stage0_lexical_bridge")
        assert bridge is not None
        # "play" is in query → activity_sport frame fires → terms_added > 0
        assert bridge["terms_added"] > 0
        assert "activity_sport" in bridge["frames_applied"]

    def test_bridge_expansion_weights_all_below_one(
        self, kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        monkeypatch.setenv("AI_KNOT_LEXICAL_BRIDGE", "1")
        kb.add("Tom works as a software engineer at Acme Corp")
        _, trace = kb.recall_facts_with_trace("What does Tom do for work?")
        bridge = trace.get("stage0_lexical_bridge")
        assert bridge is not None
        for term, w in bridge["expansion_weights"].items():
            assert w < 1.0, f"Expansion weight >= 1.0 for term '{term}': {w}"

    def test_bridge_flag_default_off(
        self, kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch, tmp_path: object
    ) -> None:
        """By default AI_KNOT_LEXICAL_BRIDGE is not set — trace key must be absent."""
        monkeypatch.delenv("AI_KNOT_LEXICAL_BRIDGE", raising=False)
        kb.add("Alice loves cooking and baking")
        _, trace = kb.recall_facts_with_trace("What does Alice enjoy doing?")
        # When flag is off, key should be absent (None means not set)
        assert trace.get("stage0_lexical_bridge") is None
