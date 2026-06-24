"""Tests for E1.4 — _mmr_select receives lambda from PipelineConfig via _execute_recall."""

from __future__ import annotations

import pathlib
from unittest.mock import patch

import pytest

from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    storage = YAMLStorage(base_dir=str(tmp_path))
    kb = KnowledgeBase(agent_id="mmr_dispatch_test", storage=storage)
    # Add enough facts for MMR to be triggered
    for i in range(8):
        kb.add(f"Alice hobby number {i}: activity item", tags=["alice"])
    return kb


class TestMMRLambdaDispatch:
    def test_aggregational_query_uses_low_lambda(self, kb: KnowledgeBase) -> None:
        """'List all options' → AGGREGATIONAL → mmr_lambda=0.3."""
        calls: list[float] = []
        original = kb._mmr_select

        def capturing(*args: object, **kwargs: object) -> object:
            calls.append(kwargs.get("lambda_", -1.0))  # type: ignore[arg-type]
            return original(*args, **kwargs)  # type: ignore[arg-type]

        with patch.object(kb, "_mmr_select", side_effect=capturing):
            kb.recall_facts("List all Alice hobbies and options", top_k=4)

        assert len(calls) >= 1, "_mmr_select was not called"
        assert calls[0] <= 0.4, f"AGGREGATIONAL expected lambda_ ≤ 0.4, got {calls[0]}"

    def test_factual_query_uses_high_lambda(self, kb: KnowledgeBase) -> None:
        """'What is Alice hobby 3?' → FACTUAL → mmr_lambda=0.85."""
        calls: list[float] = []
        original = kb._mmr_select

        def capturing(*args: object, **kwargs: object) -> object:
            calls.append(kwargs.get("lambda_", -1.0))  # type: ignore[arg-type]
            return original(*args, **kwargs)  # type: ignore[arg-type]

        with patch.object(kb, "_mmr_select", side_effect=capturing):
            kb.recall_facts("What is Alice hobby 3", top_k=4)

        assert len(calls) >= 1, "_mmr_select was not called"
        assert calls[0] >= 0.75, f"FACTUAL expected lambda_ ≥ 0.75, got {calls[0]}"

    def test_exploratory_query_uses_mid_lambda(self, kb: KnowledgeBase) -> None:
        """'Why did Alice choose these activities?' → EXPLORATORY → mmr_lambda=0.65."""
        calls: list[float] = []
        original = kb._mmr_select

        def capturing(*args: object, **kwargs: object) -> object:
            calls.append(kwargs.get("lambda_", -1.0))  # type: ignore[arg-type]
            return original(*args, **kwargs)  # type: ignore[arg-type]

        with patch.object(kb, "_mmr_select", side_effect=capturing):
            kb.recall_facts("Why did Alice choose these hobby activities", top_k=4)

        assert len(calls) >= 1, "_mmr_select was not called"
        lambda_used = calls[0]
        assert 0.5 <= lambda_used <= 0.75, (
            f"EXPLORATORY expected lambda_ in [0.5, 0.75], got {lambda_used}"
        )
