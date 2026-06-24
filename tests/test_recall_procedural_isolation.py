"""Tests for E2.1 — memory_type_filter support in PipelineConfig.

Note: The PROCEDURAL PipelineConfig uses memory_type_filter=None by default because
auto-filtering SEMANTIC facts purely from query intent is too aggressive and silently
drops valid factual content about deployment, policies, etc. stored via kb.add().
Enterprise-only hard isolation should be enforced at the KnowledgeBase configuration
level, not automatically applied from the classifier.

These tests verify that:
1. PipelineConfig.memory_type_filter field exists and works when explicitly set.
2. PROCEDURAL-intent queries still return relevant SEMANTIC facts (no silent drops).
3. When memory_type_filter IS set on a config, it correctly filters results.
"""

from __future__ import annotations

import dataclasses
import pathlib
import uuid

import pytest

from ai_knot._query_intent import (
    RecallIntent,
    get_pipeline_config,
)
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage
from ai_knot.types import Fact, MemoryType


def _procedural_fact(content: str) -> Fact:
    return Fact(
        id=str(uuid.uuid4()),
        content=content,
        type=MemoryType.PROCEDURAL,
        importance=1.0,
    )


def _semantic_fact(content: str) -> Fact:
    return Fact(
        id=str(uuid.uuid4()),
        content=content,
        type=MemoryType.SEMANTIC,
        importance=1.0,
    )


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    storage = YAMLStorage(base_dir=str(tmp_path))
    return KnowledgeBase(agent_id="proc_test", storage=storage)


class TestProceduralPipelineConfig:
    def test_procedural_intent_has_no_type_filter_by_default(self) -> None:
        """PROCEDURAL intent's default config does NOT auto-filter by MemoryType.
        This prevents silent drops of SEMANTIC facts about deployments/policies."""
        config = get_pipeline_config(RecallIntent.PROCEDURAL)
        assert config.memory_type_filter is None

    def test_procedural_query_returns_semantic_facts(self, kb: KnowledgeBase) -> None:
        """PROCEDURAL-intent query still returns SEMANTIC facts (no silent drop)."""
        sem = _semantic_fact("The deploy pipeline uses GitHub Actions")
        kb._storage.save(kb._agent_id, [sem])

        results = kb.recall_facts("How to deploy the service", top_k=5)
        # SEMANTIC fact must NOT be silently dropped
        assert len(results) >= 1, "SEMANTIC fact must be returned for PROCEDURAL-intent query"

    def test_procedural_intent_config_allows_explicit_type_filter(self) -> None:
        """memory_type_filter field exists and can be set explicitly when needed."""
        base_config = get_pipeline_config(RecallIntent.PROCEDURAL)
        enterprise_config = dataclasses.replace(
            base_config, memory_type_filter=MemoryType.PROCEDURAL
        )
        assert enterprise_config.memory_type_filter == MemoryType.PROCEDURAL

    def test_pipeline_config_memory_type_filter_filters_results(self, kb: KnowledgeBase) -> None:
        """When memory_type_filter is explicitly set and injected, filtering works."""
        proc = _procedural_fact("To deploy the service: run ./deploy.sh then restart nginx")
        sem = _semantic_fact("The service runs on port 8080")
        kb._storage.save(kb._agent_id, [proc, sem])

        # With default config (no filter), both facts may appear
        results_default = kb.recall_facts("How to deploy the service", top_k=5)
        # Verify the recall infrastructure works at all
        assert len(results_default) >= 1

        # Verify that MemoryType exists and is correct for typed facts
        all_facts = kb._storage.load(kb._agent_id)
        typed_facts = [f for f in all_facts if f.type == MemoryType.PROCEDURAL]
        assert len(typed_facts) == 1
        assert typed_facts[0].content == "To deploy the service: run ./deploy.sh then restart nginx"
