"""Tests for E1.1 — RecallIntent classifier and PipelineConfig matrix."""

from __future__ import annotations

import pytest

from ai_knot._query_intent import (
    PipelineConfig,
    RecallIntent,
    classify_recall_intent,
    get_pipeline_config,
)


class TestRecallIntentClassifier:
    @pytest.mark.parametrize(
        "query,expected",
        [
            # BROAD_CONTEXT — ≤1 content token after stopword removal
            ("hi", RecallIntent.BROAD_CONTEXT),
            ("what is it", RecallIntent.BROAD_CONTEXT),  # all tokens are stopwords
            # PROCEDURAL
            ("how to deploy the service", RecallIntent.PROCEDURAL),
            ("what are the steps to onboard a user", RecallIntent.PROCEDURAL),
            ("company policy on expenses", RecallIntent.PROCEDURAL),
            ("guidelines for code review", RecallIntent.PROCEDURAL),
            # NAVIGATIONAL
            ("show me the log file from last week", RecallIntent.NAVIGATIONAL),
            ("find the meeting notes from Tuesday", RecallIntent.NAVIGATIONAL),
            ("open the report for Q1", RecallIntent.NAVIGATIONAL),
            # AGGREGATIONAL
            ("list all hobbies Melanie has", RecallIntent.AGGREGATIONAL),
            ("what are all the topics we discussed", RecallIntent.AGGREGATIONAL),
            ("summarize everything about the project", RecallIntent.AGGREGATIONAL),
            # EXPLORATORY
            ("why did Alice decide to move to Boston", RecallIntent.EXPLORATORY),
            ("how does the caching mechanism work", RecallIntent.EXPLORATORY),
            ("what happened before the conference last year", RecallIntent.EXPLORATORY),
            # FACTUAL — default
            ("what is Alice salary", RecallIntent.FACTUAL),
            ("where does Bob live", RecallIntent.FACTUAL),
            ("when is the next meeting", RecallIntent.FACTUAL),
        ],
    )
    def test_classifier_cases(self, query: str, expected: RecallIntent) -> None:
        assert classify_recall_intent(query) == expected, (
            f"classify_recall_intent({query!r}) = "
            f"{classify_recall_intent(query)!r}, expected {expected!r}"
        )


class TestPipelineConfigMatrix:
    def test_all_intents_have_config(self) -> None:
        for intent in RecallIntent:
            config = get_pipeline_config(intent)
            assert isinstance(config, PipelineConfig)

    def test_rrf_weights_length_six(self) -> None:
        for intent in RecallIntent:
            config = get_pipeline_config(intent)
            assert len(config.rrf_weights) == 6, (
                f"{intent}: rrf_weights length={len(config.rrf_weights)}, expected 6"
            )

    def test_mmr_lambda_in_range(self) -> None:
        for intent in RecallIntent:
            config = get_pipeline_config(intent)
            assert 0.0 <= config.mmr_lambda <= 1.0, (
                f"{intent}: mmr_lambda={config.mmr_lambda} out of [0,1]"
            )

    def test_sort_strategy_valid(self) -> None:
        valid = {"relevance", "sandwich", "chronological"}
        for intent in RecallIntent:
            config = get_pipeline_config(intent)
            assert config.sort_strategy in valid, (
                f"{intent}: sort_strategy={config.sort_strategy!r} not in {valid}"
            )

    def test_factual_point_query_signals(self) -> None:
        config = get_pipeline_config(RecallIntent.FACTUAL)
        # BM25 is moderate (≥4) for lexical precision on point queries.
        assert config.rrf_weights[0] >= 4, "FACTUAL should have meaningful BM25 weight"
        # Dense at BM25 parity — prevents multi-fact displacement while retaining signal.
        assert config.dense_rrf_weight >= 4, "FACTUAL should have meaningful dense weight"
        assert config.mmr_lambda >= 0.8, "FACTUAL should have high MMR lambda (precision)"
        assert config.sort_strategy == "relevance"
        assert config.skip_prf is True

    def test_aggregational_low_lambda_sandwich(self) -> None:
        config = get_pipeline_config(RecallIntent.AGGREGATIONAL)
        assert config.mmr_lambda <= 0.4, "AGGREGATIONAL should have low MMR lambda (diversity)"
        assert config.sort_strategy == "sandwich"
        assert config.skip_prf is False

    def test_exploratory_uses_ddsa(self) -> None:
        config = get_pipeline_config(RecallIntent.EXPLORATORY)
        assert config.use_ddsa is True
        assert config.sort_strategy == "chronological"

    def test_navigational_field_weights_override(self) -> None:
        config = get_pipeline_config(RecallIntent.NAVIGATIONAL)
        assert config.field_weights_override is not None
        assert config.field_weights_override.get("tags", 0) >= 3.0

    def test_procedural_no_auto_type_filter(self) -> None:
        """PROCEDURAL config does not auto-filter by type to avoid silent SEMANTIC drops."""
        config = get_pipeline_config(RecallIntent.PROCEDURAL)
        assert config.memory_type_filter is None

    def test_broad_context_no_filter(self) -> None:
        config = get_pipeline_config(RecallIntent.BROAD_CONTEXT)
        assert config.memory_type_filter is None

    def test_dense_rrf_weight_all_intents(self) -> None:
        """Every intent must have a dense_rrf_weight ≥ 0; FACTUAL must be highest."""
        for intent in RecallIntent:
            config = get_pipeline_config(intent)
            assert config.dense_rrf_weight >= 0.0, (
                f"{intent}: dense_rrf_weight={config.dense_rrf_weight} must be non-negative"
            )
        factual_w = get_pipeline_config(RecallIntent.FACTUAL).dense_rrf_weight
        broad_w = get_pipeline_config(RecallIntent.BROAD_CONTEXT).dense_rrf_weight
        assert factual_w > broad_w, (
            "FACTUAL dense weight must exceed BROAD_CONTEXT to keep importance dominant "
            "on short/vague queries"
        )

    def test_dense_rrf_weight_nonneg(self) -> None:
        """Regression: dense_rrf_weight must never go negative (Stage-3 RRF guard)."""
        for intent in RecallIntent:
            w = get_pipeline_config(intent).dense_rrf_weight
            assert w >= 0.0
