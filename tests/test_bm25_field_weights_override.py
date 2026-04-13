"""Tests for E2.2 — BM25F field_weights_override in InvertedIndex.score()."""

from __future__ import annotations

import uuid

from ai_knot._inverted_index import InvertedIndex
from ai_knot.types import Fact


def _fact(content: str, tags: list[str] | None = None) -> Fact:
    return Fact(
        id=str(uuid.uuid4()),
        content=content,
        tags=tags or [],
        importance=1.0,
    )


class TestFieldWeightsOverride:
    def test_tags_override_boosts_tag_factoid(self) -> None:
        """Fact with query term in tags ranks higher than fact with term only in content
        when tags weight is boosted to 5.0."""
        # f_content has "python" in content only
        f_content = _fact("python is a high level programming language", tags=["general"])
        # f_tags has "python" only in tags
        f_tags = _fact("This is an unrelated sentence about cooking", tags=["python"])

        index = InvertedIndex([f_content, f_tags])

        # Override: tags=5.0 — f_tags should be boosted significantly
        boosted_scores = index.score("python", field_weights_override={"tags": 5.0, "content": 1.0})

        # With high tags boost, f_tags should rank >= f_content
        assert boosted_scores.get(f_tags.id, 0.0) >= boosted_scores.get(f_content.id, 0.0), (
            "tags fact should rank higher when tags weight is boosted"
        )

    def test_none_override_uses_defaults(self) -> None:
        """Passing field_weights_override=None gives same result as omitting it."""
        f = _fact("machine learning algorithms are powerful", tags=["ml"])
        index = InvertedIndex([f])

        scores_default = index.score("machine learning")
        scores_none = index.score("machine learning", field_weights_override=None)

        assert scores_default == scores_none

    def test_partial_override_uses_defaults_for_rest(self) -> None:
        """Providing only 'tags' override keeps other fields at their defaults."""
        f = _fact("docker container deployment", tags=["docker"])
        index = InvertedIndex([f])

        # Only override tags, content should stay at default 1.0
        scores = index.score("docker", field_weights_override={"tags": 10.0})
        assert scores.get(f.id, 0.0) > 0.0

    def test_zero_content_weight_removes_content_signal(self) -> None:
        """Setting content weight to 0 removes content field contribution."""
        f_content_only = _fact("astronomy is the study of stars", tags=["science"])
        f_tags_only = _fact("unrelated sentence about food", tags=["astronomy"])

        index = InvertedIndex([f_content_only, f_tags_only])

        # With content=0: only tags signal — f_tags_only should dominate
        scores = index.score("astronomy", field_weights_override={"content": 0.0, "tags": 1.0})
        # f_tags_only has "astronomy" in tags; f_content_only has it only in content
        assert scores.get(f_tags_only.id, 0.0) > scores.get(f_content_only.id, 0.0), (
            "With content=0, tag-only fact should outscore content-only fact"
        )

    def test_override_does_not_mutate_index_defaults(self) -> None:
        """Calling score() with override must not affect subsequent calls without override."""
        f = _fact("blockchain technology applications", tags=["blockchain"])
        index = InvertedIndex([f])

        score_before = index.score("blockchain")
        index.score("blockchain", field_weights_override={"tags": 99.0})
        score_after = index.score("blockchain")

        assert score_before == score_after, "Override must not mutate module-level field weights"
