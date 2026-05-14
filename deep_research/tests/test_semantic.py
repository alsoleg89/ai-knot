from __future__ import annotations

import numpy as np

from deep_research.semantic import MockEmbedder, MockReranker


def test_mock_embedder_deterministic() -> None:
    emb = MockEmbedder()
    v1 = emb.embed(["hello world"])
    v2 = emb.embed(["hello world"])
    np.testing.assert_array_equal(v1, v2)


def test_mock_embedder_different_texts_differ() -> None:
    emb = MockEmbedder()
    v1 = emb.embed(["hello world"])
    v2 = emb.embed(["completely different zxqvj"])
    assert not np.allclose(v1, v2)


def test_mock_embedder_unit_norm() -> None:
    emb = MockEmbedder()
    vecs = emb.embed(["foo", "bar", "baz qux"])
    for v in vecs:
        assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_mock_embedder_batch_shape() -> None:
    emb = MockEmbedder()
    vecs = emb.embed(["a", "b", "c"])
    assert vecs.shape == (3, MockEmbedder._DIM)


def test_mock_embedder_single_text_shape() -> None:
    emb = MockEmbedder()
    vecs = emb.embed(["single"])
    assert vecs.shape == (1, MockEmbedder._DIM)


def test_mock_reranker_higher_overlap_scores_higher() -> None:
    rr = MockReranker()
    scores = rr.rerank(
        "multi-agent memory protocol",
        [
            "multi-agent memory consensus algorithm",
            "cooking recipe for pasta dinner",
        ],
    )
    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_mock_reranker_length_matches_candidates() -> None:
    rr = MockReranker()
    scores = rr.rerank("query", ["a", "b", "c", "d"])
    assert len(scores) == 4


def test_mock_reranker_empty_candidates() -> None:
    rr = MockReranker()
    scores = rr.rerank("query", [])
    assert scores == []
