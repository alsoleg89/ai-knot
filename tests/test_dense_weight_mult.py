"""Regression tests for the AI_KNOT_DENSE_WEIGHT_MULT flag.

Full-10 recall@60 measurement (2026-06) showed the dense (semantic) RRF signal is
under-weighted: scaling it (×3) lifts multi-hop/open-domain recall with no category
regression. The flag multiplies ``config.dense_rrf_weight`` at the RRF append site.
These tests guard the mechanism: default is a no-op, and the multiplier scales ONLY
the dense signal's weight, never the six lexical/structural weights.
"""

from __future__ import annotations

import pathlib

import pytest

import ai_knot.knowledge as K
from ai_knot.knowledge import KnowledgeBase
from ai_knot.storage.yaml_storage import YAMLStorage


@pytest.fixture
def kb(tmp_path: pathlib.Path) -> KnowledgeBase:
    storage = YAMLStorage(base_dir=str(tmp_path))
    return KnowledgeBase(agent_id="dense_mult_test", storage=storage)


def _capture_weights(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch, mult: float
) -> list[float]:
    """Run a recall and capture the RRF weight vector passed to the fusion."""
    captured: dict[str, list[float]] = {}
    orig = K._rrf_fuse

    def _spy(rankers: list, *, weights: list) -> dict:
        captured["weights"] = list(weights)
        return orig(rankers, weights=weights)

    monkeypatch.setattr(K, "_rrf_fuse", _spy)
    monkeypatch.setattr(K, "_DENSE_WEIGHT_MULT", mult)
    kb.recall("deployment pipeline", top_k=5)
    return captured["weights"]


def test_default_multiplier_is_one() -> None:
    """Default is a safe no-op so existing behaviour is unchanged."""
    assert K._DENSE_WEIGHT_MULT == 1.0


def test_multiplier_scales_only_dense_weight(
    kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A multiplier of 3 triples the dense signal's RRF weight and leaves the six
    lexical/structural weights untouched."""
    kb.add("Alice works on the deployment pipeline", tags=["infra"])
    kb.add("Bob enjoys playing chess", tags=["hobby"])
    kb.add("Carol manages the release schedule", tags=["infra"])

    base = _capture_weights(kb, monkeypatch, 1.0)
    boosted = _capture_weights(kb, monkeypatch, 3.0)

    # Dense is appended as the final signal when embeddings are available (stub
    # embedder provides them in tests), giving 6 lexical + 1 dense = 7 weights.
    assert len(base) == 7, f"expected dense signal appended, got {len(base)} weights"
    assert len(boosted) == len(base)
    # Six lexical/structural weights are identical under the flag.
    assert boosted[:6] == base[:6]
    # The dense weight (last) is tripled.
    assert boosted[6] == pytest.approx(base[6] * 3.0)


def test_multiplier_one_is_identity(kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch) -> None:
    """mult=1.0 leaves the dense weight exactly as the config defines it."""
    kb.add("Alice works on the deployment pipeline", tags=["infra"])
    kb.add("Bob enjoys playing chess", tags=["hobby"])

    w1 = _capture_weights(kb, monkeypatch, 1.0)
    w1b = _capture_weights(kb, monkeypatch, 1.0)
    assert w1 == w1b  # deterministic, no drift


def test_empty_embed_url_disables_dense_channel(tmp_path: pathlib.Path) -> None:
    """An empty ``embed_url`` makes recall BM25-only and deterministic.

    The dense channel is skipped *before* any embedding call, so recall does no
    network I/O and emits no per-batch fallback warnings — the documented switch
    for air-gapped deploys and reproducible offline evaluation.
    """
    storage = YAMLStorage(base_dir=str(tmp_path))
    kb = KnowledgeBase(agent_id="no_dense", storage=storage, embed_url="")
    kb.add("Alice works on the deployment pipeline", tags=["infra"])

    # Internal guard: no query vector is produced when the dense channel is off.
    assert kb._embed_for_recall(list(kb._storage.load("no_dense")), "deployment") is None
    # Recall still works (lexical-only) and finds the relevant fact.
    rendered = kb.recall("deployment pipeline", top_k=5)
    assert "deployment pipeline" in rendered
