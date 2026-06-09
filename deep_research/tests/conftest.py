from __future__ import annotations

from pathlib import Path

import pytest

from deep_research.corpus import Corpus
from deep_research.llm import MockLLMClient
from deep_research.memory import SemanticMemory
from deep_research.semantic import MockEmbedder, MockReranker


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub out all network calls in tests — keeps suite offline and fast."""
    monkeypatch.setattr("deep_research.roles.scout.search_arxiv", lambda *a, **kw: [])
    monkeypatch.setattr("deep_research.roles.scout.fetch_arxiv_text", lambda *a, **kw: "")


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient()


@pytest.fixture
def campaign_corpus(tmp_path: Path) -> Corpus:
    corpus = Corpus(tmp_path / "campaign")
    corpus.initialize("test-campaign-id", "testhash00000000")
    return corpus


@pytest.fixture
def semantic_memory(campaign_corpus: Corpus) -> SemanticMemory:
    """Offline SemanticMemory backed by MockEmbedder + MockReranker."""
    return SemanticMemory(campaign_corpus, MockEmbedder(), MockReranker())
