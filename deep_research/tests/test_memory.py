from __future__ import annotations

from pathlib import Path

from deep_research.corpus import Corpus
from deep_research.memory import SemanticMemory
from deep_research.semantic import MockEmbedder, MockReranker


def _make(tmp_path: Path, name: str = "c") -> tuple[Corpus, SemanticMemory]:
    corpus = Corpus(tmp_path / name)
    corpus.initialize(f"id-{name}", "testhash")
    sem = SemanticMemory(corpus, MockEmbedder(), MockReranker())
    return corpus, sem


def test_recall_empty(tmp_path: Path) -> None:
    _, sem = _make(tmp_path)
    assert sem.recall("query") == []


def test_sync_and_recall_top_result(tmp_path: Path) -> None:
    corpus, sem = _make(tmp_path)
    target_text = "alpha beta gamma delta unique phrase"
    corpus.append_source({"tick": 0, "content": target_text})
    corpus.append_source({"tick": 0, "content": "zzzz qqqq xxxx yyyy garbled"})
    sem.sync()

    # Same text as the query → same hash-derived vector → cosine 1.0
    results = sem.recall(target_text, k=1, stream="sources")
    assert len(results) == 1
    assert results[0]["entry"]["content"] == target_text


def test_sync_idempotent(tmp_path: Path) -> None:
    corpus, sem = _make(tmp_path)
    corpus.append_source({"tick": 0, "content": "some content"})
    sem.sync()
    count = len(sem._index)
    sem.sync()
    assert len(sem._index) == count


def test_stream_filter(tmp_path: Path) -> None:
    corpus, sem = _make(tmp_path)
    corpus.append_source({"tick": 0, "content": "source entry"})
    corpus.append_proof({"tick": 0, "content": "proof entry"})
    sem.sync()
    results = sem.recall("entry", k=5, stream="sources")
    assert all(r["stream"] == "sources" for r in results)
    assert len(results) == 1


def test_sync_catches_new_entries(tmp_path: Path) -> None:
    corpus, sem = _make(tmp_path)
    sem.sync()
    assert len(sem._index) == 0

    corpus.append_source({"tick": 1, "content": "new source added later"})
    sem.sync()
    assert len(sem._index) == 1


def test_multiple_streams_indexed(tmp_path: Path) -> None:
    corpus, sem = _make(tmp_path)
    corpus.append_source({"tick": 0, "content": "source"})
    corpus.append_proof({"tick": 0, "content": "proof"})
    corpus.append_critique({"tick": 0, "content": "critique"})
    sem.sync()
    assert len(sem._index) == 3
    streams = {r["stream"] for r in sem._index}
    assert streams == {"sources", "proofs", "critique"}


def test_rebuild_on_embedder_name_change(tmp_path: Path) -> None:
    corpus, sem = _make(tmp_path)
    corpus.append_source({"tick": 0, "content": "some content"})
    sem.sync()
    assert len(sem._index) == 1

    class AltEmbedder(MockEmbedder):
        @property
        def name(self) -> str:
            return "alt-mock"

    sem2 = SemanticMemory(corpus, AltEmbedder())
    # Rebuilds index (empty until sync is called) — old vectors cleared
    assert len(sem2._index) == 0
    sem2.sync()
    assert len(sem2._index) == 1


def test_crash_resume_f32_truncation(tmp_path: Path) -> None:
    corpus, sem = _make(tmp_path)
    corpus.append_source({"tick": 0, "content": "entry one"})
    sem.sync()
    assert len(sem._index) == 1

    # Simulate crash: extra f32 row without a matching index line.
    vec_path = corpus.root / "embeddings" / "vectors.f32"
    extra = sem._matrix[0].tobytes()
    with vec_path.open("ab") as fh:
        fh.write(extra)

    sem2 = SemanticMemory(corpus, MockEmbedder())
    assert len(sem2._index) == 1
    assert sem2._matrix.shape == (1, MockEmbedder._DIM)


def test_cosine_correct_on_known_vectors(tmp_path: Path) -> None:
    corpus, sem = _make(tmp_path)
    # Both entries; recall should return exactly k results when k <= total.
    corpus.append_source({"tick": 0, "content": "one"})
    corpus.append_source({"tick": 0, "content": "two"})
    sem.sync()
    results = sem.recall("one", k=2, stream="sources")
    assert len(results) == 2
    # First result should be the "one" entry (identical text → cosine 1.0)
    assert results[0]["entry"]["content"] == "one"


def test_reranker_reorders_results(tmp_path: Path) -> None:
    corpus, sem = _make(tmp_path)
    # Both entries match "memory" — MockReranker orders by token overlap.
    corpus.append_source({"tick": 0, "content": "memory protocol consensus"})
    corpus.append_source({"tick": 0, "content": "memory"})
    sem.sync()
    results = sem.recall("memory protocol consensus", k=2, stream="sources")
    assert len(results) == 2
    # MockReranker: "memory protocol consensus" has 3 token overlaps with first entry
    # and 1 overlap with second — so first entry should rank higher.
    assert results[0]["entry"]["content"] == "memory protocol consensus"


def test_persist_and_reload(tmp_path: Path) -> None:
    corpus, sem = _make(tmp_path)
    corpus.append_source({"tick": 0, "content": "persisted content"})
    sem.sync()

    # Reload from disk
    sem2 = SemanticMemory(corpus, MockEmbedder())
    assert len(sem2._index) == 1
    assert sem2._matrix.shape == (1, MockEmbedder._DIM)


def test_recall_k_capped_to_available(tmp_path: Path) -> None:
    corpus, sem = _make(tmp_path)
    corpus.append_source({"tick": 0, "content": "only one entry"})
    sem.sync()
    results = sem.recall("query", k=10)
    assert len(results) == 1


def test_entry_loaded_from_corpus(tmp_path: Path) -> None:
    corpus, sem = _make(tmp_path)
    corpus.append_source({"tick": 5, "focus": "test-focus", "content": "test-content"})
    sem.sync()
    results = sem.recall("test-content", k=1)
    assert results[0]["entry"]["tick"] == 5
    assert results[0]["entry"]["focus"] == "test-focus"
