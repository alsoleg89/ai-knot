from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class EmbeddingBackend(ABC):
    @property
    @abstractmethod
    def dim(self) -> int: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def embed(self, texts: list[str]) -> NDArray[np.float32]: ...


class MockEmbedder(EmbeddingBackend):
    """Deterministic embedder for offline tests — no network, no torch."""

    _DIM = 32

    @property
    def dim(self) -> int:
        return self._DIM

    @property
    def name(self) -> str:
        return "mock"

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        vecs = []
        for text in texts:
            seed = int(hashlib.md5(text.encode(), usedforsecurity=False).hexdigest(), 16) % (2**31)
            rng = np.random.default_rng(seed)
            v = rng.standard_normal(self._DIM).astype(np.float32)
            norm = float(np.linalg.norm(v))
            v /= norm + 1e-10
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)


class SentenceTransformerEmbedder(EmbeddingBackend):
    """BAAI/bge-m3 — multilingual (100+ languages), SOTA retrieval, no query prefix needed."""

    def __init__(self, model: str = "BAAI/bge-m3") -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        self._model_name = model
        self._st_model = SentenceTransformer(model, device=device)
        self._dim = int(self._st_model.get_embedding_dimension() or 1024)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._model_name

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        vecs = self._st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, candidates: list[str]) -> list[float]: ...


class MockReranker(Reranker):
    """Deterministic reranker for offline tests — scores by token overlap."""

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        q_tokens = set(query.lower().split())
        scores = []
        for c in candidates:
            c_tokens = set(c.lower().split())
            scores.append(float(len(q_tokens & c_tokens)))
        return scores


class CrossEncoderReranker(Reranker):
    """BAAI/bge-reranker-v2-m3 — multilingual cross-encoder, pairs with bge-m3."""

    def __init__(self, model: str = "BAAI/bge-reranker-v2-m3") -> None:
        from sentence_transformers import CrossEncoder

        self._ce_model = CrossEncoder(model)

    def rerank(self, query: str, candidates: list[str]) -> list[float]:
        pairs = [[query, c] for c in candidates]
        return list(self._ce_model.predict(pairs))
