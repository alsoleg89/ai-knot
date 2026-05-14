from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from deep_research.semantic import EmbeddingBackend, Reranker

if TYPE_CHECKING:
    from deep_research.corpus import Corpus

# Corpus streams to embed and their relative paths inside corpus.root.
_STREAMS: dict[str, str] = {
    "sources": "sources/sources.jsonl",
    "proofs": "proofs/proofs.jsonl",
    "critique": "critique/critique.jsonl",
    "experiments": "experiments/experiments.jsonl",
    "candidates": "theory_population/candidates.jsonl",
}

_EMBEDDINGS_DIR = "embeddings"


def _entry_text(entry: dict[str, Any]) -> str:
    parts: list[str] = []
    if focus := entry.get("focus"):
        parts.append(str(focus))
    if content := entry.get("content"):
        parts.append(str(content))
    return " ".join(parts)[:2000]


class SemanticMemory:
    """Append-only semantic index over the research corpus.

    Storage layout (corpus_root/embeddings/):
      vectors.f32  — raw float32 rows, one per indexed entry (append-only)
      index.jsonl  — one JSON line per row: {stream, line_no, tick, text_preview}
      meta.json    — {embedder_name, dim, count}

    Crash-safety: f32 row is written before the index line. On startup the f32
    file is truncated to the index line count, so sync() fills the gap.
    """

    def __init__(
        self,
        corpus: Corpus,
        embedder: EmbeddingBackend,
        reranker: Reranker | None = None,
    ) -> None:
        self._corpus = corpus
        self._embedder = embedder
        self._reranker = reranker
        self._emb_dir: Path = corpus.root / _EMBEDDINGS_DIR
        self._emb_dir.mkdir(exist_ok=True)
        self._vec_path = self._emb_dir / "vectors.f32"
        self._idx_path = self._emb_dir / "index.jsonl"
        self._meta_path = self._emb_dir / "meta.json"
        # Always initialize before _load_or_rebuild_empty sets them.
        self._matrix: NDArray[np.float32] = np.empty((0, self._embedder.dim), dtype=np.float32)
        self._index: list[dict[str, Any]] = []
        self._load_or_rebuild_empty()

    # ── Startup ───────────────────────────────────────────────────────────────

    def _load_or_rebuild_empty(self) -> None:
        if self._meta_path.exists():
            meta = json.loads(self._meta_path.read_text())
            if meta.get("embedder_name") != self._embedder.name:
                self._reset()
                return
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self._idx_path.exists():
            return  # keep empty defaults from __init__

        idx_lines = [ln for ln in self._idx_path.read_text().splitlines() if ln.strip()]
        n_idx = len(idx_lines)

        if not self._vec_path.exists():
            return

        raw = self._vec_path.read_bytes()
        row_bytes = self._embedder.dim * 4
        n_vec = len(raw) // row_bytes

        # Truncate f32 if it has more rows than index lines (crash mid-write).
        if n_vec > n_idx:
            self._vec_path.write_bytes(raw[: n_idx * row_bytes])
            raw = raw[: n_idx * row_bytes]
            n_vec = n_idx

        n = min(n_vec, n_idx)
        if n == 0:
            return

        flat = np.frombuffer(raw[: n * row_bytes], dtype=np.float32).copy()
        self._matrix = flat.reshape(n, self._embedder.dim)
        self._index = [json.loads(ln) for ln in idx_lines[:n]]

    def _reset(self) -> None:
        self._vec_path.unlink(missing_ok=True)
        self._idx_path.unlink(missing_ok=True)
        self._meta_path.unlink(missing_ok=True)
        self._matrix = np.empty((0, self._embedder.dim), dtype=np.float32)
        self._index = []

    # ── Sync ─────────────────────────────────────────────────────────────────

    def sync(self) -> None:
        """Embed any corpus entries not yet in the index. Idempotent."""
        for stream, rel_path in _STREAMS.items():
            path = self._corpus.root / rel_path
            if not path.exists():
                continue
            all_lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
            already = sum(1 for r in self._index if r["stream"] == stream)
            new_lines = all_lines[already:]
            if not new_lines:
                continue
            new_entries = [json.loads(ln) for ln in new_lines]
            texts = [_entry_text(e) for e in new_entries]
            vecs = self._embedder.embed(texts)
            for i, (entry, vec) in enumerate(zip(new_entries, vecs)):
                line_no = already + i
                self._append_vector(
                    vec.astype(np.float32),
                    {
                        "stream": stream,
                        "line_no": line_no,
                        "tick": int(entry.get("tick", -1)),
                        "text_preview": texts[i][:80],
                    },
                )
        self._write_meta()

    def rebuild(self) -> None:
        """Full re-embed from scratch (called when embedder changes)."""
        self._reset()
        self.sync()

    # ── Recall ────────────────────────────────────────────────────────────────

    def recall(
        self,
        query: str,
        k: int = 5,
        stream: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return k most semantically relevant corpus entries for query."""
        n = len(self._index)
        if n == 0:
            return []

        q_vec = self._embedder.embed([query])[0]
        # Cosine similarity — vectors are unit-normalised.
        scores: NDArray[np.float32] = np.dot(self._matrix, q_vec).astype(np.float32)

        if stream is not None:
            mask = np.array([r["stream"] == stream for r in self._index], dtype=np.bool_)
            scores = np.where(mask, scores, np.float32(-2.0)).astype(np.float32)

        candidate_k = min(k * 4, n)
        if candidate_k == n:
            top_idx: NDArray[np.intp] = np.argsort(scores)[::-1].astype(np.intp)
        else:
            part = np.argpartition(scores, -candidate_k)[-candidate_k:]
            top_idx = part[np.argsort(scores[part])[::-1]].astype(np.intp)

        if self._reranker is not None:
            previews = [self._index[int(i)]["text_preview"] for i in top_idx]
            rr_scores = self._reranker.rerank(query, previews)
            order = sorted(range(len(top_idx)), key=lambda j: rr_scores[j], reverse=True)
            top_idx = top_idx[np.array(order, dtype=np.intp)]

        results: list[dict[str, Any]] = []
        for i in top_idx:
            if len(results) >= k:
                break
            rec = dict(self._index[int(i)])
            if stream is not None and rec["stream"] != stream:
                continue
            rec["entry"] = self._load_entry(rec["stream"], rec["line_no"])
            results.append(rec)
        return results

    def _load_entry(self, stream: str, line_no: int) -> dict[str, Any]:
        rel = _STREAMS.get(stream, "")
        if not rel:
            return {}
        path = self._corpus.root / rel
        if not path.exists():
            return {}
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        if line_no >= len(lines):
            return {}
        result: dict[str, Any] = json.loads(lines[line_no])
        return result

    # ── Internal ──────────────────────────────────────────────────────────────

    def _append_vector(self, vec: NDArray[np.float32], meta: dict[str, Any]) -> None:
        # Write f32 row FIRST, then index line (crash-safety).
        with self._vec_path.open("ab") as f:
            f.write(vec.tobytes())
        with self._idx_path.open("a") as f:
            f.write(json.dumps(meta) + "\n")
        if self._matrix.shape[0] == 0:
            self._matrix = vec.reshape(1, -1)
        else:
            self._matrix = np.vstack([self._matrix, vec]).astype(np.float32)
        self._index.append(meta)

    def _write_meta(self) -> None:
        meta = {
            "embedder_name": self._embedder.name,
            "dim": self._embedder.dim,
            "count": len(self._index),
        }
        tmp = self._meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(self._meta_path)
