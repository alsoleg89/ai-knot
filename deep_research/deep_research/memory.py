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

    Crash-safety: f32 rows are written before index lines. On startup the f32
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
        self._matrix: NDArray[np.float32] = np.empty((0, self._embedder.dim), dtype=np.float32)
        self._index: list[dict[str, Any]] = []
        self._stream_counts: dict[str, int] = {}
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
        for rec in self._index:
            s = rec["stream"]
            self._stream_counts[s] = self._stream_counts.get(s, 0) + 1

    def _reset(self) -> None:
        self._vec_path.unlink(missing_ok=True)
        self._idx_path.unlink(missing_ok=True)
        self._meta_path.unlink(missing_ok=True)
        self._matrix = np.empty((0, self._embedder.dim), dtype=np.float32)
        self._index = []
        self._stream_counts = {}

    # ── Sync ─────────────────────────────────────────────────────────────────

    def sync(self) -> None:
        """Embed any corpus entries not yet in the index. Idempotent."""
        changed = False
        for stream, rel_path in _STREAMS.items():
            path = self._corpus.root / rel_path
            if not path.exists():
                continue
            all_lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
            already = self._stream_counts.get(stream, 0)
            new_lines = all_lines[already:]
            if not new_lines:
                continue
            new_entries = [json.loads(ln) for ln in new_lines]
            texts = [_entry_text(e) for e in new_entries]
            vecs = self._embedder.embed(texts)
            metas = [
                {
                    "stream": stream,
                    "line_no": already + i,
                    "tick": int(entry.get("tick", -1)),
                    "text_preview": texts[i][:80],
                }
                for i, entry in enumerate(new_entries)
            ]
            # Write f32 rows FIRST, then index lines (crash-safety).
            batch = vecs.astype(np.float32)
            with self._vec_path.open("ab") as f:
                f.write(batch.tobytes())
            with self._idx_path.open("a") as f:
                for meta in metas:
                    f.write(json.dumps(meta) + "\n")
            self._matrix = np.vstack([self._matrix, batch]).astype(np.float32)
            self._index.extend(metas)
            self._stream_counts[stream] = already + len(new_entries)
            changed = True
        if changed:
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

        # Collect result records (stream guard handles reranker reordering).
        result_recs: list[dict[str, Any]] = []
        for i in top_idx:
            if len(result_recs) >= k:
                break
            rec = dict(self._index[int(i)])
            if stream is not None and rec["stream"] != stream:
                continue
            result_recs.append(rec)

        # Load entries once per stream — one file read per stream, not per result.
        stream_lines: dict[str, list[str]] = {}
        for rec in result_recs:
            s = rec["stream"]
            if s not in stream_lines:
                rel = _STREAMS.get(s, "")
                if rel:
                    p = self._corpus.root / rel
                    stream_lines[s] = (
                        [ln for ln in p.read_text().splitlines() if ln.strip()]
                        if p.exists()
                        else []
                    )
                else:
                    stream_lines[s] = []
            lines = stream_lines[s]
            ln = rec["line_no"]
            rec["entry"] = json.loads(lines[ln]) if ln < len(lines) else {}

        return result_recs

    # ── Internal ──────────────────────────────────────────────────────────────

    def _write_meta(self) -> None:
        meta = {
            "embedder_name": self._embedder.name,
            "dim": self._embedder.dim,
            "count": len(self._index),
        }
        tmp = self._meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, indent=2))
        tmp.replace(self._meta_path)
