"""InvertedIndex — BM25F multi-field inverted index and trigram helpers."""

from __future__ import annotations

import math

from ai_knot.tokenizer import tokenize as _tokenize
from ai_knot.types import Fact

# BM25F parameters.
_BM25_K1: float = 1.5  # Term saturation parameter.
_BM25_B_CONTENT: float = 0.75  # Length normalization for content field.
_BM25_B_TAGS: float = 0.3  # Length normalization for tags field.
_BM25_B_CANONICAL: float = 0.5  # Length normalization for canonical_surface field.
_W_CONTENT: float = 1.0  # Content field weight.
_W_TAGS: float = 2.0  # Tags field weight (more specific, higher boost).
_W_CANONICAL: float = 1.5  # canonical_surface field weight (normalized paraphrase).
_BM25_B_EVIDENCE: float = 0.75  # Length normalization for evidence (source_snippets) field.
_W_EVIDENCE: float = 0.8  # Evidence field weight (lower than content to avoid over-boosting).

# IDF high-DF threshold: terms in >90% of docs get zero IDF weight.
# Raised from 0.7 to 0.9 so entity names (speakers, recurring names) that
# appear in many turns are not completely zeroed out.
_IDF_HIGH_DF_RATIO: float = 0.9

# PRF parameters.
_PRF_TOP_K: int = 3  # Number of feedback documents.
_PRF_EXPANSION_TERMS: int = 5  # Max expansion terms.
_PRF_ALPHA: float = 0.5  # PRF expansion term weight (statistical fallback).
# When LLM expansion is active, PRF is skipped entirely (see search()).

# RRF parameter.
_RRF_K: int = 60  # Rank smoothing constant (Cormack et al., 2009).


class InvertedIndex:
    """Pre-computed inverted index for BM25F scoring.

    Indexes both ``fact.content`` and ``fact.tags`` as separate fields,
    combining them via BM25F (Robertson, Zaragoza & Taylor 2004).
    Terms appearing in >70% of documents are auto-filtered (IDF zeroed).
    """

    def __init__(self, facts: list[Fact]) -> None:
        self._facts: dict[str, Fact] = {}  # id -> Fact
        self._content_lengths: dict[str, int] = {}  # id -> content token count
        self._tags_lengths: dict[str, int] = {}  # id -> tags token count
        self._canonical_lengths: dict[str, int] = {}  # id -> canonical_surface token count
        self._content_postings: dict[str, dict[str, int]] = {}  # term -> {id: tf}
        self._tags_postings: dict[str, dict[str, int]] = {}  # term -> {id: tf}
        self._canonical_postings: dict[str, dict[str, int]] = {}  # term -> {id: tf}
        self._content_trigrams: dict[str, frozenset[str]] = {}  # id -> trigrams
        self._canonical_trigrams: dict[str, frozenset[str]] = {}  # id -> trigrams
        self._evidence_lengths: dict[str, int] = {}  # id -> evidence token count
        self._evidence_postings: dict[str, dict[str, int]] = {}  # term -> {id: tf}
        self._evidence_trigrams: dict[str, frozenset[str]] = {}  # id -> trigrams
        self._slot_tokens: dict[str, frozenset[str]] = {}  # id -> tokenised slot_key
        self._doc_count: int = 0
        self._avg_content_dl: float = 0.0
        self._avg_tags_dl: float = 0.0
        self._avg_canonical_dl: float = 0.0
        self._avg_evidence_dl: float = 0.0
        self._build(facts)

    def _build(self, facts: list[Fact]) -> None:
        total_content_len = 0
        total_tags_len = 0
        total_canonical_len = 0
        total_evidence_len = 0

        for fact in facts:
            self._facts[fact.id] = fact

            # Content field.
            content_tokens = _tokenize(fact.content)
            self._content_lengths[fact.id] = len(content_tokens)
            total_content_len += len(content_tokens)

            content_tf: dict[str, int] = {}
            for token in content_tokens:
                content_tf[token] = content_tf.get(token, 0) + 1
            for term, tf in content_tf.items():
                if term not in self._content_postings:
                    self._content_postings[term] = {}
                self._content_postings[term][fact.id] = tf

            # Tags field.
            tags_text = " ".join(fact.tags)
            tags_tokens = _tokenize(tags_text) if tags_text else []
            self._tags_lengths[fact.id] = len(tags_tokens)
            total_tags_len += len(tags_tokens)

            tags_tf: dict[str, int] = {}
            for token in tags_tokens:
                tags_tf[token] = tags_tf.get(token, 0) + 1
            for term, tf in tags_tf.items():
                if term not in self._tags_postings:
                    self._tags_postings[term] = {}
                self._tags_postings[term][fact.id] = tf

            # Canonical surface field (LLM-normalised paraphrase, e.g. "person earns salary").
            canonical_tokens = _tokenize(fact.canonical_surface) if fact.canonical_surface else []
            self._canonical_lengths[fact.id] = len(canonical_tokens)
            total_canonical_len += len(canonical_tokens)

            canonical_tf: dict[str, int] = {}
            for token in canonical_tokens:
                canonical_tf[token] = canonical_tf.get(token, 0) + 1
            for term, tf in canonical_tf.items():
                if term not in self._canonical_postings:
                    self._canonical_postings[term] = {}
                self._canonical_postings[term][fact.id] = tf

            # Evidence field (joined source_snippets — original conversation turns).
            evidence_text = " ".join(fact.source_snippets) if fact.source_snippets else ""
            evidence_tokens = _tokenize(evidence_text) if evidence_text else []
            self._evidence_lengths[fact.id] = len(evidence_tokens)
            total_evidence_len += len(evidence_tokens)

            evidence_tf: dict[str, int] = {}
            for token in evidence_tokens:
                evidence_tf[token] = evidence_tf.get(token, 0) + 1
            for term, tf in evidence_tf.items():
                if term not in self._evidence_postings:
                    self._evidence_postings[term] = {}
                self._evidence_postings[term][fact.id] = tf

            # Precompute trigrams and slot tokens once at index-build time so
            # search() does not recompute them for every query.
            self._content_trigrams[fact.id] = _char_trigrams(fact.content)
            self._canonical_trigrams[fact.id] = (
                _char_trigrams(fact.canonical_surface) if fact.canonical_surface else frozenset()
            )
            self._evidence_trigrams[fact.id] = (
                _char_trigrams(evidence_text) if evidence_text else frozenset()
            )
            if fact.slot_key:
                slot_text = fact.slot_key.replace("::", " ")
                self._slot_tokens[fact.id] = frozenset(_tokenize(slot_text))
            else:
                self._slot_tokens[fact.id] = frozenset()

        self._doc_count = len(facts)
        self._avg_content_dl = total_content_len / self._doc_count if self._doc_count else 1.0
        self._avg_tags_dl = total_tags_len / self._doc_count if self._doc_count else 1.0
        self._avg_canonical_dl = total_canonical_len / self._doc_count if self._doc_count else 1.0
        self._avg_evidence_dl = total_evidence_len / self._doc_count if self._doc_count else 1.0

    def _combined_df(self, term: str) -> int:
        """Number of documents containing *term* in any field."""
        content_docs = set(self._content_postings.get(term, {}).keys())
        tags_docs = set(self._tags_postings.get(term, {}).keys())
        canonical_docs = set(self._canonical_postings.get(term, {}).keys())
        evidence_docs = set(self._evidence_postings.get(term, {}).keys())
        return len(content_docs | tags_docs | canonical_docs | evidence_docs)

    def score(
        self,
        query: str,
        *,
        k1: float = _BM25_K1,
        b: float = _BM25_B_CONTENT,
        expansion_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Return BM25F scores for all documents matching any query term.

        Args:
            query: Search query string.
            k1: BM25 saturation parameter.
            b: Length normalization parameter (used for content field).
            expansion_weights: Optional PRF expansion terms with weights.

        Returns:
            Dict mapping document id to BM25F score.
        """
        query_tokens = _tokenize(query)
        # Build term weights: original query terms get weight 1.0.
        term_weights: dict[str, float] = {}
        for t in query_tokens:
            term_weights[t] = 1.0

        # Merge expansion terms.
        if expansion_weights:
            for t, w in expansion_weights.items():
                if t not in term_weights:
                    term_weights[t] = w

        scores: dict[str, float] = {}
        n = self._doc_count
        high_df_threshold = _IDF_HIGH_DF_RATIO * n

        for term, q_weight in term_weights.items():
            df = self._combined_df(term)
            if not df:
                continue
            # IDF auto-filter: zero weight for terms in >70% of docs.
            # Only apply when corpus is large enough (>= 5 docs) to avoid
            # filtering legitimate terms in tiny document sets.
            if n >= 5 and df > high_df_threshold:
                continue
            idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)

            content_posting = self._content_postings.get(term, {})
            tags_posting = self._tags_postings.get(term, {})
            canonical_posting = self._canonical_postings.get(term, {})
            evidence_posting = self._evidence_postings.get(term, {})
            doc_ids = (
                set(content_posting.keys())
                | set(tags_posting.keys())
                | set(canonical_posting.keys())
                | set(evidence_posting.keys())
            )

            for doc_id in doc_ids:
                # BM25F: weighted sum of normalized tf across all four fields.
                tf_content = content_posting.get(doc_id, 0)
                dl_c = self._content_lengths[doc_id]
                norm_tf_c = tf_content / (1.0 + b * (dl_c / self._avg_content_dl - 1.0))

                tf_tags = tags_posting.get(doc_id, 0)
                dl_t = self._tags_lengths[doc_id]
                avg_t = self._avg_tags_dl if self._avg_tags_dl > 0 else 1.0
                norm_tf_t = tf_tags / (1.0 + _BM25_B_TAGS * (dl_t / avg_t - 1.0))

                tf_canonical = canonical_posting.get(doc_id, 0)
                dl_can = self._canonical_lengths[doc_id]
                avg_can = self._avg_canonical_dl if self._avg_canonical_dl > 0 else 1.0
                norm_tf_can = tf_canonical / (1.0 + _BM25_B_CANONICAL * (dl_can / avg_can - 1.0))

                tf_evidence = evidence_posting.get(doc_id, 0)
                dl_ev = self._evidence_lengths[doc_id]
                avg_ev = self._avg_evidence_dl if self._avg_evidence_dl > 0 else 1.0
                norm_tf_ev = tf_evidence / (1.0 + _BM25_B_EVIDENCE * (dl_ev / avg_ev - 1.0))

                tf_bm25f = (
                    _W_CONTENT * norm_tf_c
                    + _W_TAGS * norm_tf_t
                    + _W_CANONICAL * norm_tf_can
                    + _W_EVIDENCE * norm_tf_ev
                )
                tf_score = (k1 + 1.0) * tf_bm25f / (k1 + tf_bm25f)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf_score * q_weight

        return scores

    @property
    def facts(self) -> dict[str, Fact]:
        """Return the id -> Fact mapping."""
        return self._facts

    @property
    def content_lengths(self) -> dict[str, int]:
        """Return document content lengths for PRF."""
        return self._content_lengths

    @property
    def content_trigrams(self) -> dict[str, frozenset[str]]:
        """Precomputed char-trigrams for fact content (id → trigrams)."""
        return self._content_trigrams

    @property
    def canonical_trigrams(self) -> dict[str, frozenset[str]]:
        """Precomputed char-trigrams for canonical_surface (id → trigrams)."""
        return self._canonical_trigrams

    @property
    def evidence_trigrams(self) -> dict[str, frozenset[str]]:
        """Precomputed char-trigrams for evidence/source_snippets (id → trigrams)."""
        return self._evidence_trigrams

    @property
    def slot_tokens(self) -> dict[str, frozenset[str]]:
        """Precomputed tokenised slot_key (id → token set)."""
        return self._slot_tokens


def _char_trigrams(text: str) -> frozenset[str]:
    """Extract character-level trigrams from *text* (lowercased).

    Character trigrams bridge morphological variants not caught by stemming
    (e.g. "employer" / "employed" share "empl", "mpl", etc.).
    """
    t = text.lower()
    return frozenset(t[i : i + 3] for i in range(len(t) - 2)) if len(t) >= 3 else frozenset()


def _trigram_jaccard_against(qt: frozenset[str], text: str) -> float:
    """Jaccard similarity between precomputed query trigrams and *text* (0.0–1.0).

    Use this in hot paths where the query trigrams are already computed.
    """
    tt = _char_trigrams(text)
    if not qt or not tt:
        return 0.0
    return len(qt & tt) / len(qt | tt)


def _char_trigram_jaccard(query: str, text: str) -> float:
    """Char-trigram Jaccard similarity between *query* and *text* (0.0–1.0)."""
    return _trigram_jaccard_against(_char_trigrams(query), text)


def _slot_exact_score(query_tokens: frozenset[str], fact: Fact) -> float:
    """Fraction of slot address tokens covered by the query.

    ``slot_key`` is ``"{entity}::{attribute}"``; both parts are tokenized and
    matched against query tokens.  Returns 0.0 for facts with no slot.

    Example: query "Alex salary" with slot_key "Alex Chen::salary" →
    tokens {"alex", "chen", "salari"} → query coverage = 2/3 ≈ 0.67.
    """
    if not fact.slot_key:
        return 0.0
    slot_text = fact.slot_key.replace("::", " ")
    slot_tokens = frozenset(_tokenize(slot_text))
    if not slot_tokens:
        return 0.0
    return len(query_tokens & slot_tokens) / len(slot_tokens)
